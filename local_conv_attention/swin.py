"""A compact Swin-style U-Net baseline for segmentation comparison."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import HEAUNetModelConfig
from .encoder import ActKind, NormKind, make_activation


def _channel_last_norm(x: Tensor, norm: nn.LayerNorm) -> Tensor:
    x = x.permute(0, 2, 3, 1)
    x = norm(x)
    return x.permute(0, 3, 1, 2)


def window_partition(x: Tensor, window_size: int) -> tuple[Tensor, tuple[int, int, int, int]]:
    """Partition [B, H, W, C] into windows with padding as needed."""
    batch, height, width, channels = x.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        x = x.permute(0, 2, 3, 1)

    padded_h = height + pad_h
    padded_w = width + pad_w
    x = x.view(
        batch,
        padded_h // window_size,
        window_size,
        padded_w // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, channels)
    return windows, (height, width, padded_h, padded_w)


def window_reverse(
    windows: Tensor,
    window_size: int,
    meta: tuple[int, int, int, int],
    batch: int,
    channels: int,
) -> Tensor:
    """Reverse window partition back to [B, H, W, C]."""
    height, width, padded_h, padded_w = meta
    x = windows.view(
        batch,
        padded_h // window_size,
        padded_w // window_size,
        window_size,
        window_size,
        channels,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, padded_h, padded_w, channels)
    return x[:, :height, :width, :]


class WindowAttention2d(nn.Module):
    """Standard window attention over non-overlapping 2D windows."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        window_size: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: Tensor) -> Tensor:
        batch_windows, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).reshape(batch_windows, num_tokens, channels)
        return self.proj(output)


class SwinMLP(nn.Module):
    """A simple MLP block on channel-last windowed tokens."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, act: ActKind = "gelu") -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = make_activation(act)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwinBlock2d(nn.Module):
    """A basic shifted-window transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_eps: float = 1e-5,
        act: ActKind = "gelu",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = WindowAttention2d(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
        )
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = SwinMLP(dim=dim, mlp_ratio=mlp_ratio, act=act)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        x_ch_last = x.permute(0, 2, 3, 1)
        residual = x_ch_last
        x_ch_last = self.norm1(x_ch_last)

        if self.shift_size > 0:
            x_ch_last = torch.roll(x_ch_last, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows, meta = window_partition(x_ch_last, self.window_size)
        attended = self.attn(windows)
        x_ch_last = window_reverse(attended, self.window_size, meta, batch, channels)

        if self.shift_size > 0:
            x_ch_last = torch.roll(x_ch_last, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x_ch_last = residual + x_ch_last
        x_ch_last = x_ch_last + self.mlp(self.norm2(x_ch_last))
        return x_ch_last.permute(0, 3, 1, 2).contiguous()


class SwinStage2d(nn.Module):
    """A stack of Swin blocks with alternating shifted windows."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        *,
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        act: ActKind,
    ) -> None:
        super().__init__()
        blocks = []
        for index in range(depth):
            shift_size = 0 if index % 2 == 0 else window_size // 2
            blocks.append(
                SwinBlock2d(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    act=act,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class SwinDownsample2d(nn.Module):
    """A simple stride-2 patch-merging surrogate."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class SwinDecoderStage2d(nn.Module):
    """Upsample, concatenate skip, and refine with Swin blocks."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        act: ActKind,
    ) -> None:
        super().__init__()
        self.up_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.skip_proj = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1)
        self.refine = SwinStage2d(
            dim=out_channels,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            act=act,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up_proj(x)
        x = torch.cat([x, skip], dim=1)
        x = self.skip_proj(x)
        return self.refine(x)


class SwinUNet(nn.Module):
    """A compact Swin-style U-Net baseline."""

    def __init__(self, config: HEAUNetModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        channels = [config.base_channels * mult for mult in config.channel_multipliers]
        self.channels = channels
        self.stem = nn.Conv2d(
            config.in_channels,
            channels[0],
            kernel_size=config.patch_size,
            stride=1,
            padding=config.patch_size // 2,
        )

        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList([nn.Identity()])
        self.encoder_stages.append(
            SwinStage2d(
                dim=channels[0],
                depth=config.encoder_depths[0],
                num_heads=config.swin_stage_heads[0],
                window_size=config.swin_window_size,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.attention.qkv_bias,
                act=config.act,
            )
        )
        for scale in range(1, len(channels)):
            self.downsamples.append(SwinDownsample2d(channels[scale - 1], channels[scale]))
            self.encoder_stages.append(
                SwinStage2d(
                    dim=channels[scale],
                    depth=config.encoder_depths[scale],
                    num_heads=config.swin_stage_heads[scale],
                    window_size=config.swin_window_size,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.attention.qkv_bias,
                    act=config.act,
                )
            )

        self.bottleneck = SwinStage2d(
            dim=channels[-1],
            depth=max(1, config.bottleneck_depth),
            num_heads=config.swin_stage_heads[-1],
            window_size=config.swin_window_size,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.attention.qkv_bias,
            act=config.act,
        )

        self.decoder_stages = nn.ModuleDict()
        for target_scale in reversed(range(len(channels) - 1)):
            decoder_depth_index = len(channels) - 2 - target_scale
            self.decoder_stages[str(target_scale)] = SwinDecoderStage2d(
                in_channels=channels[target_scale + 1],
                skip_channels=channels[target_scale],
                out_channels=channels[target_scale],
                depth=config.decoder_depths[decoder_depth_index],
                num_heads=config.swin_stage_heads[target_scale],
                window_size=config.swin_window_size,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.attention.qkv_bias,
                act=config.act,
            )

        self.segmentation_head = nn.Conv2d(channels[0], config.num_classes, kernel_size=1)

    def forward_features(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        x = self.stem(x)
        encoder_features = []
        for scale, stage in enumerate(self.encoder_stages):
            x = self.downsamples[scale](x)
            x = stage(x)
            encoder_features.append(x)

        x = self.bottleneck(encoder_features[-1])
        for target_scale in reversed(range(len(self.channels) - 1)):
            x = self.decoder_stages[str(target_scale)](x, encoder_features[target_scale])
        return x, encoder_features

    def forward(self, x: Tensor) -> Tensor:
        features, _ = self.forward_features(x)
        return self.segmentation_head(features)
