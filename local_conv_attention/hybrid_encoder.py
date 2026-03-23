"""Configurable hybrid Conv + local-attention encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn

from .attention import AttentionImplementation, BoundaryPadMode, LocalSelfAttention2d
from .utils import ChannelLayerNorm2d


OutputMode = Literal["feature_map", "pooled", "logits"]


@dataclass(frozen=True)
class ResidualStemConfig:
    """Configuration for the initial residual Conv2d stem."""

    in_channels: int = 3
    hidden_channels: int = 64
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    use_bias: bool = True

    def validate(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("stem.in_channels must be positive.")
        if self.hidden_channels <= 0:
            raise ValueError("stem.hidden_channels must be positive.")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("stem.kernel_size must be a positive odd integer.")
        if self.stride <= 0:
            raise ValueError("stem.stride must be positive.")
        if self.padding < 0:
            raise ValueError("stem.padding must be non-negative.")


@dataclass(frozen=True)
class HybridAttentionBlockConfig:
    """Configuration for one repeated [attn -> linear+gelu -> linear+gelu] block."""

    channels: int = 64
    num_heads: int = 4
    window_size: int = 3
    dilation: int = 1
    implementation: AttentionImplementation = "optimized"
    boundary_pad: BoundaryPadMode = "zeros"
    qkv_bias: bool = True
    out_bias: bool = True
    mlp_bias: bool = True
    hidden_channels: int = 256
    #: If True, apply channel LayerNorm before attention and before the MLP (same idea as
    #: :class:`~local_conv_attention.block.LocalTransformerBlock2d`). Recommended for
    #: stable features; if False, uses the legacy path (attention → MLP, no norm).
    use_pre_norm: bool = True
    norm_eps: float = 1e-5

    def validate(self) -> None:
        if self.channels <= 0:
            raise ValueError("block.channels must be positive.")
        if self.num_heads <= 0:
            raise ValueError("block.num_heads must be positive.")
        if self.channels % self.num_heads != 0:
            raise ValueError(
                f"block.channels ({self.channels}) must be divisible by block.num_heads ({self.num_heads})."
            )
        if self.window_size <= 0 or self.window_size % 2 == 0:
            raise ValueError("block.window_size must be a positive odd integer.")
        if self.dilation <= 0:
            raise ValueError("block.dilation must be positive.")
        if self.hidden_channels <= 0:
            raise ValueError("block.hidden_channels must be positive.")
        if self.norm_eps <= 0:
            raise ValueError("block.norm_eps must be positive.")


@dataclass(frozen=True)
class HybridConvAttentionEncoderConfig:
    """Top-level architecture configuration."""

    stem: ResidualStemConfig = ResidualStemConfig()
    block: HybridAttentionBlockConfig = HybridAttentionBlockConfig()
    depth: int = 4
    output_mode: OutputMode = "feature_map"
    num_classes: int | None = None
    #: If True, apply global average pooling after the block stack so activations are
    #: spatially flat ``[B, C, 1, 1]`` (one vector per channel per image). If False,
    #: keep full-resolution maps ``[B, C, H, W]``.
    global_avg_pool_features: bool = False

    def validate(self) -> None:
        self.stem.validate()
        self.block.validate()
        if self.depth <= 0:
            raise ValueError("depth must be positive.")
        if self.stem.hidden_channels != self.block.channels:
            raise ValueError(
                "stem.hidden_channels must match block.channels so the repeated blocks can run."
            )
        if self.output_mode == "logits" and (self.num_classes is None or self.num_classes <= 0):
            raise ValueError("num_classes must be a positive int when output_mode='logits'.")


class _LinearGELU2d(nn.Module):
    """1x1 linear projection followed by GELU."""

    def __init__(self, channels: int, out_channels: int, *, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, out_channels, kernel_size=1, bias=bias)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.proj(x))


class ResidualStem2d(nn.Module):
    """Conv2d + linear+gelu + linear+gelu with residual from stem input."""

    def __init__(self, config: ResidualStemConfig) -> None:
        super().__init__()
        self.config = config
        self.conv = nn.Conv2d(
            config.in_channels,
            config.hidden_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            bias=config.use_bias,
        )
        self.linear1 = _LinearGELU2d(config.hidden_channels, config.hidden_channels, bias=config.use_bias)
        self.linear2 = _LinearGELU2d(config.hidden_channels, config.hidden_channels, bias=config.use_bias)
        self.skip = nn.Conv2d(
            config.in_channels,
            config.hidden_channels,
            kernel_size=1,
            stride=config.stride,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        skip = self.skip(x)
        x = self.conv(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x + skip


class HybridAttentionBlock2d(nn.Module):
    """One block: local self-attention + linear+gelu + linear+gelu with residual.

    With ``use_pre_norm=True`` (default), uses pre-norm like
    :class:`~local_conv_attention.block.LocalTransformerBlock2d` for healthier gradients
    and less tendency toward degenerate (spatially constant) activations during training.
    """

    def __init__(self, config: HybridAttentionBlockConfig) -> None:
        super().__init__()
        self.use_pre_norm = config.use_pre_norm
        if self.use_pre_norm:
            self.norm1 = ChannelLayerNorm2d(config.channels, eps=config.norm_eps)
            self.norm2 = ChannelLayerNorm2d(config.channels, eps=config.norm_eps)
        else:
            self.norm1 = None  # type: ignore[assignment]
            self.norm2 = None  # type: ignore[assignment]
        self.attn = LocalSelfAttention2d(
            dim=config.channels,
            num_heads=config.num_heads,
            window_size=config.window_size,
            qkv_bias=config.qkv_bias,
            out_bias=config.out_bias,
            dilation=config.dilation,
            implementation=config.implementation,
            boundary_pad=config.boundary_pad,
        )
        self.linear1 = _LinearGELU2d(config.channels, config.hidden_channels, bias=config.mlp_bias)
        self.linear2 = _LinearGELU2d(config.hidden_channels, config.channels, bias=config.mlp_bias)

    def _mlp(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        return self.linear2(x)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_pre_norm:
            assert self.norm1 is not None and self.norm2 is not None
            x = x + self.attn(self.norm1(x))
            x = x + self._mlp(self.norm2(x))
            return x
        residual = x
        x = self.attn(x)
        x = self._mlp(x)
        return x + residual


class HybridConvAttentionEncoder(nn.Module):
    """Configurable encoder matching the requested stem and repeated local-attn blocks."""

    def __init__(self, config: HybridConvAttentionEncoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.stem = ResidualStem2d(config.stem)
        self.blocks = nn.Sequential(*[HybridAttentionBlock2d(config.block) for _ in range(config.depth)])
        if config.output_mode == "logits":
            self.head = nn.Conv2d(config.block.channels, int(config.num_classes), kernel_size=1)
        else:
            self.head = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self._gap_after_blocks = bool(config.global_avg_pool_features)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        if self._gap_after_blocks:
            x = self.pool(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        output = self.head(features)
        if self.config.output_mode == "feature_map":
            return output
        if self.config.output_mode == "pooled":
            return self.pool(output).flatten(1)
        return output
