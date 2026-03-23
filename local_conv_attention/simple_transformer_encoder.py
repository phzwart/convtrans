"""N × LocalTransformerBlock2d stack: conv stem + repeated pre-norm transformer blocks."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .attention import AttentionImplementation, BoundaryPadMode
from .block import LocalTransformerBlock2d


@dataclass(frozen=True)
class SimpleTransformerEncoderConfig:
    """A shallow conv stem then ``num_blocks`` :class:`LocalTransformerBlock2d` layers."""

    in_channels: int = 1
    dim: int = 64
    num_blocks: int = 4
    num_heads: int = 4
    window_size: int = 7
    mlp_ratio: float = 4.0
    dilation: int = 1
    implementation: AttentionImplementation = "optimized"
    boundary_pad: BoundaryPadMode = "zeros"
    stem_kernel_size: int = 3
    stem_stride: int = 1
    stem_padding: int | None = None
    qkv_bias: bool = True
    out_bias: bool = True
    mlp_bias: bool = True
    norm_eps: float = 1e-5

    def validate(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("encoder.in_channels must be positive.")
        if self.dim <= 0:
            raise ValueError("encoder.dim must be positive.")
        if self.dim % self.num_heads != 0:
            raise ValueError(f"encoder.dim ({self.dim}) must be divisible by num_heads ({self.num_heads}).")
        if self.num_blocks <= 0:
            raise ValueError("encoder.num_blocks must be positive.")
        if self.num_heads <= 0:
            raise ValueError("encoder.num_heads must be positive.")
        if self.window_size <= 0 or self.window_size % 2 == 0:
            raise ValueError("encoder.window_size must be a positive odd integer.")
        if self.stem_kernel_size <= 0 or self.stem_kernel_size % 2 == 0:
            raise ValueError("encoder.stem_kernel_size must be a positive odd integer.")
        if self.stem_stride <= 0:
            raise ValueError("encoder.stem_stride must be positive.")
        if self.stem_padding is not None and self.stem_padding < 0:
            raise ValueError("encoder.stem_padding must be non-negative when set.")


class SimpleTransformerEncoder2d(nn.Module):
    """Conv stem to ``dim`` channels, then ``num_blocks`` local transformer blocks (2D grid)."""

    def __init__(self, config: SimpleTransformerEncoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        pad = config.stem_padding
        if pad is None:
            if config.stem_stride != 1:
                raise ValueError(
                    "Set encoder.stem_padding explicitly when stem_stride != 1 "
                    "(no default same-size padding for arbitrary stride)."
                )
            pad = config.stem_kernel_size // 2
        self.stem = nn.Conv2d(
            config.in_channels,
            config.dim,
            kernel_size=config.stem_kernel_size,
            stride=config.stem_stride,
            padding=pad,
            bias=True,
        )
        self.blocks = nn.ModuleList(
            [
                LocalTransformerBlock2d(
                    dim=config.dim,
                    num_heads=config.num_heads,
                    window_size=config.window_size,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    out_bias=config.out_bias,
                    mlp_bias=config.mlp_bias,
                    dilation=config.dilation,
                    norm_eps=config.norm_eps,
                    implementation=config.implementation,
                    boundary_pad=config.boundary_pad,
                )
                for _ in range(config.num_blocks)
            ]
        )

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)
