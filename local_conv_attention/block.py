"""Transformer-style blocks using exact local 2D attention."""

from __future__ import annotations

from torch import Tensor, nn

from .attention import AttentionImplementation, BoundaryPadMode, LocalSelfAttention2d
from .reference import ReferenceLocalSelfAttention2d
from .utils import ChannelLayerNorm2d, MLP2d


class LocalTransformerBlock2d(nn.Module):
    """Pre-norm transformer block with exact fixed-scaffold local attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        implementation: AttentionImplementation = "optimized",
        boundary_pad: BoundaryPadMode = "zeros",
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = ChannelLayerNorm2d(dim, eps=norm_eps)
        self.attn = LocalSelfAttention2d(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
            dilation=dilation,
            implementation=implementation,
            boundary_pad=boundary_pad,
        )
        self.norm2 = ChannelLayerNorm2d(dim, eps=norm_eps)
        self.mlp = MLP2d(dim=dim, hidden_dim=hidden_dim, bias=mlp_bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ReferenceLocalTransformerBlock2d(nn.Module):
    """Reference transformer block using flattened masked local attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
        dilation: int = 1,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = ChannelLayerNorm2d(dim, eps=norm_eps)
        self.attn = ReferenceLocalSelfAttention2d(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
            dilation=dilation,
        )
        self.norm2 = ChannelLayerNorm2d(dim, eps=norm_eps)
        self.mlp = MLP2d(dim=dim, hidden_dim=hidden_dim, bias=mlp_bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
