"""Exact local self-attention built on fixed 2D shift scaffolds."""

from __future__ import annotations

from typing import Literal, Tuple, Type

import torch
from torch import Tensor, nn

from .ops import ConvShiftBank2d, NeighborhoodShift2d
from .reference import FlattenedMaskedLocalAttention2d, ReferenceLocalAttention2d
from .utils import merge_heads, reshape_heads, scaled_dot_product_scale


def _apply_local_attention_from_neighbors(
    q_heads: Tensor,
    k_neighbors: Tensor,
    v_neighbors: Tensor,
    valid_mask: Tensor,
    *,
    return_attention: bool = False,
) -> Tensor | Tuple[Tensor, Tensor]:
    """Apply local attention once neighbors are already aligned.

    Args:
        q_heads: [B, heads, d_head, H, W]
        k_neighbors: [B, heads, K, d_head, H, W]
        v_neighbors: [B, heads, K, d_head, H, W]
        valid_mask: [1, 1, K, H, W]
    """
    scores = (q_heads.unsqueeze(2) * k_neighbors).sum(dim=3)
    scores = scores * scaled_dot_product_scale(q_heads.size(2))
    scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
    attention = torch.softmax(scores, dim=2)
    output = (attention.unsqueeze(3) * v_neighbors).sum(dim=2)
    output = merge_heads(output)
    if return_attention:
        return output, attention
    return output


class ShiftLocalAttention2d(nn.Module):
    """Readable reference implementation using explicit pad-and-slice shifts."""

    def __init__(
        self,
        num_heads: int,
        window_size: int,
        *,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.dilation = dilation
        self.shift = NeighborhoodShift2d(window_size=window_size, dilation=dilation)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        q_heads = reshape_heads(q, self.num_heads)
        k_heads = reshape_heads(k, self.num_heads)
        v_heads = reshape_heads(v, self.num_heads)

        k_neighbors, valid_mask = self.shift(k_heads, return_mask=True)
        v_neighbors = self.shift(v_heads)
        return _apply_local_attention_from_neighbors(
            q_heads,
            k_neighbors,
            v_neighbors,
            valid_mask.squeeze(3),
            return_attention=return_attention,
        )

    @classmethod
    def from_qkv(
        cls,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        num_heads: int,
        window_size: int,
        dilation: int = 1,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """Convenience wrapper for one-off local attention calls."""
        module = cls(num_heads=num_heads, window_size=window_size, dilation=dilation)
        return module(q, k, v, return_attention=return_attention)


class ConvLocalAttention2d(nn.Module):
    """Optimized exact local attention using a grouped-conv shift bank."""

    def __init__(
        self,
        num_heads: int,
        window_size: int,
        *,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.dilation = dilation
        self.shift = ConvShiftBank2d(window_size=window_size, dilation=dilation)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        batch, channels, height, width = q.shape
        if k.shape != q.shape or v.shape != q.shape:
            raise ValueError("q, k, and v must have the same shape.")
        if channels % self.num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({self.num_heads})."
            )

        head_dim = channels // self.num_heads
        q_heads = reshape_heads(q, self.num_heads)

        # Extract K and V neighbors in one backend call by stacking along batch.
        kv_neighbors, valid_mask = self.shift(
            torch.cat([k.contiguous(), v.contiguous()], dim=0),
            return_mask=True,
        )
        k_neighbors, v_neighbors = kv_neighbors.split(batch, dim=0)

        k_neighbors = k_neighbors.contiguous().reshape(batch, self.num_heads, head_dim, -1, height, width)
        v_neighbors = v_neighbors.contiguous().reshape(batch, self.num_heads, head_dim, -1, height, width)
        k_neighbors = k_neighbors.permute(0, 1, 3, 2, 4, 5).contiguous()
        v_neighbors = v_neighbors.permute(0, 1, 3, 2, 4, 5).contiguous()

        return _apply_local_attention_from_neighbors(
            q_heads,
            k_neighbors,
            v_neighbors,
            valid_mask,
            return_attention=return_attention,
        )

    @classmethod
    def from_qkv(
        cls,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        num_heads: int,
        window_size: int,
        dilation: int = 1,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """Convenience wrapper for one-off local attention calls."""
        module = cls(num_heads=num_heads, window_size=window_size, dilation=dilation)
        return module(q, k, v, return_attention=return_attention)


class LocalAttention2d(ConvLocalAttention2d):
    """Default exact local attention implementation.

    This optimized path uses a grouped `conv2d` shift bank with fixed one-hot
    kernels. Use `ShiftLocalAttention2d` when you want the more explicit pad-and-
    slice reference implementation.
    """

    pass


class MultiHeadLocalAttention2d(LocalAttention2d):
    """Alias emphasizing multi-head usage."""

    pass


AttentionImplementation = Literal["optimized", "shift", "unfold", "flattened"]


def make_attention_impl(
    implementation: AttentionImplementation,
) -> Type[nn.Module]:
    """Resolve a named attention backend."""
    mapping: dict[AttentionImplementation, Type[nn.Module]] = {
        "optimized": LocalAttention2d,
        "shift": ShiftLocalAttention2d,
        "unfold": ReferenceLocalAttention2d,
        "flattened": FlattenedMaskedLocalAttention2d,
    }
    return mapping[implementation]


class LocalSelfAttention2d(nn.Module):
    """Self-attention with learned 1x1 QKV projections and fixed local mixing."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        *,
        qkv_bias: bool = True,
        out_bias: bool = True,
        dilation: int = 1,
        implementation: AttentionImplementation = "optimized",
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.implementation = implementation
        self.qkv = nn.Conv2d(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.attention = make_attention_impl(implementation)(
            num_heads=num_heads,
            window_size=window_size,
            dilation=dilation,
        )
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=out_bias)

    def forward(self, x: Tensor, *, return_attention: bool = False) -> Tensor | Tuple[Tensor, Tensor]:
        q, k, v = self.qkv(x).chunk(3, dim=1)
        attended = self.attention(q, k, v, return_attention=return_attention)
        if return_attention:
            output, attention = attended
            return self.out_proj(output), attention
        return self.out_proj(attended)


def local_attention_from_qkv(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    num_heads: int,
    window_size: int,
    dilation: int = 1,
    return_attention: bool = False,
    implementation: AttentionImplementation = "optimized",
) -> Tensor | Tuple[Tensor, Tensor]:
    """Functional-style entry point for exact local attention."""
    attention_cls = make_attention_impl(implementation)
    module = attention_cls(
        num_heads=num_heads,
        window_size=window_size,
        dilation=dilation,
    )
    return module(q, k, v, return_attention=return_attention)
