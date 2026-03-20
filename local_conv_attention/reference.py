"""Reference implementations for local attention correctness checks."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .masks import flattened_local_attention_mask, local_validity_mask
from .ops import BoundaryPadMode, pad_spatial_hw
from .utils import merge_heads, reshape_heads, scaled_dot_product_scale, window_radius


class ReferenceLocalAttention2d(nn.Module):
    """Direct local attention reference using `torch.nn.functional.unfold`."""

    def __init__(
        self,
        num_heads: int,
        window_size: int,
        *,
        dilation: int = 1,
        boundary_pad: BoundaryPadMode = "zeros",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.dilation = dilation
        self.boundary_pad: BoundaryPadMode = boundary_pad
        self.padding = window_radius(window_size, dilation=dilation)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must have the same shape.")

        batch, channels, height, width = q.shape
        q_heads = reshape_heads(q, self.num_heads)
        head_dim = q_heads.size(2)
        neighborhood = self.window_size * self.window_size

        if self.boundary_pad == "zeros":
            k_in, v_in = k, v
            unfold_pad = self.padding
        else:
            k_in = pad_spatial_hw(k, self.padding, self.boundary_pad)
            v_in = pad_spatial_hw(v, self.padding, self.boundary_pad)
            unfold_pad = 0

        k_neighbors = F.unfold(
            k_in,
            kernel_size=self.window_size,
            dilation=self.dilation,
            padding=unfold_pad,
            stride=1,
        )
        v_neighbors = F.unfold(
            v_in,
            kernel_size=self.window_size,
            dilation=self.dilation,
            padding=unfold_pad,
            stride=1,
        )

        k_neighbors = k_neighbors.view(batch, self.num_heads, head_dim, neighborhood, height, width)
        v_neighbors = v_neighbors.view(batch, self.num_heads, head_dim, neighborhood, height, width)
        k_neighbors = k_neighbors.permute(0, 1, 3, 2, 4, 5)
        v_neighbors = v_neighbors.permute(0, 1, 3, 2, 4, 5)

        if self.boundary_pad == "zeros":
            valid_mask = local_validity_mask(
                height,
                width,
                self.window_size,
                dilation=self.dilation,
                device=q.device,
            ).view(1, 1, neighborhood, height, width)
        else:
            valid_mask = torch.ones(
                (1, 1, neighborhood, height, width),
                dtype=torch.bool,
                device=q.device,
            )

        scores = (q_heads.unsqueeze(2) * k_neighbors).sum(dim=3)
        scores = scores * scaled_dot_product_scale(head_dim)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=2)

        output = (attention.unsqueeze(3) * v_neighbors).sum(dim=2)
        output = merge_heads(output)
        if return_attention:
            return output, attention
        return output


class FlattenedMaskedLocalAttention2d(nn.Module):
    """Reference local attention via flattened transformer-style masked attention."""

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

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must have the same shape.")

        batch, _, height, width = q.shape
        q_heads = reshape_heads(q, self.num_heads)
        k_heads = reshape_heads(k, self.num_heads)
        v_heads = reshape_heads(v, self.num_heads)
        head_dim = q_heads.size(2)
        num_tokens = height * width

        q_seq = q_heads.permute(0, 1, 3, 4, 2).reshape(batch, self.num_heads, num_tokens, head_dim)
        k_seq = k_heads.permute(0, 1, 3, 4, 2).reshape(batch, self.num_heads, num_tokens, head_dim)
        v_seq = v_heads.permute(0, 1, 3, 4, 2).reshape(batch, self.num_heads, num_tokens, head_dim)

        scores = torch.matmul(q_seq, k_seq.transpose(-2, -1))
        scores = scores * scaled_dot_product_scale(head_dim)

        local_mask = flattened_local_attention_mask(
            height,
            width,
            self.window_size,
            dilation=self.dilation,
            device=q.device,
        )
        scores = scores.masked_fill(~local_mask.view(1, 1, num_tokens, num_tokens), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v_seq)
        output = output.view(batch, self.num_heads, height, width, head_dim).permute(0, 1, 4, 2, 3)
        output = merge_heads(output)

        if return_attention:
            return output, attention
        return output


class ReferenceLocalSelfAttention2d(nn.Module):
    """Self-attention layer backed by the flattened masked reference path."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        *,
        qkv_bias: bool = True,
        out_bias: bool = True,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Conv2d(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.attention = FlattenedMaskedLocalAttention2d(
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


def flattened_local_attention_from_qkv(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    num_heads: int,
    window_size: int,
    dilation: int = 1,
    return_attention: bool = False,
) -> Tensor | Tuple[Tensor, Tensor]:
    """Functional-style flattened local attention reference."""
    module = FlattenedMaskedLocalAttention2d(
        num_heads=num_heads,
        window_size=window_size,
        dilation=dilation,
    )
    return module(q, k, v, return_attention=return_attention)
