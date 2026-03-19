"""Utility helpers for local convolutional attention."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
from torch import Tensor, nn


Offset2d = Tuple[int, int]


def validate_window_size(window_size: int) -> None:
    """Validate that the window size is an odd positive integer."""
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}.")


def validate_dilation(dilation: int) -> None:
    """Validate a positive dilation factor."""
    if dilation <= 0:
        raise ValueError(f"dilation must be positive, got {dilation}.")


def window_radius(window_size: int, dilation: int = 1) -> int:
    """Return the padded radius in pixels for a local window."""
    validate_window_size(window_size)
    validate_dilation(dilation)
    return (window_size // 2) * dilation


def make_offsets(window_size: int, dilation: int = 1) -> List[Offset2d]:
    """Return row-major 2D offsets for an odd local window."""
    radius = window_size // 2
    validate_window_size(window_size)
    validate_dilation(dilation)
    return [
        (dy * dilation, dx * dilation)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]


def reshape_heads(x: Tensor, num_heads: int) -> Tensor:
    """Reshape an NCHW tensor into [B, heads, head_dim, H, W]."""
    if x.dim() != 4:
        raise ValueError(f"Expected a 4D NCHW tensor, got shape {tuple(x.shape)}.")
    x = x.contiguous()
    batch, channels, height, width = x.shape
    if channels % num_heads != 0:
        raise ValueError(
            f"channels ({channels}) must be divisible by num_heads ({num_heads})."
        )
    head_dim = channels // num_heads
    return x.reshape(batch, num_heads, head_dim, height, width)


def merge_heads(x: Tensor) -> Tensor:
    """Merge a [B, heads, head_dim, H, W] tensor back to NCHW."""
    if x.dim() != 5:
        raise ValueError(f"Expected a 5D tensor, got shape {tuple(x.shape)}.")
    x = x.contiguous()
    batch, num_heads, head_dim, height, width = x.shape
    return x.reshape(batch, num_heads * head_dim, height, width)


def attention_tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    """Return recommended equivalence tolerances for the given dtype."""
    if dtype == torch.float64:
        return 1e-10, 1e-8
    if dtype == torch.float32:
        return 1e-5, 1e-4
    raise ValueError(f"Unsupported test dtype: {dtype}.")


def format_shape(shape: Sequence[int]) -> str:
    """Pretty-print a tensor shape."""
    return "x".join(str(part) for part in shape)


class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW tensors via explicit NHWC permutation."""

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class MLP2d(nn.Module):
    """A simple transformer-style MLP applied independently at each spatial location."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


def scaled_dot_product_scale(head_dim: int) -> float:
    """Return the standard attention scaling factor."""
    return 1.0 / math.sqrt(head_dim)
