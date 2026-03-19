"""Mask utilities for local attention on 2D grids."""

from __future__ import annotations

import torch
from torch import Tensor

from .utils import make_offsets, validate_dilation, validate_window_size


def local_validity_mask(
    height: int,
    width: int,
    window_size: int,
    dilation: int = 1,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Return a [K, H, W] boolean mask for valid shifted neighbors."""
    validate_window_size(window_size)
    validate_dilation(dilation)

    ys = torch.arange(height, device=device)
    xs = torch.arange(width, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    masks = []
    for dy, dx in make_offsets(window_size, dilation=dilation):
        valid = (yy + dy >= 0) & (yy + dy < height) & (xx + dx >= 0) & (xx + dx < width)
        masks.append(valid)
    return torch.stack(masks, dim=0)


def flattened_local_attention_mask(
    height: int,
    width: int,
    window_size: int,
    dilation: int = 1,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Return a [L, L] boolean mask for flattened 2D local attention."""
    validate_window_size(window_size)
    validate_dilation(dilation)

    num_tokens = height * width
    mask = torch.zeros((num_tokens, num_tokens), dtype=torch.bool, device=device)

    for y in range(height):
        for x in range(width):
            query_index = y * width + x
            for dy, dx in make_offsets(window_size, dilation=dilation):
                yy = y + dy
                xx = x + dx
                if 0 <= yy < height and 0 <= xx < width:
                    key_index = yy * width + xx
                    mask[query_index, key_index] = True
    return mask
