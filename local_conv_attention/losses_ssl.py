"""Dense self-supervised losses for HEA latent pretraining."""

from __future__ import annotations

import torch
from torch import Tensor


def dense_invariance_loss(
    latents: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Dense LeJEPA invariance loss over aligned views.

    Args:
        latents: Tensor with shape [B, V, D, H, W].
        valid_mask: Optional tensor with shape [B, 1, H, W] or [B, H, W].
    """
    if latents.dim() != 5:
        raise ValueError(f"Expected latents with shape [B, V, D, H, W], got {tuple(latents.shape)}.")
    mean_latent = latents.mean(dim=1, keepdim=True)
    squared_error = (latents - mean_latent).square()
    if valid_mask is None:
        return squared_error.mean()
    if valid_mask.dim() == 3:
        valid_mask = valid_mask.unsqueeze(1)
    if valid_mask.dim() != 4:
        raise ValueError(f"valid_mask must have shape [B, 1, H, W] or [B, H, W], got {tuple(valid_mask.shape)}.")
    mask = valid_mask.unsqueeze(1).to(dtype=latents.dtype)
    numerator = (squared_error * mask).sum()
    denominator = mask.sum() * latents.size(1) * latents.size(2)
    return numerator / denominator.clamp_min(1.0)
