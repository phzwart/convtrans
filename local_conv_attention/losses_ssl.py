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

    For each batch ``b``, channel ``d``, and pixel ``(h, w)``, this is the mean squared
    deviation of ``latents[b, :, d, h, w]`` from its mean **across views** ``V``.
    So the loss is **zero iff every view produces the same latent at every pixel**
    (perfect agreement across augmentations).

    **Interpretation:**
    - **~0** does *not* mean the map is spatially constant within a single view; it means
      **view 0, 1, …, V−1 all match** at each location. That is consistent with a rich
      spatial pattern that is **identical** for every augmentation (strong invariance).
    - If **input views differ** (pixels not identical) but this loss is ~0, the model may
      be **ignoring augmentations** (collapse) or views may be accidentally identical
      (check ``(views[:,0]-views[:,1]).abs().mean()``).
    - To measure **spatial** variation within one view, use ``latents[:,0].std(dim=(2,3))``
      or per-channel maps, not this loss.

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


def dense_lejepa_inv_diagnostics(latents: Tensor) -> dict[str, Tensor]:
    """Cheap stats to interpret :func:`dense_invariance_loss` (needs ``V >= 2``).

    If ``mean_std_across_views`` is ~0, latents match across views at almost all pixels
    (same as inv loss ~0). Compare with input view differences using your ``views`` tensor.
    """
    if latents.dim() != 5:
        raise ValueError(f"Expected [B, V, D, H, W], got {tuple(latents.shape)}.")
    if latents.size(1) < 2:
        raise ValueError("Need at least 2 views for diagnostics.")
    std_v = latents.float().std(dim=1)
    d01 = (latents[:, 0] - latents[:, 1]).abs().mean()
    return {
        "mean_std_across_views": std_v.mean(),
        "max_std_across_views": std_v.max(),
        "mean_abs_diff_latent_view0_vs_view1": d01,
    }
