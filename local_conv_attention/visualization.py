"""Visualization helpers for HEA region-level explanations."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor


def _to_numpy_image(image: Tensor | np.ndarray) -> np.ndarray:
    """Convert a grayscale or RGB tensor/image into an HWC numpy array."""
    if isinstance(image, Tensor):
        image = image.detach().cpu().float()
        if image.dim() == 3 and image.size(0) in {1, 3}:
            image = image.permute(1, 2, 0).numpy()
        elif image.dim() == 2:
            image = image.numpy()
        else:
            raise ValueError(f"Unsupported image tensor shape {tuple(image.shape)}.")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in {1, 3}:
        return image
    raise ValueError(f"Unsupported image array shape {tuple(image.shape)}.")


def upsample_region_heatmap(
    heatmap: Tensor,
    *,
    scale_factor: int,
    output_shape: tuple[int, int],
) -> Tensor:
    """Nearest-repeat a coarse region heatmap back to image space."""
    if heatmap.dim() != 2:
        raise ValueError(f"heatmap must be 2D, got shape {tuple(heatmap.shape)}.")
    expanded = heatmap.repeat_interleave(scale_factor, dim=0).repeat_interleave(scale_factor, dim=1)
    return expanded[: output_shape[0], : output_shape[1]]


def combine_upsampled_heatmaps(
    heatmaps: Mapping[int, Tensor],
    *,
    output_shape: tuple[int, int],
) -> Tensor:
    """Sum per-scale heatmaps after upsampling them into image space."""
    combined = None
    for scale_factor, heatmap in heatmaps.items():
        upsampled = upsample_region_heatmap(
            heatmap,
            scale_factor=scale_factor,
            output_shape=output_shape,
        )
        combined = upsampled if combined is None else combined + upsampled
    if combined is None:
        return torch.zeros(output_shape)
    return combined


def overlay_heatmap_on_image(
    image: Tensor | np.ndarray,
    heatmap: Tensor | np.ndarray,
    *,
    alpha: float = 0.45,
    cmap: str = "magma",
) -> np.ndarray:
    """Overlay a heatmap on top of an image for quick qualitative inspection."""
    image_np = _to_numpy_image(image)
    if isinstance(heatmap, Tensor):
        heatmap = heatmap.detach().cpu().float().numpy()
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got shape {tuple(heatmap.shape)}.")

    if image_np.ndim == 2:
        base = np.stack([image_np] * 3, axis=-1)
    else:
        base = image_np[..., :3]
    base = base.astype(np.float32)
    if base.max() > 1.0:
        base = base / 255.0

    heat = heatmap.astype(np.float32)
    if np.allclose(heat.max(), heat.min()):
        normalized = np.zeros_like(heat)
    else:
        normalized = (heat - heat.min()) / (heat.max() - heat.min())
    colors = plt.get_cmap(cmap)(normalized)[..., :3].astype(np.float32)
    return np.clip((1.0 - alpha) * base + alpha * colors, 0.0, 1.0)


def visualize_explanation(
    image: Tensor | np.ndarray,
    explanation: Mapping[str, object],
    *,
    target_xy: tuple[int, int] | None = None,
    show_combined: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Render a HEA explanation with one panel per scale and an optional overlay."""
    per_scale_heatmaps = explanation["per_scale_heatmaps"]
    ordered_scales = sorted(per_scale_heatmaps)
    num_panels = len(ordered_scales) + 1 + int(show_combined)
    figsize = figsize or (4.0 * num_panels, 4.0)
    fig, axes = plt.subplots(1, num_panels, figsize=figsize)
    axes = np.atleast_1d(axes)

    image_np = _to_numpy_image(image)
    axes[0].imshow(image_np if image_np.ndim == 3 else image_np, cmap=None if image_np.ndim == 3 else "gray")
    axes[0].set_title("Input")
    if target_xy is not None:
        axes[0].scatter([target_xy[1]], [target_xy[0]], c="cyan", s=30)
    axes[0].axis("off")

    for axis, scale in zip(axes[1:], ordered_scales):
        heatmap = per_scale_heatmaps[scale]
        if isinstance(heatmap, Tensor):
            heatmap = heatmap.detach().cpu().float().numpy()
        axis.imshow(heatmap, cmap="magma")
        axis.set_title(f"Scale {scale}")
        axis.axis("off")

    if show_combined:
        combined_axis = axes[-1]
        combined_heatmap = explanation["combined_heatmap"]
        combined_axis.imshow(
            overlay_heatmap_on_image(image_np, combined_heatmap),
            cmap=None,
        )
        combined_axis.set_title("Combined")
        combined_axis.axis("off")

    fig.tight_layout()
    return fig


def visualize_signed_explanation(
    image: Tensor | np.ndarray,
    positive_heatmap: Tensor,
    negative_heatmap: Tensor,
    *,
    target_xy: tuple[int, int] | None = None,
    figsize: tuple[float, float] = (12.0, 4.0),
) -> Figure:
    """Render separate positive and negative signed contribution overlays."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    image_np = _to_numpy_image(image)

    axes[0].imshow(image_np if image_np.ndim == 3 else image_np, cmap=None if image_np.ndim == 3 else "gray")
    if target_xy is not None:
        axes[0].scatter([target_xy[1]], [target_xy[0]], c="cyan", s=30)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(overlay_heatmap_on_image(image_np, positive_heatmap, cmap="Reds"), cmap=None)
    axes[1].set_title("Positive")
    axes[1].axis("off")

    axes[2].imshow(overlay_heatmap_on_image(image_np, negative_heatmap, cmap="Blues"), cmap=None)
    axes[2].set_title("Negative")
    axes[2].axis("off")

    fig.tight_layout()
    return fig
