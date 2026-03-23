"""Visualization helpers for HEA region-level explanations."""

from __future__ import annotations

import math
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


def _slice_latent_map(
    latents: Tensor | np.ndarray,
    *,
    batch_index: int,
    view_index: int | None,
) -> Tensor:
    """Return ``[D, H, W]`` from ``[D,H,W]``, ``[B,D,H,W]``, or ``[B,V,D,H,W]``."""
    if isinstance(latents, np.ndarray):
        x = torch.from_numpy(np.asarray(latents, dtype=np.float32))
    else:
        x = latents.detach().cpu().float()
    if x.dim() == 3:
        return x
    if x.dim() == 4:
        if batch_index < 0 or batch_index >= x.size(0):
            raise IndexError(f"batch_index {batch_index} out of range for size {x.size(0)}.")
        return x[batch_index]
    if x.dim() == 5:
        if batch_index < 0 or batch_index >= x.size(0):
            raise IndexError(f"batch_index {batch_index} out of range for size {x.size(0)}.")
        v = 0 if view_index is None else view_index
        if v < 0 or v >= x.size(1):
            raise IndexError(f"view_index {v} out of range for {x.size(1)} views.")
        return x[batch_index, v]
    raise ValueError(
        "latents must be [D,H,W], [B,D,H,W], or [B,V,D,H,W]; "
        f"got dim={x.dim()} shape={tuple(x.shape)}."
    )


def plot_latent_channels(
    latents: Tensor | np.ndarray,
    *,
    batch_index: int = 0,
    view_index: int | None = None,
    max_cols: int = 8,
    cmap: str = "magma",
    per_channel_norm: bool = True,
    global_norm: bool = False,
    figsize_per_axis: float = 1.35,
    suptitle: str | None = None,
) -> Figure:
    """Plot every latent channel as its own heatmap (dense LeJEPA / projector output).

    Args:
        latents: Tensor or array, one of:

            - ``[D, H, W]``
            - ``[B, D, H, W]`` (uses ``batch_index``)
            - ``[B, V, D, H, W]`` (uses ``batch_index`` and ``view_index``, default view 0)

        batch_index: Which batch element when ``B`` is present.
        view_index: Which view when ``V`` is present; default ``0`` if ``view_index`` is None.
        max_cols: Grid columns before wrapping to the next row.
        cmap: Matplotlib colormap name.
        per_channel_norm: If True (and not ``global_norm``), scale each channel to
            ``[0, 1]`` by its own min/max so weak channels stay visible.
        global_norm: If True, use one min/max over all channels (overrides ``per_channel_norm``).
        figsize_per_axis: Approximate inch size per subplot cell.
        suptitle: Optional figure title.

    Returns:
        A :class:`matplotlib.figure.Figure` with ``D`` subplots. Call ``plt.show()`` or save as needed.
    """
    z = _slice_latent_map(latents, batch_index=batch_index, view_index=view_index)
    if z.dim() != 3:
        raise ValueError(f"Internal slice must be [D,H,W], got {tuple(z.shape)}.")
    depth, height, width = int(z.size(0)), int(z.size(1)), int(z.size(2))
    if depth == 0:
        raise ValueError("latent depth D must be positive.")

    ncols = min(max_cols, depth)
    nrows = int(math.ceil(depth / ncols))
    fig_w = figsize_per_axis * ncols
    fig_h = figsize_per_axis * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    arr = z.numpy()
    if global_norm:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            lo, hi = lo, lo + 1.0
    else:
        lo = hi = None

    for d in range(depth):
        r, c = divmod(d, ncols)
        ax = axes[r][c]
        ch = arr[d]
        if global_norm:
            norm = (ch - lo) / (hi - lo)
            norm = np.clip(norm, 0.0, 1.0)
            ax.imshow(norm, cmap=cmap, vmin=0.0, vmax=1.0)
        elif per_channel_norm:
            cmin, cmax = float(ch.min()), float(ch.max())
            if cmax <= cmin:
                ax.imshow(np.zeros_like(ch), cmap=cmap, vmin=0.0, vmax=1.0)
            else:
                ax.imshow((ch - cmin) / (cmax - cmin), cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            ax.imshow(ch, cmap=cmap)
        ax.set_title(f"d={d}")
        ax.axis("off")

    for d in range(depth, nrows * ncols):
        r, c = divmod(d, ncols)
        axes[r][c].axis("off")
        axes[r][c].set_visible(False)

    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig
