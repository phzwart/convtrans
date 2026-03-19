"""Target generation helpers for bottom-up instance segmentation."""

from __future__ import annotations

import torch
from torch import Tensor

from .config import CenterTargetConfig, InstanceTargetConfig, OffsetTargetConfig


def _as_target_config(config: InstanceTargetConfig | dict) -> InstanceTargetConfig:
    if isinstance(config, InstanceTargetConfig):
        return config
    center = config.get("center", {})
    offsets = config.get("offsets", {})
    return InstanceTargetConfig(
        center=CenterTargetConfig(**center),
        offsets=OffsetTargetConfig(**offsets),
    )


def build_foreground_target(instance_labels: Tensor) -> Tensor:
    """Return a binary foreground mask from an integer instance map."""
    return (instance_labels > 0).to(dtype=torch.float32)


def _instance_centers(instance_map: Tensor) -> dict[int, tuple[float, float]]:
    centers: dict[int, tuple[float, float]] = {}
    for instance_id in instance_map.unique(sorted=True).tolist():
        if instance_id <= 0:
            continue
        mask = instance_map == int(instance_id)
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.numel() == 0:
            continue
        center = coords.to(dtype=torch.float32).mean(dim=0)
        centers[int(instance_id)] = (float(center[0].item()), float(center[1].item()))
    return centers


def _draw_gaussian(
    heatmap: Tensor,
    center_y: float,
    center_x: float,
    *,
    sigma: float,
    radius: int,
) -> Tensor:
    height, width = heatmap.shape
    y0 = max(int(center_y) - radius, 0)
    y1 = min(int(center_y) + radius + 1, height)
    x0 = max(int(center_x) - radius, 0)
    x1 = min(int(center_x) + radius + 1, width)
    if y0 >= y1 or x0 >= x1:
        return heatmap

    y = torch.arange(y0, y1, device=heatmap.device, dtype=torch.float32)
    x = torch.arange(x0, x1, device=heatmap.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    gaussian = torch.exp(-((yy - center_y) ** 2 + (xx - center_x) ** 2) / (2 * sigma ** 2))
    heatmap[y0:y1, x0:x1] = torch.maximum(heatmap[y0:y1, x0:x1], gaussian)
    return heatmap


def build_center_heatmap_target(
    instance_labels: Tensor,
    config: CenterTargetConfig,
) -> Tensor:
    """Build a per-instance Gaussian center heatmap."""
    if instance_labels.ndim != 3:
        raise ValueError("instance_labels must have shape [B, H, W].")

    heatmaps = torch.zeros(
        (instance_labels.size(0), 1, instance_labels.size(1), instance_labels.size(2)),
        device=instance_labels.device,
        dtype=torch.float32,
    )
    for batch_index in range(instance_labels.size(0)):
        centers = _instance_centers(instance_labels[batch_index])
        for center_y, center_x in centers.values():
            _draw_gaussian(
                heatmaps[batch_index, 0],
                center_y,
                center_x,
                sigma=config.sigma,
                radius=config.radius,
            )
    return heatmaps


def build_offset_target(
    instance_labels: Tensor,
    *,
    normalize: bool = False,
) -> tuple[Tensor, Tensor]:
    """Build per-pixel offsets toward each instance center and a supervision mask."""
    if instance_labels.ndim != 3:
        raise ValueError("instance_labels must have shape [B, H, W].")

    batch, height, width = instance_labels.shape
    offsets = torch.zeros((batch, 2, height, width), device=instance_labels.device, dtype=torch.float32)
    weights = torch.zeros((batch, 1, height, width), device=instance_labels.device, dtype=torch.float32)

    for batch_index in range(batch):
        centers = _instance_centers(instance_labels[batch_index])
        for instance_id, (center_y, center_x) in centers.items():
            mask = instance_labels[batch_index] == instance_id
            coords = torch.nonzero(mask, as_tuple=False)
            if coords.numel() == 0:
                continue
            ys = coords[:, 0].to(dtype=torch.float32)
            xs = coords[:, 1].to(dtype=torch.float32)
            dy = center_y - ys
            dx = center_x - xs
            if normalize:
                dy = dy / max(height - 1, 1)
                dx = dx / max(width - 1, 1)
            offsets[batch_index, 0, mask] = dy
            offsets[batch_index, 1, mask] = dx
            weights[batch_index, 0, mask] = 1.0
    return offsets, weights


def build_instance_targets(
    instance_labels: Tensor,
    config: InstanceTargetConfig | dict,
    *,
    semantic_labels: Tensor | None = None,
) -> dict[str, Tensor]:
    """Build foreground, center, and offset targets from an integer instance map."""
    target_config = _as_target_config(config)
    target_config.validate()
    foreground = build_foreground_target(instance_labels).unsqueeze(1)
    center = build_center_heatmap_target(instance_labels, target_config.center)
    offsets, offset_weight = build_offset_target(
        instance_labels,
        normalize=target_config.offsets.normalize,
    )

    targets = {
        "foreground_target": foreground,
        "center_target": center,
        "offset_target": offsets,
        "offset_weight": offset_weight,
        "valid_mask": offset_weight,
        "instance_labels": instance_labels,
    }
    if semantic_labels is not None:
        targets["semantic_target"] = semantic_labels
    return targets
