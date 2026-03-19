"""Deterministic postprocessing for bottom-up instance predictions."""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import InstancePostprocessConfig


def _as_postprocess_config(config: InstancePostprocessConfig | dict) -> InstancePostprocessConfig:
    if isinstance(config, InstancePostprocessConfig):
        return config
    return InstancePostprocessConfig(**config)


def _connected_components(mask: Tensor) -> Tensor:
    mask_cpu = mask.to(device="cpu", dtype=torch.bool)
    height, width = mask_cpu.shape
    labels = torch.zeros((height, width), dtype=torch.int64)
    current_label = 0
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(height):
        for x in range(width):
            if not mask_cpu[y, x] or labels[y, x] != 0:
                continue
            current_label += 1
            queue: deque[tuple[int, int]] = deque([(y, x)])
            labels[y, x] = current_label
            while queue:
                cy, cx = queue.popleft()
                for dy, dx in neighbors:
                    ny = cy + dy
                    nx = cx + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if not mask_cpu[ny, nx] or labels[ny, nx] != 0:
                        continue
                    labels[ny, nx] = current_label
                    queue.append((ny, nx))
    return labels.to(device=mask.device)


def _prune_small_instances(instance_map: Tensor, min_area: int) -> tuple[Tensor, Tensor]:
    if min_area <= 0:
        ids = torch.unique(instance_map)
        return instance_map, ids[ids > 0]

    relabeled = torch.zeros_like(instance_map)
    kept_ids: list[int] = []
    next_id = 1
    for instance_id in torch.unique(instance_map, sorted=True).tolist():
        if instance_id <= 0:
            continue
        mask = instance_map == int(instance_id)
        if int(mask.sum().item()) < min_area:
            continue
        relabeled[mask] = next_id
        kept_ids.append(int(instance_id))
        next_id += 1
    return relabeled, torch.tensor(kept_ids, device=instance_map.device, dtype=torch.long)


def _denormalize_offsets(offsets: Tensor, *, height: int, width: int, normalized: bool) -> Tensor:
    if not normalized:
        return offsets
    denormalized = offsets.clone()
    denormalized[:, 0] = denormalized[:, 0] * max(height - 1, 1)
    denormalized[:, 1] = denormalized[:, 1] * max(width - 1, 1)
    return denormalized


def decode_instances(
    predictions: dict[str, Tensor],
    config: InstancePostprocessConfig | dict,
) -> dict[str, Tensor | list[Tensor]]:
    """Decode dense predictions into integer instance maps."""
    postprocess_config = _as_postprocess_config(config)
    postprocess_config.validate()

    foreground_logits = predictions["foreground_logits"]
    center_logits = (
        predictions["center_logits"]
        if "center_logits" in predictions
        else predictions["center_heatmap_logits"]
    )
    offsets = predictions["offsets"]

    foreground_probs = torch.sigmoid(foreground_logits)
    center_probs = torch.sigmoid(center_logits)
    batch, _, height, width = foreground_probs.shape
    offsets = _denormalize_offsets(
        offsets,
        height=height,
        width=width,
        normalized=postprocess_config.offsets_normalized,
    )

    peak_values = F.max_pool2d(
        center_probs,
        kernel_size=postprocess_config.nms_kernel_size,
        stride=1,
        padding=postprocess_config.nms_kernel_size // 2,
    )
    peak_mask = (center_probs >= postprocess_config.center_threshold) & (center_probs == peak_values)

    instance_maps: list[Tensor] = []
    center_points: list[Tensor] = []
    for batch_index in range(batch):
        foreground_mask = foreground_probs[batch_index, 0] >= postprocess_config.foreground_threshold
        if postprocess_config.method == "connected_components":
            instance_map = _connected_components(foreground_mask)
            instance_map, _ = _prune_small_instances(instance_map, postprocess_config.min_instance_area)
            instance_maps.append(instance_map)
            kept_centers = torch.zeros((0, 2), device=foreground_mask.device, dtype=torch.float32)
            center_points.append(kept_centers)
            continue

        if not foreground_mask.any():
            instance_maps.append(torch.zeros((height, width), device=foreground_mask.device, dtype=torch.int64))
            center_points.append(torch.zeros((0, 2), device=foreground_mask.device, dtype=torch.float32))
            continue

        detected_centers = torch.nonzero(peak_mask[batch_index, 0], as_tuple=False).to(dtype=torch.float32)
        if detected_centers.numel() == 0:
            instance_maps.append(torch.zeros((height, width), device=foreground_mask.device, dtype=torch.int64))
            center_points.append(torch.zeros((0, 2), device=foreground_mask.device, dtype=torch.float32))
            continue

        coords = torch.nonzero(foreground_mask, as_tuple=False)
        predicted_centers = coords.to(dtype=torch.float32) + offsets[batch_index, :, foreground_mask].T
        distances = torch.cdist(predicted_centers, detected_centers)
        nearest_distance, nearest_index = distances.min(dim=1)

        valid_assignment = nearest_distance <= postprocess_config.max_assignment_distance
        instance_map = torch.zeros((height, width), device=foreground_mask.device, dtype=torch.int64)
        if valid_assignment.any():
            assigned_coords = coords[valid_assignment]
            assigned_labels = nearest_index[valid_assignment] + 1
            instance_map[assigned_coords[:, 0], assigned_coords[:, 1]] = assigned_labels.to(dtype=torch.int64)

        instance_map, kept_ids = _prune_small_instances(instance_map, postprocess_config.min_instance_area)
        if kept_ids.numel() == 0:
            kept_centers = torch.zeros((0, 2), device=foreground_mask.device, dtype=torch.float32)
        else:
            kept_centers = detected_centers[kept_ids - 1]
        instance_maps.append(instance_map)
        center_points.append(kept_centers)

    return {
        "instance_map": torch.stack(instance_maps, dim=0),
        "center_points": center_points,
        "foreground_probs": foreground_probs,
        "center_probs": center_probs,
    }
