from __future__ import annotations

import torch

from local_conv_attention import (
    InstanceTargetConfig,
    build_center_heatmap_target,
    build_foreground_target,
    build_instance_targets,
    build_offset_target,
)


def _toy_instance_map() -> torch.Tensor:
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    labels[0, 1:3, 1:3] = 1
    labels[0, 5:7, 5:7] = 2
    return labels


def test_instance_targets_shapes_and_masks() -> None:
    labels = _toy_instance_map()
    targets = build_instance_targets(labels, InstanceTargetConfig())
    assert targets["foreground_target"].shape == (1, 1, 8, 8)
    assert targets["center_target"].shape == (1, 1, 8, 8)
    assert targets["offset_target"].shape == (1, 2, 8, 8)
    assert targets["offset_weight"].shape == (1, 1, 8, 8)
    assert torch.equal(targets["foreground_target"][0, 0], (labels[0] > 0).float())


def test_center_heatmap_places_peaks_near_instance_centers() -> None:
    labels = _toy_instance_map()
    heatmap = build_center_heatmap_target(labels, InstanceTargetConfig().center)
    assert heatmap[0, 0, 1, 1] > 0.8
    assert heatmap[0, 0, 5, 5] > 0.8


def test_offset_targets_point_toward_instance_center() -> None:
    labels = _toy_instance_map()
    offsets, weights = build_offset_target(labels, normalize=False)
    assert torch.all(weights[labels.unsqueeze(1) > 0] == 1)
    torch.testing.assert_close(offsets[0, :, 1, 1], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(offsets[0, :, 2, 2], torch.tensor([-0.5, -0.5]))


def test_foreground_target_builder() -> None:
    labels = _toy_instance_map()
    foreground = build_foreground_target(labels)
    assert foreground.sum().item() == 8
