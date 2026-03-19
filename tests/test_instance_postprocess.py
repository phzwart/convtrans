from __future__ import annotations

import torch

from local_conv_attention import InstancePostprocessConfig, decode_instances


def _synthetic_predictions() -> dict[str, torch.Tensor]:
    height = width = 24
    foreground_logits = torch.full((1, 1, height, width), -8.0)
    center_logits = torch.full((1, 1, height, width), -8.0)
    offsets = torch.zeros((1, 2, height, width))

    boxes = [
        (slice(3, 9), slice(4, 10), (5.5, 6.5)),
        (slice(14, 20), slice(14, 20), (16.5, 16.5)),
    ]
    for ys, xs, center in boxes:
        foreground_logits[:, :, ys, xs] = 8.0
        center_logits[:, :, round(center[0]), round(center[1])] = 8.0
        yy, xx = torch.meshgrid(
            torch.arange(ys.start, ys.stop, dtype=torch.float32),
            torch.arange(xs.start, xs.stop, dtype=torch.float32),
            indexing="ij",
        )
        offsets[0, 0, ys, xs] = center[0] - yy
        offsets[0, 1, ys, xs] = center[1] - xx

    return {
        "foreground_logits": foreground_logits,
        "center_logits": center_logits,
        "offsets": offsets,
    }


def test_decode_instances_groups_pixels_around_detected_centers() -> None:
    decoded = decode_instances(
        _synthetic_predictions(),
        InstancePostprocessConfig(
            foreground_threshold=0.5,
            center_threshold=0.5,
            max_assignment_distance=2.0,
            min_instance_area=4,
        ),
    )
    instance_map = decoded["instance_map"][0]
    instance_ids = torch.unique(instance_map)
    assert torch.equal(instance_ids, torch.tensor([0, 1, 2]))
    assert int((instance_map == 1).sum().item()) > 0
    assert int((instance_map == 2).sum().item()) > 0


def test_decode_instances_handles_empty_foreground() -> None:
    decoded = decode_instances(
        {
            "foreground_logits": torch.full((1, 1, 8, 8), -8.0),
            "center_logits": torch.full((1, 1, 8, 8), -8.0),
            "offsets": torch.zeros((1, 2, 8, 8)),
        },
        InstancePostprocessConfig(),
    )
    assert decoded["instance_map"].sum().item() == 0


def test_decode_instances_handles_missing_centers() -> None:
    predictions = _synthetic_predictions()
    predictions["center_logits"].fill_(-8.0)
    decoded = decode_instances(predictions, InstancePostprocessConfig(center_threshold=0.8))
    assert decoded["instance_map"].sum().item() == 0
