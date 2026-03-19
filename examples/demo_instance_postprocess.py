from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_conv_attention import InstancePostprocessConfig, decode_instances


def main() -> None:
    height = width = 32
    foreground_logits = torch.full((1, 1, height, width), -8.0)
    center_logits = torch.full((1, 1, height, width), -8.0)
    offsets = torch.zeros((1, 2, height, width))

    boxes = [
        (slice(4, 14), slice(5, 15), (8.0, 9.0)),
        (slice(18, 28), slice(18, 28), (22.0, 22.0)),
    ]
    for ys, xs, center in boxes:
        foreground_logits[:, :, ys, xs] = 8.0
        center_logits[:, :, int(center[0]), int(center[1])] = 8.0
        yy, xx = torch.meshgrid(
            torch.arange(ys.start, ys.stop, dtype=torch.float32),
            torch.arange(xs.start, xs.stop, dtype=torch.float32),
            indexing="ij",
        )
        offsets[0, 0, ys, xs] = center[0] - yy
        offsets[0, 1, ys, xs] = center[1] - xx

    decoded = decode_instances(
        {
            "foreground_logits": foreground_logits,
            "center_logits": center_logits,
            "offsets": offsets,
        },
        InstancePostprocessConfig(
            foreground_threshold=0.5,
            center_threshold=0.5,
            max_assignment_distance=3.0,
            min_instance_area=4,
        ),
    )
    print(decoded["instance_map"][0])
    print("centers:", decoded["center_points"][0].tolist())


if __name__ == "__main__":
    main()
