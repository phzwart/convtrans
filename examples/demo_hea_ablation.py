from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_conv_attention import HEAExplainer, HEAUNet, HEAUNetModelConfig


def main() -> None:
    torch.manual_seed(2)
    config = HEAUNetModelConfig(
        in_channels=1,
        num_classes=1,
        base_channels=8,
        channel_multipliers=[1, 2, 4, 8],
        encoder_depths=[1, 1, 1, 1],
        decoder_depths=[1, 1, 1],
    )
    model = HEAUNet(config).eval()
    x = torch.randn(1, 1, 64, 64)

    explainer = HEAExplainer(model)
    explanation = explainer.explain_pixel(
        x,
        target_xy=(20, 20),
        mode="gated_magnitude",
    )
    scale_heatmap = explanation["per_scale_heatmaps"][1]
    top_coord = torch.nonzero(scale_heatmap == scale_heatmap.max(), as_tuple=False)[0]

    result = explainer.ablate_region(
        x,
        memory_scale=1,
        coarse_coord=(int(top_coord[0]), int(top_coord[1])),
        target_xy=(20, 20),
    )
    print(
        "Ablation delta:",
        float(result["delta"]),
        "baseline:",
        float(result["baseline_score"]),
        "ablated:",
        float(result["ablated_score"]),
    )


if __name__ == "__main__":
    main()
