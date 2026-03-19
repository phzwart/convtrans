from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_conv_attention import HEAExplainer, HEAUNet, HEAUNetModelConfig
from local_conv_attention.visualization import visualize_signed_explanation


def main() -> None:
    torch.manual_seed(1)
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
    result = explainer.explain_patch(
        x,
        center_xy=(30, 30),
        patch_radius=2,
        mode="signed_logit",
        target_channel=0,
    )
    fig = visualize_signed_explanation(
        x[0],
        positive_heatmap=result["positive_heatmap"],
        negative_heatmap=result["negative_heatmap"],
        target_xy=result["target_xy"],
    )
    fig.savefig("examples/demo_hea_explain_patch.png", dpi=150)
    plt.close(fig)
    print("Saved examples/demo_hea_explain_patch.png")


if __name__ == "__main__":
    main()
