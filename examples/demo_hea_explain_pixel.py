from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_conv_attention import HEAExplainer, HEAUNet, HEAUNetModelConfig


def main() -> None:
    torch.manual_seed(0)
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
    result = explainer.explain_pixel(
        x,
        target_xy=(24, 24),
        mode="gated_magnitude",
    )
    fig = explainer.visualize_explanation(x[0], result)
    fig.savefig("examples/demo_hea_explain_pixel.png", dpi=150)
    plt.close(fig)
    print("Saved examples/demo_hea_explain_pixel.png")


if __name__ == "__main__":
    main()
