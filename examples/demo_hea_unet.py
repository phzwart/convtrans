from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_conv_attention import build_model_from_yaml


def main() -> None:
    model = build_model_from_yaml(Path("configs") / "hea_unet_small.yaml")
    x = torch.randn(1, model.config.in_channels, 128, 128)
    y = model(x)
    print("input shape:", tuple(x.shape))
    print("output shape:", tuple(y.shape))
    print("enabled decoder HEA stages:", model.config.hea.enabled_decoder_stages)


if __name__ == "__main__":
    main()
