from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_conv_attention import build_model_from_yaml


def main() -> None:
    configs = [
        Path("configs") / "basic_unet_small.yaml",
        Path("configs") / "swin_unet_small.yaml",
    ]
    x = torch.randn(1, 3, 128, 128)
    for config_path in configs:
        model = build_model_from_yaml(config_path)
        y = model(x)
        print(config_path.name, tuple(y.shape))


if __name__ == "__main__":
    main()
