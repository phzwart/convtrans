from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_conv_attention import build_model_from_yaml


def main() -> None:
    model = build_model_from_yaml(PROJECT_ROOT / "configs" / "hea_dense_lejepa_default.yaml")
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    print("latents:", tuple(out["latents"].shape))
    print("inv_loss:", float(out["inv_loss"].detach()))
    print("sigreg_loss:", float(out["sigreg_loss"].detach()))
    print("loss:", float(out["loss"].detach()))


if __name__ == "__main__":
    main()
