from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_conv_attention import build_model_from_yaml


def main() -> None:
    torch.manual_seed(0)
    model = build_model_from_yaml(PROJECT_ROOT / "configs" / "hea_dense_lejepa_default.yaml")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(2, 1, 64, 64)

    model.train()
    out = model(x)
    out["loss"].backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print("train_step_loss:", float(out["loss"].detach()))


if __name__ == "__main__":
    main()
