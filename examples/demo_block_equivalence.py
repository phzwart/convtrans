from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_conv_attention import LocalTransformerBlock2d, ReferenceLocalTransformerBlock2d


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 32, 6, 6)

    optimized = LocalTransformerBlock2d(dim=32, num_heads=4, window_size=3, implementation="optimized")
    shift_reference = LocalTransformerBlock2d(dim=32, num_heads=4, window_size=3, implementation="shift")
    flattened_reference = ReferenceLocalTransformerBlock2d(dim=32, num_heads=4, window_size=3)
    shift_reference.load_state_dict(optimized.state_dict())
    flattened_reference.load_state_dict(optimized.state_dict())

    y_optimized = optimized(x)
    y_shift = shift_reference(x)
    y_flattened = flattened_reference(x)

    print("block output shape:", tuple(y_optimized.shape))
    print("max |optimized - shift|:", (y_optimized - y_shift).abs().max().item())
    print("max |optimized - flattened|:", (y_optimized - y_flattened).abs().max().item())


if __name__ == "__main__":
    main()
