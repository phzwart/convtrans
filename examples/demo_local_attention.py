from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_conv_attention import (
    FlattenedMaskedLocalAttention2d,
    LocalAttention2d,
    ReferenceLocalAttention2d,
    ShiftLocalAttention2d,
)


def main() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 16, 5, 5)
    k = torch.randn(1, 16, 5, 5)
    v = torch.randn(1, 16, 5, 5)

    optimized = LocalAttention2d(num_heads=4, window_size=3)
    shift_ref = ShiftLocalAttention2d(num_heads=4, window_size=3)
    unfold_ref = ReferenceLocalAttention2d(num_heads=4, window_size=3)
    flat_ref = FlattenedMaskedLocalAttention2d(num_heads=4, window_size=3)

    out_optimized = optimized(q, k, v)
    out_shift = shift_ref(q, k, v)
    out_unfold = unfold_ref(q, k, v)
    out_flat = flat_ref(q, k, v)

    print("optimized output shape:", tuple(out_optimized.shape))
    print("max |optimized - shift|:", (out_optimized - out_shift).abs().max().item())
    print("max |optimized - unfold|:", (out_optimized - out_unfold).abs().max().item())
    print("max |optimized - flattened|:", (out_optimized - out_flat).abs().max().item())


if __name__ == "__main__":
    main()
