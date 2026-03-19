from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_conv_attention import HEAFusionBlock2d


def main() -> None:
    torch.manual_seed(0)
    block = HEAFusionBlock2d(
        query_dim=32,
        memory_dims=[32, 64],
        scale_factors=[2, 4],
        num_heads=4,
        head_dim=8,
        window_sizes=[3, 5],
        dilations=[1, 2],
        implementation="optimized",
        fusion_mode="per_scale",
    )
    query = torch.randn(1, 32, 64, 64)
    memories = [torch.randn(1, 32, 32, 32), torch.randn(1, 64, 16, 16)]
    output = block(query, memories)
    print("query shape:", tuple(query.shape))
    print("memory shapes:", [tuple(memory.shape) for memory in memories])
    print("output shape:", tuple(output.shape))


if __name__ == "__main__":
    main()
