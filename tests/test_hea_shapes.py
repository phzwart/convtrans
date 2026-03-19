from __future__ import annotations

import pytest
import torch

from local_conv_attention import HEAFusionBlock2d, HierarchicalElevatorAttention2d


def test_hierarchical_elevator_attention_output_shape() -> None:
    module = HierarchicalElevatorAttention2d(
        query_dim=16,
        memory_dims=[16, 32],
        scale_factors=[2, 4],
        num_heads=4,
        head_dim=8,
        window_sizes=[3, 5],
        dilations=[1, 1],
        implementation="optimized",
        fusion_mode="per_scale",
    )
    query = torch.randn(2, 16, 32, 32)
    memory_1 = torch.randn(2, 16, 16, 16)
    memory_2 = torch.randn(2, 32, 8, 8)
    output = module(query, [memory_1, memory_2])
    assert output.shape == (2, 32, 32, 32)


@pytest.mark.parametrize("fusion", ["gated_residual", "additive", "concat_proj"])
def test_hea_fusion_block_preserves_query_shape(fusion: str) -> None:
    block = HEAFusionBlock2d(
        query_dim=24,
        memory_dims=[24, 48],
        scale_factors=[2, 4],
        num_heads=3,
        head_dim=8,
        window_sizes=[3, 3],
        dilations=[1, 1],
        implementation="optimized",
        fusion_mode="per_scale",
        residual_fusion=fusion,
    )
    query = torch.randn(1, 24, 32, 32)
    memories = [torch.randn(1, 24, 16, 16), torch.randn(1, 48, 8, 8)]
    output = block(query, memories)
    assert output.shape == query.shape
