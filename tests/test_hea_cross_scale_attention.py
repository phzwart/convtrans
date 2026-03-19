from __future__ import annotations

import torch

from local_conv_attention import HierarchicalElevatorAttention2d


def test_window_one_cross_scale_attention_retrieves_coarse_anchor_values() -> None:
    module = HierarchicalElevatorAttention2d(
        query_dim=4,
        memory_dims=[4],
        scale_factors=[2],
        num_heads=1,
        head_dim=4,
        window_sizes=[1],
        dilations=[1],
        implementation="optimized",
        fusion_mode="per_scale",
        qkv_bias=False,
        joint_scale_projection=False,
    )
    with torch.no_grad():
        eye = torch.eye(4).view(4, 4, 1, 1)
        module.query_proj.weight.copy_(eye)
        module.key_projs[0].weight.copy_(eye)
        module.value_projs[0].weight.copy_(eye)

    query = torch.randn(1, 4, 4, 4)
    memory = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[10.0, 20.0], [30.0, 40.0]],
                [[100.0, 200.0], [300.0, 400.0]],
                [[1000.0, 2000.0], [3000.0, 4000.0]],
            ]
        ]
    )
    output = module(query, [memory])
    expected = memory.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    torch.testing.assert_close(output, expected)
