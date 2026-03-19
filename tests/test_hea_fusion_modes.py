from __future__ import annotations

import torch

from local_conv_attention import HierarchicalElevatorAttention2d
from local_conv_attention.utils import attention_tolerances


def test_per_scale_and_joint_softmax_match_for_single_memory_scale() -> None:
    kwargs = dict(
        query_dim=8,
        memory_dims=[8],
        scale_factors=[2],
        num_heads=2,
        head_dim=4,
        window_sizes=[3],
        dilations=[1],
        implementation="optimized",
    )
    per_scale = HierarchicalElevatorAttention2d(**kwargs, fusion_mode="per_scale")
    joint = HierarchicalElevatorAttention2d(**kwargs, fusion_mode="joint_softmax")
    joint.load_state_dict(per_scale.state_dict())

    query = torch.randn(2, 8, 16, 16)
    memory = torch.randn(2, 8, 8, 8)
    out_a = per_scale(query, [memory])
    out_b = joint(query, [memory])

    atol, rtol = attention_tolerances(torch.float32)
    torch.testing.assert_close(out_a, out_b, atol=atol, rtol=rtol)
