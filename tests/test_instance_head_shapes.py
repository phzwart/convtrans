from __future__ import annotations

import torch

from local_conv_attention import BottomUpInstanceHead2d, InstanceHeadConfig


def test_instance_head_shapes() -> None:
    head = BottomUpInstanceHead2d(
        in_channels=16,
        config=InstanceHeadConfig(shared_dim=24, shared_depth=1, branch_depth=1),
        num_classes=3,
    )
    x = torch.randn(2, 16, 32, 32)
    outputs = head(x)
    assert outputs["foreground_logits"].shape == (2, 1, 32, 32)
    assert outputs["center_logits"].shape == (2, 1, 32, 32)
    assert outputs["offsets"].shape == (2, 2, 32, 32)


def test_instance_head_optional_semantic_branch() -> None:
    head = BottomUpInstanceHead2d(
        in_channels=8,
        config=InstanceHeadConfig(semantic_branch=True, shared_dim=8),
        num_classes=4,
    )
    outputs = head(torch.randn(1, 8, 16, 16))
    assert outputs["semantic_logits"].shape == (1, 4, 16, 16)
