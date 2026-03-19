from __future__ import annotations

import torch

from local_conv_attention import BottomUpInstanceLoss, BottomUpInstanceLossConfig, build_instance_targets


def test_instance_loss_runs_and_is_finite() -> None:
    labels = torch.zeros(2, 16, 16, dtype=torch.long)
    labels[0, 3:8, 3:8] = 1
    labels[1, 5:12, 4:10] = 1
    targets = build_instance_targets(labels, {})

    predictions = {
        "foreground_logits": torch.randn(2, 1, 16, 16, requires_grad=True),
        "center_logits": torch.randn(2, 1, 16, 16, requires_grad=True),
        "offsets": torch.randn(2, 2, 16, 16, requires_grad=True),
    }
    criterion = BottomUpInstanceLoss(BottomUpInstanceLossConfig())
    loss_dict = criterion(predictions, targets)

    assert torch.isfinite(loss_dict["loss"])
    loss_dict["loss"].backward()
    assert predictions["foreground_logits"].grad is not None


def test_offset_loss_ignores_background_pixels() -> None:
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    labels[0, 2:6, 2:6] = 1
    targets = build_instance_targets(labels, {})
    criterion = BottomUpInstanceLoss(BottomUpInstanceLossConfig())

    pred_a = {
        "foreground_logits": torch.zeros(1, 1, 8, 8),
        "center_logits": torch.zeros(1, 1, 8, 8),
        "offsets": torch.zeros(1, 2, 8, 8),
    }
    pred_b = {
        "foreground_logits": torch.zeros(1, 1, 8, 8),
        "center_logits": torch.zeros(1, 1, 8, 8),
        "offsets": torch.zeros(1, 2, 8, 8),
    }
    pred_b["offsets"][:, :, 0, 0] = 99.0

    loss_a = criterion(pred_a, targets)["offset_loss"]
    loss_b = criterion(pred_b, targets)["offset_loss"]
    torch.testing.assert_close(loss_a, loss_b)
