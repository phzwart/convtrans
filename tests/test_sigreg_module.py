from __future__ import annotations

import torch

from local_conv_attention import SIGRegLoss


def test_sigreg_loss_is_finite_for_random_embeddings() -> None:
    torch.manual_seed(0)
    loss_fn = SIGRegLoss(num_slices=32, num_knots=9, t_max=2.5)
    embeddings = torch.randn(64, 16)
    loss = loss_fn(embeddings)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_sigreg_loss_supports_gradients() -> None:
    torch.manual_seed(1)
    loss_fn = SIGRegLoss(num_slices=16, num_knots=7, t_max=3.0)
    embeddings = torch.randn(32, 8, requires_grad=True)
    loss = loss_fn(embeddings)
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
