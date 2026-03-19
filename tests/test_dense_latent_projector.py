from __future__ import annotations

import torch

from local_conv_attention import DenseLatentProjector2d


def test_dense_latent_projector_shape() -> None:
    projector = DenseLatentProjector2d(16, 12, depth=2)
    x = torch.randn(2, 16, 8, 8)
    y = projector(x)
    assert y.shape == (2, 12, 8, 8)


def test_dense_latent_projector_can_normalize() -> None:
    projector = DenseLatentProjector2d(8, 4, depth=1, normalize_latents=True)
    x = torch.randn(1, 8, 4, 4)
    y = projector(x)
    norms = torch.linalg.vector_norm(y, dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-4)
