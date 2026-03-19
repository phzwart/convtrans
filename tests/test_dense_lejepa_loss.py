from __future__ import annotations

import torch

from local_conv_attention.dense_lejepa import DenseLeJEPAModel
from local_conv_attention.losses_ssl import dense_invariance_loss
from local_conv_attention.config import HEAUNetModelConfig


def test_dense_invariance_loss_is_near_zero_for_identical_views() -> None:
    latents = torch.randn(2, 1, 6, 4, 4).repeat(1, 3, 1, 1, 1)
    loss = dense_invariance_loss(latents)
    assert loss.item() < 1e-8


def test_dense_invariance_loss_increases_for_different_views() -> None:
    base = torch.randn(2, 1, 6, 4, 4)
    identical = base.repeat(1, 3, 1, 1, 1)
    different = torch.cat([base, base + 0.5, base - 0.5], dim=1)
    assert dense_invariance_loss(different) > dense_invariance_loss(identical)


def test_dense_lejepa_accepts_precomputed_multi_view_batch() -> None:
    config = HEAUNetModelConfig(
        name="hea_dense_lejepa",
        in_channels=1,
        base_channels=4,
        channel_multipliers=[1, 2, 4, 8],
        encoder_depths=[1, 1, 1, 1],
        decoder_depths=[1, 1, 1],
    )
    config.lejepa.num_views = 2
    config.latent.latent_dim = 8
    config.lejepa.sigreg.num_slices = 8
    config.lejepa.sigreg.num_knots = 5
    model = DenseLeJEPAModel(config)

    x = torch.randn(1, 2, 1, 32, 32)
    out = model(x)

    assert out["latents"].shape == (1, 2, 8, 32, 32)
