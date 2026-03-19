from __future__ import annotations

import torch

from local_conv_attention.config import DenseLeJEPAObjectiveConfig
from local_conv_attention.views import DenseAlignedViewGenerator
from local_conv_attention import DenseLeJEPAModel, HEAUNetModelConfig


def test_aligned_same_geometry_preserves_spatial_layout() -> None:
    cfg = DenseLeJEPAObjectiveConfig(num_views=3)
    cfg.views.corruption.intensity_jitter = False
    cfg.views.corruption.blur = False
    cfg.views.corruption.gaussian_noise = False
    cfg.views.corruption.random_block_mask = False
    generator = DenseAlignedViewGenerator(cfg)

    x = torch.zeros(1, 1, 16, 16)
    x[:, :, 5, 7] = 1.0
    views, valid_mask = generator(x)

    assert valid_mask is None
    for view_index in range(views.size(1)):
        nonzero = torch.nonzero(views[0, view_index, 0] == 1.0, as_tuple=False)
        assert nonzero.shape[0] == 1
        assert tuple(nonzero[0].tolist()) == (5, 7)


def test_shared_crop_returns_valid_mask_and_downsampled_mask() -> None:
    config = HEAUNetModelConfig(
        name="hea_dense_lejepa",
        in_channels=1,
        base_channels=4,
        channel_multipliers=[1, 2, 4, 8],
        encoder_depths=[1, 1, 1, 1],
        decoder_depths=[1, 1, 1],
    )
    config.lejepa.views.mode = "aligned_shared_crop"
    config.lejepa.views.shared_crop_ratio = 0.5
    config.lejepa.invariance.loss_on_valid_only = True
    config.lejepa.num_views = 2
    config.lejepa.sigreg.num_slices = 8
    config.lejepa.sigreg.num_knots = 5
    model = DenseLeJEPAModel(config)

    x = torch.randn(2, 1, 32, 32)
    out = model(x)

    assert out["valid_mask"] is not None
    assert out["valid_mask"].shape[-2:] == out["latents"].shape[-2:]


def test_small_multi_block_masking_masks_multiple_local_regions() -> None:
    cfg = DenseLeJEPAObjectiveConfig(num_views=2)
    cfg.views.corruption.intensity_jitter = False
    cfg.views.corruption.blur = False
    cfg.views.corruption.gaussian_noise = False
    cfg.views.corruption.random_block_mask = True
    cfg.views.corruption.block_mask_ratio = 0.125
    cfg.views.corruption.block_mask_num_blocks = 4
    generator = DenseAlignedViewGenerator(cfg)

    x = torch.ones(1, 1, 32, 32)
    views, _ = generator(x)
    masked = views[0, 0, 0]

    zero_count = int((masked == 0).sum().item())
    assert zero_count > 0
    assert zero_count < x.numel()
