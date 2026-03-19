from __future__ import annotations

import torch

from local_conv_attention import DenseLeJEPAModel, HEAUNetModelConfig


def _make_config() -> HEAUNetModelConfig:
    config = HEAUNetModelConfig(
        name="hea_dense_lejepa",
        in_channels=1,
        base_channels=4,
        channel_multipliers=[1, 2, 4, 8],
        encoder_depths=[1, 1, 1, 1],
        decoder_depths=[1, 1, 1],
    )
    config.attention.heads = 2
    config.attention.head_dim = 4
    config.latent.latent_dim = 10
    config.lejepa.num_views = 3
    config.lejepa.sigreg.num_slices = 16
    config.lejepa.sigreg.num_knots = 7
    config.validate()
    return config


def test_dense_lejepa_forward_runs_end_to_end() -> None:
    torch.manual_seed(2)
    model = DenseLeJEPAModel(_make_config())
    x = torch.randn(2, 1, 32, 32)

    out = model(x)

    assert out["latents"].shape == (2, 3, 10, 32, 32)
    assert torch.isfinite(out["inv_loss"])
    assert torch.isfinite(out["sigreg_loss"])
    assert torch.isfinite(out["loss"])


def test_dense_lejepa_backward_flows_through_backbone_and_projector() -> None:
    torch.manual_seed(3)
    model = DenseLeJEPAModel(_make_config())
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    out["loss"].backward()

    backbone_grad = model.backbone.stem.proj.conv.weight.grad
    projector_grad = model.projector.proj[0].weight.grad
    assert backbone_grad is not None
    assert projector_grad is not None


def test_dense_lejepa_multi_source_trains_all_hooks() -> None:
    cfg = _make_config()
    cfg.latent.sources = ["encoder_0", "top"]
    cfg.validate()
    model = DenseLeJEPAModel(cfg)
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    assert set(out["latents_by_source"].keys()) == {"encoder_0", "top"}
    assert out["latents"].shape == out["latents_by_source"]["encoder_0"].shape
    assert out["latents_by_source"]["top"].shape[-2:] == (32, 32)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.backbone.stem.proj.conv.weight.grad is not None


def test_dense_lejepa_sequential_view_forward_runs() -> None:
    """Sequential forward saves activation memory; grads should still flow."""
    torch.manual_seed(4)
    cfg = _make_config()
    cfg.lejepa.sequential_view_forward = True
    model = DenseLeJEPAModel(cfg)
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    assert out["latents"].shape == (2, 3, 10, 32, 32)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.backbone.stem.proj.conv.weight.grad is not None
