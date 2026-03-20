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


def test_dense_lejepa_forward_with_backbone_gradient_checkpointing() -> None:
    cfg = _make_config()
    cfg.backbone_gradient_checkpointing = True
    cfg.validate()
    model = DenseLeJEPAModel(cfg)
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.backbone.stem.proj.conv.weight.grad is not None
    assert model.projector.proj[0].weight.grad is not None


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


def test_dense_lejepa_rotate_step_mode_cycles_hooks() -> None:
    cfg = _make_config()
    cfg.latent.sources = ["encoder_0", "top"]
    cfg.latent.step_mode = "rotate"
    cfg.validate()
    model = DenseLeJEPAModel(cfg)
    x = torch.randn(2, 1, 32, 32)
    o0 = model(x, rotate_latent_index=0)
    o1 = model(x, rotate_latent_index=1)
    assert o0["active_latent_source"] == "encoder_0"
    assert o1["active_latent_source"] == "top"
    assert set(o0["latents_by_source"].keys()) == {"encoder_0"}
    assert set(o1["latents_by_source"].keys()) == {"top"}
    (o0["loss"] + o1["loss"]).backward()


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


def test_dense_lejepa_decoder_hooks_with_raw_skips_and_3x3_projector() -> None:
    """Decoder-only hooks + concat skips + 3x3 projector — the production config."""
    from local_conv_attention import default_decoder_latent_hooks

    cfg = HEAUNetModelConfig(
        name="hea_dense_lejepa",
        in_channels=1,
        base_channels=4,
        channel_multipliers=[1, 2, 4],
        encoder_depths=[1, 1, 1],
        decoder_depths=[2, 2],
        swin_stage_heads=[2, 4, 8],
        use_raw_skips=True,
    )
    cfg.attention.heads = 2
    cfg.attention.head_dim = 4
    cfg.hea.enabled_decoder_stages = [0, 1]
    cfg.semantic_memory.enabled_scales = [1, 2]
    cfg.semantic_memory.block_depths = [1, 1]
    cfg.semantic_memory.window_sizes = [3, 3]
    cfg.semantic_memory.dilations = [1, 1]
    cfg.hea.per_scale_window_sizes = [3, 3]
    cfg.hea.per_scale_dilations = [1, 1]
    cfg.latent.sources = default_decoder_latent_hooks(3)
    cfg.latent.latent_dim = 8
    cfg.latent.projector_depth = 2
    cfg.latent.projector_kernel_size = 3
    cfg.lejepa.num_views = 2
    cfg.lejepa.sigreg.num_slices = 8
    cfg.lejepa.sigreg.num_knots = 5
    cfg.validate()

    torch.manual_seed(42)
    model = DenseLeJEPAModel(cfg)
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    expected_hooks = {"bottleneck", "decoder_0", "decoder_1", "top"}
    assert set(out["latents_by_source"].keys()) == expected_hooks
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.backbone.stem.proj.conv.weight.grad is not None
