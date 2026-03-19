from __future__ import annotations

from pathlib import Path

from local_conv_attention import build_model, build_model_from_yaml, load_experiment_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_build_model_from_yaml_presets() -> None:
    for name in [
        "hea_unet_default.yaml",
        "hea_unet_small.yaml",
        "hea_unet_progressive.yaml",
        "hea_unet_ablation_raw_skips.yaml",
    ]:
        model = build_model_from_yaml(CONFIG_DIR / name)
        assert model.__class__.__name__ == "HEAUNet"


def test_build_model_from_loaded_config() -> None:
    config = load_experiment_config(CONFIG_DIR / "hea_unet_small.yaml")
    model = build_model(config)
    assert model.config.name == "hea_unet"
