from __future__ import annotations

from pathlib import Path

from local_conv_attention import DenseLeJEPAModel, build_model_from_yaml, load_experiment_config


def test_dense_lejepa_yaml_config_loads() -> None:
    path = Path("configs/hea_dense_lejepa_default.yaml")
    config = load_experiment_config(path)
    assert config.model.name == "hea_dense_lejepa"
    assert config.model.latent.latent_dim == 64


def test_dense_lejepa_factory_builds_model() -> None:
    model = build_model_from_yaml("configs/hea_dense_lejepa_default.yaml")
    assert isinstance(model, DenseLeJEPAModel)
