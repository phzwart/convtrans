"""Dataclass-based configuration for segmentation and instance models."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from types import UnionType
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml


T = TypeVar("T")

ModelName = Literal[
    "hea_unet",
    "basic_unet",
    "swin_unet",
    "hea_unet_instance",
    "basic_unet_instance",
    "hea_dense_lejepa",
]


@dataclass
class HEAAttentionConfig:
    operator_backend: Literal["optimized", "shift"] = "optimized"
    fusion_mode: Literal["per_scale", "joint_softmax"] = "per_scale"
    elevator_mode: Literal["direct", "progressive"] = "direct"
    heads: int = 4
    head_dim: int = 16
    out_bias: bool = True
    qkv_bias: bool = True


@dataclass
class SemanticMemoryConfig:
    enabled_scales: list[int] = field(default_factory=lambda: [1, 2, 3])
    block_depths: list[int] = field(default_factory=lambda: [1, 1, 2])
    window_sizes: list[int] = field(default_factory=lambda: [3, 3, 5])
    dilations: list[int] = field(default_factory=lambda: [1, 2, 2])
    use_local_transformer_block: bool = True

    def validate(self) -> None:
        expected = len(self.enabled_scales)
        for name, values in {
            "block_depths": self.block_depths,
            "window_sizes": self.window_sizes,
            "dilations": self.dilations,
        }.items():
            if len(values) != expected:
                raise ValueError(f"{name} must have length {expected}, got {len(values)}.")


@dataclass
class HEAFusionConfig:
    enabled_decoder_stages: list[int] = field(default_factory=lambda: [0])
    query_source: Literal["decoder"] = "decoder"
    key_value_source: Literal["semantic_memory"] = "semantic_memory"
    per_scale_window_sizes: list[int] = field(default_factory=lambda: [3, 3, 5])
    per_scale_dilations: list[int] = field(default_factory=lambda: [1, 1, 2])
    fusion: Literal["gated_residual", "additive", "concat_proj"] = "gated_residual"
    project_context: bool = True
    joint_scale_projection: bool = True

    def validate(self, num_memory_scales: int) -> None:
        for name, values in {
            "per_scale_window_sizes": self.per_scale_window_sizes,
            "per_scale_dilations": self.per_scale_dilations,
        }.items():
            if len(values) != num_memory_scales:
                raise ValueError(
                    f"{name} must have length {num_memory_scales}, got {len(values)}."
                )


@dataclass
class TrunkConfig:
    type: Literal["hea_unet", "basic_unet", "hea_backbone"] | None = None
    hea_enabled: bool | None = None
    hea_enabled_decoder_stages: list[int] | None = None
    elevator_mode: Literal["direct", "progressive"] | None = None
    fusion_mode: Literal["per_scale", "joint_softmax"] | None = None
    operator_backend: Literal["optimized", "shift"] | None = None


def _validate_latent_hook_name(name: str) -> None:
    if name in ("top", "bottleneck"):
        return
    if name.startswith("encoder_"):
        int(name.split("_", maxsplit=1)[1])
        return
    if name.startswith("decoder_"):
        int(name.split("_", maxsplit=1)[1])
        return
    raise ValueError(
        f"Unsupported latent hook {name!r}; use top, bottleneck, encoder_<k>, or decoder_<k>."
    )


def default_all_latent_hooks(num_scales: int) -> list[str]:
    """All standard hooks for a pyramid with ``num_scales`` encoder stages.

    Order: all ``encoder_*``, ``bottleneck``, all ``decoder_*``, ``top``.
    """
    if num_scales < 1:
        raise ValueError("num_scales must be positive.")
    hooks: list[str] = [f"encoder_{k}" for k in range(num_scales)]
    hooks.append("bottleneck")
    hooks.extend(f"decoder_{k}" for k in range(max(0, num_scales - 1)))
    hooks.append("top")
    return hooks


def default_decoder_latent_hooks(num_scales: int) -> list[str]:
    """Decoder-side hooks only: ``bottleneck``, all ``decoder_*``, ``top``.

    These hooks carry cross-scale context from semantic memories and HEA fusion
    (unlike encoder hooks which are pure conv features).
    """
    if num_scales < 1:
        raise ValueError("num_scales must be positive.")
    hooks: list[str] = ["bottleneck"]
    hooks.extend(f"decoder_{k}" for k in range(max(0, num_scales - 1)))
    hooks.append("top")
    return hooks


def _validate_latent_hooks_for_pyramid(hooks: list[str], num_scales: int) -> None:
    """``num_scales`` = len(encoder stages); decoder stages are ``0 .. num_scales-2``."""
    if num_scales < 1:
        raise ValueError("num_scales must be positive.")
    max_encoder = num_scales - 1
    max_decoder = num_scales - 2
    for name in hooks:
        _validate_latent_hook_name(name)
        if name.startswith("encoder_"):
            k = int(name.split("_", maxsplit=1)[1])
            if k < 0 or k > max_encoder:
                raise ValueError(
                    f"latent hook {name!r} invalid for {num_scales} encoder stages (valid 0..{max_encoder})."
                )
        if name.startswith("decoder_"):
            if max_decoder < 0:
                raise ValueError("No decoder stages for this pyramid depth.")
            k = int(name.split("_", maxsplit=1)[1])
            if k < 0 or k > max_decoder:
                raise ValueError(
                    f"latent hook {name!r} invalid for decoder (valid 0..{max_decoder})."
                )


@dataclass
class DenseLatentConfig:
    """Dense LeJEPA hooks: one or more backbone tensors to project and train."""

    source: str = "top"
    sources: list[str] | None = None
    #: ``joint`` — every forward averages loss over all ``sources``. ``rotate`` — each forward
    #: should pass ``rotate_latent_index`` (e.g. ``global_batch_idx % num_hooks``) to train one
    #: hook per batch, cycling through scales (lower memory / staged optimization).
    step_mode: Literal["joint", "rotate"] = "joint"
    latent_dim: int = 128
    projector_depth: int = 1
    #: Spatial kernel size for projector conv layers.  ``1`` = pointwise (default);
    #: ``3`` adds spatial smoothing which helps decoder/top hooks that carry upsample artifacts.
    projector_kernel_size: int = 1
    normalize_latents: bool = False

    def resolved_sources(self) -> list[str]:
        if self.sources is not None and len(self.sources) > 0:
            raw = list(self.sources)
        else:
            raw = [self.source]
        out: list[str] = []
        seen: set[str] = set()
        for name in raw:
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def validate(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent.latent_dim must be positive.")
        if self.projector_depth < 1:
            raise ValueError("latent.projector_depth must be at least 1.")
        if self.projector_kernel_size not in (1, 3, 5, 7):
            raise ValueError("latent.projector_kernel_size must be 1, 3, 5, or 7.")
        if not self.resolved_sources():
            raise ValueError("latent.sources (or latent.source) must list at least one hook.")
        for name in self.resolved_sources():
            _validate_latent_hook_name(name)
        if self.step_mode not in ("joint", "rotate"):
            raise ValueError('latent.step_mode must be "joint" or "rotate".')


@dataclass
class DenseInvarianceConfig:
    loss_type: Literal["mse"] = "mse"
    loss_on_valid_only: bool = False


@dataclass
class DenseSIGRegConfig:
    enabled: bool = True
    num_slices: int = 256
    num_knots: int = 17
    t_max: float = 3.0
    per_view: bool = True

    def validate(self) -> None:
        if self.num_slices <= 0:
            raise ValueError("lejepa.sigreg.num_slices must be positive.")
        if self.num_knots < 2:
            raise ValueError("lejepa.sigreg.num_knots must be at least 2.")
        if self.t_max <= 0:
            raise ValueError("lejepa.sigreg.t_max must be positive.")


@dataclass
class DenseViewCorruptionConfig:
    intensity_jitter: bool = True
    blur: bool = True
    gaussian_noise: bool = True
    random_block_mask: bool = False
    block_mask_ratio: float = 0.0
    block_mask_num_blocks: int = 1

    def validate(self) -> None:
        if not 0.0 <= self.block_mask_ratio <= 1.0:
            raise ValueError("lejepa.views.corruption.block_mask_ratio must be in [0, 1].")
        if self.block_mask_num_blocks <= 0:
            raise ValueError("lejepa.views.corruption.block_mask_num_blocks must be positive.")


@dataclass
class DenseViewConfig:
    mode: Literal["aligned_same_geometry", "aligned_shared_crop"] = "aligned_same_geometry"
    same_geometry: bool = True
    shared_crop_ratio: float = 0.875
    corruption: DenseViewCorruptionConfig = field(default_factory=DenseViewCorruptionConfig)

    def validate(self) -> None:
        if not 0.0 < self.shared_crop_ratio <= 1.0:
            raise ValueError("lejepa.views.shared_crop_ratio must be in (0, 1].")
        self.corruption.validate()


@dataclass
class DenseLeJEPAObjectiveConfig:
    num_views: int = 4
    lambda_sigreg: float = 0.02
    #: Run backbone+projector one view at a time (lower peak VRAM; slower). With
    #: BatchNorm, effective norm batch differs from a fused forward — consider GroupNorm
    #: if you need identical statistics to the fused path.
    sequential_view_forward: bool = False
    invariance: DenseInvarianceConfig = field(default_factory=DenseInvarianceConfig)
    sigreg: DenseSIGRegConfig = field(default_factory=DenseSIGRegConfig)
    views: DenseViewConfig = field(default_factory=DenseViewConfig)

    def validate(self) -> None:
        if self.num_views < 2:
            raise ValueError("lejepa.num_views must be at least 2.")
        if not 0.0 <= self.lambda_sigreg <= 1.0:
            raise ValueError("lejepa.lambda_sigreg must be in [0, 1].")
        self.sigreg.validate()
        self.views.validate()


@dataclass
class InstanceHeadConfig:
    shared_dim: int = 64
    shared_depth: int = 1
    branch_depth: int = 1
    foreground_channels: int = 1
    center_channels: int = 1
    offset_channels: int = 2
    boundary_branch: bool = False
    semantic_branch: bool = False

    def validate(self, num_classes: int) -> None:
        if self.shared_dim <= 0:
            raise ValueError("instance_head.shared_dim must be positive.")
        if self.shared_depth < 0:
            raise ValueError("instance_head.shared_depth must be non-negative.")
        if self.branch_depth < 0:
            raise ValueError("instance_head.branch_depth must be non-negative.")
        if self.foreground_channels != 1:
            raise ValueError("instance_head.foreground_channels must be 1.")
        if self.center_channels != 1:
            raise ValueError("instance_head.center_channels must be 1.")
        if self.offset_channels != 2:
            raise ValueError("instance_head.offset_channels must be 2.")
        if self.semantic_branch and num_classes <= 0:
            raise ValueError("semantic_branch requires model.num_classes > 0.")


@dataclass
class CenterTargetConfig:
    mode: Literal["gaussian"] = "gaussian"
    sigma: float = 3.0
    radius: int = 5

    def validate(self) -> None:
        if self.sigma <= 0:
            raise ValueError("targets.center.sigma must be positive.")
        if self.radius < 0:
            raise ValueError("targets.center.radius must be non-negative.")


@dataclass
class OffsetTargetConfig:
    normalize: bool = False


@dataclass
class InstanceTargetConfig:
    center: CenterTargetConfig = field(default_factory=CenterTargetConfig)
    offsets: OffsetTargetConfig = field(default_factory=OffsetTargetConfig)

    def validate(self) -> None:
        self.center.validate()


@dataclass
class ForegroundLossConfig:
    type: Literal["bce", "dice_bce"] = "bce"
    weight: float = 1.0
    dice_smooth: float = 1.0

    def validate(self) -> None:
        if self.weight < 0:
            raise ValueError("loss.foreground.weight must be non-negative.")
        if self.dice_smooth <= 0:
            raise ValueError("loss.foreground.dice_smooth must be positive.")


@dataclass
class CenterLossConfig:
    type: Literal["mse", "bce"] = "mse"
    weight: float = 1.0

    def validate(self) -> None:
        if self.weight < 0:
            raise ValueError("loss.center.weight must be non-negative.")


@dataclass
class OffsetLossConfig:
    type: Literal["l1", "smooth_l1"] = "l1"
    weight: float = 1.0
    beta: float = 1.0

    def validate(self) -> None:
        if self.weight < 0:
            raise ValueError("loss.offset.weight must be non-negative.")
        if self.beta <= 0:
            raise ValueError("loss.offset.beta must be positive.")


@dataclass
class SemanticLossConfig:
    type: Literal["cross_entropy"] = "cross_entropy"
    weight: float = 1.0
    ignore_index: int = -100

    def validate(self) -> None:
        if self.weight < 0:
            raise ValueError("loss.semantic.weight must be non-negative.")


@dataclass
class BottomUpInstanceLossConfig:
    foreground: ForegroundLossConfig = field(default_factory=ForegroundLossConfig)
    center: CenterLossConfig = field(default_factory=CenterLossConfig)
    offset: OffsetLossConfig = field(default_factory=OffsetLossConfig)
    semantic: SemanticLossConfig = field(default_factory=SemanticLossConfig)

    def validate(self) -> None:
        self.foreground.validate()
        self.center.validate()
        self.offset.validate()
        self.semantic.validate()


@dataclass
class InstancePostprocessConfig:
    method: Literal["center_offsets", "connected_components"] = "center_offsets"
    foreground_threshold: float = 0.5
    center_threshold: float = 0.3
    nms_kernel_size: int = 5
    max_assignment_distance: float = 20.0
    min_instance_area: int = 8
    offsets_normalized: bool = False

    def validate(self) -> None:
        if not 0.0 <= self.foreground_threshold <= 1.0:
            raise ValueError("postprocess.foreground_threshold must be in [0, 1].")
        if not 0.0 <= self.center_threshold <= 1.0:
            raise ValueError("postprocess.center_threshold must be in [0, 1].")
        if self.nms_kernel_size <= 0 or self.nms_kernel_size % 2 == 0:
            raise ValueError("postprocess.nms_kernel_size must be a positive odd integer.")
        if self.max_assignment_distance < 0:
            raise ValueError("postprocess.max_assignment_distance must be non-negative.")
        if self.min_instance_area < 0:
            raise ValueError("postprocess.min_instance_area must be non-negative.")


@dataclass
class HEAUNetModelConfig:
    name: ModelName = "hea_unet"
    in_channels: int = 1
    num_classes: int = 2
    base_channels: int = 32
    channel_multipliers: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    encoder_depths: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    decoder_depths: list[int] = field(default_factory=lambda: [2, 2, 2])
    norm: Literal["batchnorm", "groupnorm", "instancenorm", "none"] = "batchnorm"
    act: Literal["gelu", "relu", "silu"] = "gelu"
    use_raw_skips: bool = False
    patch_size: int = 3
    mlp_ratio: float = 4.0
    #: If True, ``HEABackbone.forward_features`` uses gradient checkpointing on encode + decode
    #: to trade extra compute for lower activation memory (helps dense LeJEPA / large batches).
    #: Prefer ``groupnorm`` over ``batchnorm`` when enabled (BN updates twice per step during ckpt).
    backbone_gradient_checkpointing: bool = False
    swin_window_size: int = 7
    swin_stage_heads: list[int] = field(default_factory=lambda: [2, 4, 8, 8])
    bottleneck_depth: int = 1
    bottleneck_window_size: int = 5
    bottleneck_dilation: int = 1
    trunk: TrunkConfig = field(default_factory=TrunkConfig)
    backbone: TrunkConfig = field(default_factory=TrunkConfig)
    attention: HEAAttentionConfig = field(default_factory=HEAAttentionConfig)
    semantic_memory: SemanticMemoryConfig = field(default_factory=SemanticMemoryConfig)
    hea: HEAFusionConfig = field(default_factory=HEAFusionConfig)
    instance_head: InstanceHeadConfig = field(default_factory=InstanceHeadConfig)
    latent: DenseLatentConfig = field(default_factory=DenseLatentConfig)
    lejepa: DenseLeJEPAObjectiveConfig = field(default_factory=DenseLeJEPAObjectiveConfig)

    def is_instance_model(self) -> bool:
        return self.name in {"hea_unet_instance", "basic_unet_instance"}

    def is_dense_ssl_model(self) -> bool:
        return self.name == "hea_dense_lejepa"

    def trunk_name(self) -> Literal["hea_unet", "basic_unet", "swin_unet"]:
        for trunk_like in (self.backbone, self.trunk):
            if trunk_like.type == "hea_backbone":
                return "hea_unet"
            if trunk_like.type is not None:
                return trunk_like.type
        if self.name == "hea_dense_lejepa":
            return "hea_unet"
        if self.name in {"hea_unet", "hea_unet_instance"}:
            return "hea_unet"
        if self.name in {"basic_unet", "basic_unet_instance"}:
            return "basic_unet"
        return "swin_unet"

    def as_trunk_config(self) -> "HEAUNetModelConfig":
        trunk_config = replace(self)
        trunk_config.name = self.trunk_name()
        trunk_config.trunk = TrunkConfig()
        trunk_config.backbone = TrunkConfig()
        return trunk_config

    def _apply_trunk_overrides(self) -> None:
        for trunk_like in (self.trunk, self.backbone):
            if trunk_like.operator_backend is not None:
                self.attention.operator_backend = trunk_like.operator_backend
            if trunk_like.fusion_mode is not None:
                self.attention.fusion_mode = trunk_like.fusion_mode
            if trunk_like.elevator_mode is not None:
                self.attention.elevator_mode = trunk_like.elevator_mode
            if trunk_like.hea_enabled_decoder_stages is not None:
                self.hea.enabled_decoder_stages = list(trunk_like.hea_enabled_decoder_stages)
            if trunk_like.hea_enabled is False:
                self.hea.enabled_decoder_stages = []
            if trunk_like.hea_enabled is True and trunk_like.hea_enabled_decoder_stages is None:
                self.hea.enabled_decoder_stages = list(self.hea.enabled_decoder_stages)

    def validate(self) -> None:
        self._apply_trunk_overrides()
        num_scales = len(self.channel_multipliers)
        if len(self.encoder_depths) != num_scales:
            raise ValueError(
                f"encoder_depths must have length {num_scales}, got {len(self.encoder_depths)}."
            )
        if len(self.decoder_depths) != num_scales - 1:
            raise ValueError(
                f"decoder_depths must have length {num_scales - 1}, got {len(self.decoder_depths)}."
            )
        if len(self.swin_stage_heads) != num_scales:
            raise ValueError(
                f"swin_stage_heads must have length {num_scales}, got {len(self.swin_stage_heads)}."
            )
        channels = [self.base_channels * mult for mult in self.channel_multipliers]
        for scale, (channels_at_scale, heads) in enumerate(zip(channels, self.swin_stage_heads)):
            if channels_at_scale % heads != 0:
                raise ValueError(
                    f"swin stage {scale} has channels {channels_at_scale}, which is not divisible by {heads} heads."
                )
        if self.is_instance_model():
            self.instance_head.validate(self.num_classes)
        if self.is_dense_ssl_model():
            self.latent.validate()
            _validate_latent_hooks_for_pyramid(
                self.latent.resolved_sources(),
                len(self.channel_multipliers),
            )
            self.lejepa.validate()

        trunk_name = self.trunk_name()
        if trunk_name == "swin_unet" and self.is_instance_model():
            raise ValueError("Instance models currently support only HEA or basic U-Net trunks.")
        if self.is_dense_ssl_model() and trunk_name != "hea_unet":
            raise ValueError("hea_dense_lejepa currently supports only the HEA backbone.")
        if trunk_name != "hea_unet":
            return
        self.semantic_memory.validate()
        self.hea.validate(len(self.semantic_memory.enabled_scales))
        for scale in self.semantic_memory.enabled_scales:
            if scale <= 0 or scale >= num_scales:
                raise ValueError(
                    f"semantic memory scale {scale} is invalid for {num_scales} encoder stages."
                )
        for stage in self.hea.enabled_decoder_stages:
            if stage < 0 or stage >= num_scales - 1:
                raise ValueError(
                    f"decoder stage {stage} is invalid for {num_scales - 1} decoder stages."
                )


@dataclass
class TrainingConfig:
    image_size: int = 256


@dataclass
class HEAExperimentConfig:
    model: HEAUNetModelConfig = field(default_factory=HEAUNetModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    targets: InstanceTargetConfig = field(default_factory=InstanceTargetConfig)
    loss: BottomUpInstanceLossConfig = field(default_factory=BottomUpInstanceLossConfig)
    postprocess: InstancePostprocessConfig = field(default_factory=InstancePostprocessConfig)

    def validate(self) -> None:
        self.model.validate()
        self.targets.validate()
        self.loss.validate()
        self.postprocess.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_value(field_type: Any, value: Any) -> Any:
    origin = get_origin(field_type)
    if value is None and origin in (UnionType, Union):
        if type(None) in get_args(field_type):
            return None
    if is_dataclass(field_type):
        return _dataclass_from_dict(field_type, value)
    if origin in (UnionType, Union):
        union_args = [arg for arg in get_args(field_type) if arg is not type(None)]
        if len(union_args) == 1:
            return _coerce_value(union_args[0], value)
        return value
    if origin is list:
        item_type = get_args(field_type)[0]
        return [_coerce_value(item_type, item) for item in value]
    if origin is Literal:
        return value
    return value


def _dataclass_from_dict(cls: type[T], payload: dict[str, Any]) -> T:
    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        item_type = type_hints.get(item.name, item.type)
        if item.name not in payload:
            if item.default is not MISSING or item.default_factory is not MISSING:  # type: ignore[attr-defined]
                continue
            raise KeyError(f"Missing required config field: {item.name}")
        kwargs[item.name] = _coerce_value(item_type, payload[item.name])
    return cls(**kwargs)


def experiment_config_from_dict(payload: dict[str, Any]) -> HEAExperimentConfig:
    """Parse an experiment config from a Python dict."""
    config = _dataclass_from_dict(HEAExperimentConfig, payload)
    config.validate()
    return config


def load_experiment_config(path: str | Path) -> HEAExperimentConfig:
    """Load an experiment config from YAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("YAML config must contain a mapping at the top level.")
    return experiment_config_from_dict(payload)
