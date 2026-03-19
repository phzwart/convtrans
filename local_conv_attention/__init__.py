"""Local convolutional attention toolkit."""

from .attention import (
    ConvLocalAttention2d,
    LocalAttention2d,
    LocalSelfAttention2d,
    MultiHeadLocalAttention2d,
    ShiftLocalAttention2d,
    local_attention_from_qkv,
)
from .backbone import HEABackbone
from .baselines import BasicUNet
from .block import LocalTransformerBlock2d, ReferenceLocalTransformerBlock2d
from .config import (
    BottomUpInstanceLossConfig,
    DenseInvarianceConfig,
    DenseLatentConfig,
    DenseLeJEPAObjectiveConfig,
    DenseSIGRegConfig,
    DenseViewConfig,
    DenseViewCorruptionConfig,
    HEAExperimentConfig,
    HEAUNetModelConfig,
    InstanceHeadConfig,
    InstancePostprocessConfig,
    InstanceTargetConfig,
    default_all_latent_hooks,
    experiment_config_from_dict,
    load_experiment_config,
)
from .dense_lejepa import DenseLatentProjector2d, DenseLeJEPAModel
from .explain import HEAExplainer
from .factory import build_instance_loss, build_model, build_model_from_yaml
from .hea import HEAFusionBlock2d, HierarchicalElevatorAttention2d, SemanticMemoryBlock2d
from .instance_head import BottomUpInstanceHead2d
from .losses import BottomUpInstanceLoss
from .masks import flattened_local_attention_mask, local_validity_mask
from .ops import ConvShiftBank2d, NeighborhoodShift2d, ShiftBank2d
from .postprocess import decode_instances
from .reference import (
    FlattenedMaskedLocalAttention2d,
    ReferenceLocalAttention2d,
    ReferenceLocalSelfAttention2d,
    flattened_local_attention_from_qkv,
)
from .sigreg import SIGRegLoss
from .synthetic_data import DiscSquareDataset, DiscSquareType, generate_disc_square_image, make_disc_square_types
from .swin import SwinUNet
from .targets import (
    build_center_heatmap_target,
    build_foreground_target,
    build_instance_targets,
    build_offset_target,
)
from .unet import HEAUNet, HEAUNetInstanceModel
from .visualization import (
    combine_upsampled_heatmaps,
    overlay_heatmap_on_image,
    upsample_region_heatmap,
    visualize_explanation,
    visualize_signed_explanation,
)

__all__ = [
    "BasicUNet",
    "DenseInvarianceConfig",
    "DenseLatentConfig",
    "DenseLatentProjector2d",
    "DenseLeJEPAObjectiveConfig",
    "DenseLeJEPAModel",
    "DenseSIGRegConfig",
    "DenseViewConfig",
    "DenseViewCorruptionConfig",
    "DiscSquareDataset",
    "DiscSquareType",
    "FlattenedMaskedLocalAttention2d",
    "ConvLocalAttention2d",
    "ConvShiftBank2d",
    "HEABackbone",
    "BottomUpInstanceHead2d",
    "BottomUpInstanceLoss",
    "BottomUpInstanceLossConfig",
    "HEAExperimentConfig",
    "HEAFusionBlock2d",
    "HEAExplainer",
    "HEAUNet",
    "HEAUNetInstanceModel",
    "HEAUNetModelConfig",
    "HierarchicalElevatorAttention2d",
    "InstanceHeadConfig",
    "InstancePostprocessConfig",
    "InstanceTargetConfig",
    "LocalAttention2d",
    "LocalSelfAttention2d",
    "LocalTransformerBlock2d",
    "MultiHeadLocalAttention2d",
    "NeighborhoodShift2d",
    "ReferenceLocalAttention2d",
    "ReferenceLocalSelfAttention2d",
    "ReferenceLocalTransformerBlock2d",
    "SemanticMemoryBlock2d",
    "ShiftLocalAttention2d",
    "ShiftBank2d",
    "SIGRegLoss",
    "SwinUNet",
    "build_center_heatmap_target",
    "build_foreground_target",
    "build_instance_loss",
    "build_instance_targets",
    "build_model",
    "build_model_from_yaml",
    "build_offset_target",
    "decode_instances",
    "default_all_latent_hooks",
    "experiment_config_from_dict",
    "flattened_local_attention_from_qkv",
    "flattened_local_attention_mask",
    "generate_disc_square_image",
    "load_experiment_config",
    "local_attention_from_qkv",
    "local_validity_mask",
    "make_disc_square_types",
    "combine_upsampled_heatmaps",
    "overlay_heatmap_on_image",
    "upsample_region_heatmap",
    "visualize_explanation",
    "visualize_signed_explanation",
]
