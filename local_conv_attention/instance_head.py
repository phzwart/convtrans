"""Bottom-up dense prediction heads for instance segmentation."""

from __future__ import annotations

from torch import Tensor, nn

from .config import InstanceHeadConfig
from .encoder import ActKind, NormKind, ConvNormAct2d, ResidualConvBlock2d


def _make_refine_stack(
    dim: int,
    depth: int,
    *,
    norm: NormKind,
    act: ActKind,
) -> nn.Module:
    if depth <= 0:
        return nn.Identity()
    return nn.Sequential(
        *[ResidualConvBlock2d(dim, dim, norm=norm, act=act) for _ in range(depth)]
    )


class BottomUpInstanceHead2d(nn.Module):
    """A lightweight head that predicts foreground, centers, and offsets."""

    def __init__(
        self,
        in_channels: int,
        config: InstanceHeadConfig,
        *,
        norm: NormKind = "batchnorm",
        act: ActKind = "gelu",
        num_classes: int = 0,
    ) -> None:
        super().__init__()
        config.validate(num_classes)
        self.config = config
        shared_dim = config.shared_dim

        self.input_proj = (
            nn.Identity()
            if in_channels == shared_dim
            else ConvNormAct2d(in_channels, shared_dim, kernel_size=1, norm=norm, act=act)
        )
        self.shared = _make_refine_stack(shared_dim, config.shared_depth, norm=norm, act=act)
        self.foreground_branch = self._make_branch(
            shared_dim,
            config.branch_depth,
            config.foreground_channels,
            norm=norm,
            act=act,
        )
        self.center_branch = self._make_branch(
            shared_dim,
            config.branch_depth,
            config.center_channels,
            norm=norm,
            act=act,
        )
        self.offset_branch = self._make_branch(
            shared_dim,
            config.branch_depth,
            config.offset_channels,
            norm=norm,
            act=act,
        )
        self.boundary_branch = None
        if config.boundary_branch:
            self.boundary_branch = self._make_branch(shared_dim, config.branch_depth, 1, norm=norm, act=act)
        self.semantic_branch = None
        if config.semantic_branch:
            self.semantic_branch = self._make_branch(
                shared_dim,
                config.branch_depth,
                num_classes,
                norm=norm,
                act=act,
            )

    @staticmethod
    def _make_branch(
        in_channels: int,
        depth: int,
        out_channels: int,
        *,
        norm: NormKind,
        act: ActKind,
    ) -> nn.Module:
        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(depth):
            layers.append(ResidualConvBlock2d(current_channels, in_channels, norm=norm, act=act))
            current_channels = in_channels
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        shared = self.shared(self.input_proj(x))
        outputs = {
            "foreground_logits": self.foreground_branch(shared),
            "center_logits": self.center_branch(shared),
            "offsets": self.offset_branch(shared),
        }
        outputs["center_heatmap_logits"] = outputs["center_logits"]
        if self.boundary_branch is not None:
            outputs["boundary_logits"] = self.boundary_branch(shared)
        if self.semantic_branch is not None:
            outputs["semantic_logits"] = self.semantic_branch(shared)
        return outputs
