"""Fixed non-learned neighborhood extraction operators."""

from __future__ import annotations

from typing import Dict, Literal, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .masks import local_validity_mask
from .utils import make_offsets, window_radius

BoundaryPadMode = Literal["zeros", "reflect"]


def pad_spatial_hw(x: Tensor, pad: int, mode: BoundaryPadMode) -> Tensor:
    """Pad the last two dims (H, W) of ``x`` symmetrically by ``pad``.

    ``zeros`` matches prior implicit conv/unfold zero padding.  ``reflect`` mirrors
    interior values at the boundary (PyTorch ``mode='reflect'``).  If ``reflect``
    is invalid because ``pad`` is not strictly smaller than H and W, falls back to
    ``replicate`` (edge-clamp) so tiny feature maps still run.

    For 5D tensors ``[B, Hg, C, H, W]`` (e.g. headed layout), reflect/replicate
    use a 6-tuple so only ``H`` and ``W`` are padded (PyTorch requires this for
    non-constant 5D padding).
    """
    if pad <= 0:
        return x
    *_, height, width = x.shape
    spec4 = (pad, pad, pad, pad)  # W then H for 4D NCHW
    if mode == "zeros":
        return F.pad(x, spec4)

    if x.dim() == 4:
        spec = spec4
    elif x.dim() == 5:
        spec = (pad, pad, pad, pad, 0, 0)  # W, H, then no pad on channel-like dim
    else:
        raise ValueError(f"pad_spatial_hw expects 4D or 5D input, got {x.dim()}D.")

    if pad < height and pad < width:
        return F.pad(x, spec, mode="reflect")
    return F.pad(x, spec, mode="replicate")


class NeighborhoodShift2d(nn.Module):
    """Extract an explicit bank of fixed spatial shifts from a 2D feature map.

    The operator is non-learned: it only pads and slices so that each offset in the
    local MxM stencil is aligned with the center position.
    """

    def __init__(
        self,
        window_size: int,
        dilation: int = 1,
        *,
        boundary_pad: BoundaryPadMode = "zeros",
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.dilation = dilation
        self.boundary_pad: BoundaryPadMode = boundary_pad
        self.offsets = make_offsets(window_size, dilation=dilation)
        self.padding = window_radius(window_size, dilation=dilation)

    def forward(
        self,
        x: Tensor,
        *,
        return_mask: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """Extract aligned neighbors from [B, heads, dim, H, W]."""
        if x.dim() != 5:
            raise ValueError(
                f"NeighborhoodShift2d expects [B, heads, dim, H, W], got {tuple(x.shape)}."
            )

        _, _, _, height, width = x.shape
        padded = pad_spatial_hw(x, self.padding, self.boundary_pad)

        neighbors = []
        for dy, dx in self.offsets:
            y0 = self.padding + dy
            x0 = self.padding + dx
            neighbors.append(padded[..., y0 : y0 + height, x0 : x0 + width])

        shifted = torch.stack(neighbors, dim=2)
        if not return_mask:
            return shifted

        if self.boundary_pad == "zeros":
            mask = local_validity_mask(
                height,
                width,
                self.window_size,
                dilation=self.dilation,
                device=x.device,
            )
        else:
            mask = torch.ones(
                (len(self.offsets), height, width),
                dtype=torch.bool,
                device=x.device,
            )
        return shifted, mask.view(1, 1, len(self.offsets), 1, height, width)


class ShiftBank2d(NeighborhoodShift2d):
    """Alias for NeighborhoodShift2d."""

    pass


class ConvShiftBank2d(nn.Module):
    """Extract a fixed bank of one-hot spatial shifts via grouped `conv2d`.

    This is still an exact fixed operator: the convolution kernels are non-learned
    one-hot selectors that only encode which local offset to read at each output
    position. The result is equivalent to an explicit shift bank but can use the
    backend's optimized convolution kernels on CPU, CUDA, or MPS.
    """

    def __init__(
        self,
        window_size: int,
        dilation: int = 1,
        *,
        boundary_pad: BoundaryPadMode = "zeros",
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.dilation = dilation
        self.boundary_pad: BoundaryPadMode = boundary_pad
        self.offsets = make_offsets(window_size, dilation=dilation)
        self.padding = window_radius(window_size, dilation=dilation)
        self.register_buffer(
            "base_kernels",
            self._build_base_kernels(window_size),
            persistent=False,
        )
        self._weight_cache: Dict[tuple[int, torch.device, torch.dtype], Tensor] = {}
        self._mask_cache: Dict[tuple[str, int, int, torch.device], Tensor] = {}

    @staticmethod
    def _build_base_kernels(window_size: int) -> Tensor:
        neighborhood = window_size * window_size
        kernels = torch.zeros(neighborhood, 1, window_size, window_size)
        index = 0
        for y in range(window_size):
            for x in range(window_size):
                kernels[index, 0, y, x] = 1.0
                index += 1
        return kernels

    def _expanded_weight(self, channels: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (channels, device, dtype)
        weight = self._weight_cache.get(key)
        if weight is None:
            base = self.base_kernels.to(device=device, dtype=dtype)
            # Repeat the same local stencil once per input channel for depthwise grouped conv.
            weight = base.repeat(channels, 1, 1, 1).contiguous()
            self._weight_cache[key] = weight
        return weight

    def forward(
        self,
        x: Tensor,
        *,
        return_mask: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """Extract aligned neighbors from [B, C, H, W] into [B, C, K, H, W]."""
        if x.dim() != 4:
            raise ValueError(f"ConvShiftBank2d expects [B, C, H, W], got {tuple(x.shape)}.")

        batch, channels, height, width = x.shape
        weight = self._expanded_weight(channels, x.device, x.dtype)
        if x.device.type in {"cuda", "mps"}:
            x = x.contiguous(memory_format=torch.channels_last)
        if self.boundary_pad == "zeros":
            x_in = x
            conv_pad = self.padding
        else:
            x_in = pad_spatial_hw(x, self.padding, self.boundary_pad)
            conv_pad = 0
        shifted = F.conv2d(
            x_in,
            weight,
            bias=None,
            stride=1,
            padding=conv_pad,
            dilation=self.dilation,
            groups=channels,
        )
        # Normalize layout before reshaping so eager and compiled backends see the
        # same stride pattern even when the conv kernel prefers channels_last.
        shifted = shifted.contiguous().reshape(batch, channels, len(self.offsets), height, width)
        if not return_mask:
            return shifted

        mask_key = (self.boundary_pad, height, width, x.device)
        mask = self._mask_cache.get(mask_key)
        if mask is None:
            if self.boundary_pad == "zeros":
                mask_tensor = local_validity_mask(
                    height,
                    width,
                    self.window_size,
                    dilation=self.dilation,
                    device=x.device,
                )
            else:
                mask_tensor = torch.ones(
                    (len(self.offsets), height, width),
                    dtype=torch.bool,
                    device=x.device,
                )
            mask = mask_tensor.view(1, 1, len(self.offsets), height, width)
            self._mask_cache[mask_key] = mask
        return shifted, mask
