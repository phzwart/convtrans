"""Small synthetic test-data generators for local feature experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DiscSquareType:
    """Specification for one synthetic disc-plus-two-squares image type."""

    radius: int
    near_density: float
    far_density: float
    square_size: int
    type_index: int


def make_disc_square_types(
    *,
    radii: Sequence[int] = (28, 40),
    square_sizes: Sequence[int] = (8, 14),
    density_assignments: Sequence[tuple[float, float]] = ((0.5, 0.25), (0.25, 0.5)),
) -> list[DiscSquareType]:
    """Return the 8 canonical disc/two-square combinations.

    The 8 types are:
    - 2 disc radii
    - 2 square sizes
    - 2 assignments of densities across near/far squares
    """
    types = []
    for type_index, (radius, square_size, (near_density, far_density)) in enumerate(
        product(radii, square_sizes, density_assignments)
    ):
        types.append(
            DiscSquareType(
                radius=int(radius),
                near_density=float(near_density),
                far_density=float(far_density),
                square_size=int(square_size),
                type_index=type_index,
            )
        )
    return types


def generate_disc_square_image(
    spec: DiscSquareType,
    *,
    image_size: int = 128,
    disc_density: float = 1.0,
    background_density: float = 0.0,
) -> Tensor:
    """Generate one 2D image with a constant disc and two squares inside it.

    The disc is centered in the image. The square is axis-aligned and placed
    either near the disc edge or farther toward the center along the horizontal
    axis so the geometry stays deterministic and easy to test.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if spec.square_size <= 0:
        raise ValueError("square_size must be positive.")
    if spec.square_size >= image_size:
        raise ValueError("square_size must be smaller than image_size.")

    center = image_size // 2
    yy, xx = torch.meshgrid(
        torch.arange(image_size, dtype=torch.float32),
        torch.arange(image_size, dtype=torch.float32),
        indexing="ij",
    )
    distance = torch.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    image = torch.full((image_size, image_size), background_density, dtype=torch.float32)
    disc_mask = distance <= float(spec.radius)
    image[disc_mask] = disc_density

    half = spec.square_size // 2
    margin = 3
    near_center_x = center + spec.radius - half - margin
    far_center_x = center - max(half + margin, int(round(spec.radius * 0.35)))
    near_center_y = center
    far_center_y = center

    near_y0 = near_center_y - half
    near_x0 = near_center_x - half
    near_y1 = near_y0 + spec.square_size
    near_x1 = near_x0 + spec.square_size

    far_y0 = far_center_y - half
    far_x0 = far_center_x - half
    far_y1 = far_y0 + spec.square_size
    far_x1 = far_x0 + spec.square_size

    image[far_y0:far_y1, far_x0:far_x1] = spec.far_density
    image[near_y0:near_y1, near_x0:near_x1] = spec.near_density
    return image.unsqueeze(0)


def rotate_image_2d(image_1hw: Tensor, degrees: float) -> Tensor:
    """Rotate a single-channel map ``[1, H, W]`` by ``degrees`` counter-clockwise about the center.

    Uses ``grid_sample`` (bilinear, zero padding outside the original square).  ``degrees`` may be
    any float; common use is uniform random in ``[0, 360)``.
    """
    if image_1hw.dim() != 3:
        raise ValueError(f"rotate_image_2d expects [1, H, W], got {tuple(image_1hw.shape)}.")
    if image_1hw.size(0) != 1:
        raise ValueError("rotate_image_2d currently supports a single channel (leading dim 1).")

    _, height, width = image_1hw.shape
    device = image_1hw.device
    dtype = image_1hw.dtype
    angle = math.radians(float(degrees))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    ys, xs = torch.meshgrid(yy, xx, indexing="ij")
    # Inverse warp: output pixel samples input at R(-θ) applied to normalized output coords.
    x_src = xs * cos_a + ys * sin_a
    y_src = -xs * sin_a + ys * cos_a
    grid = torch.stack((x_src, y_src), dim=-1).unsqueeze(0)

    out = F.grid_sample(
        image_1hw.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return out.squeeze(0)


class DiscSquareDataset(Dataset):
    """Dataset that repeats the 8 canonical disc/two-square types."""

    def __init__(
        self,
        *,
        repeats_per_type: int = 1,
        image_size: int = 128,
        radii: Sequence[int] = (28, 40),
        square_sizes: Sequence[int] = (8, 14),
        random_rotation: bool = True,
        rotation_range_deg: tuple[float, float] = (0.0, 360.0),
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if repeats_per_type <= 0:
            raise ValueError("repeats_per_type must be positive.")
        low, high = rotation_range_deg
        if high < low:
            raise ValueError("rotation_range_deg must satisfy low <= high.")
        self.image_size = image_size
        self.random_rotation = random_rotation
        self.rotation_range_deg = (float(low), float(high))
        self.generator = generator
        self.types = make_disc_square_types(radii=radii, square_sizes=square_sizes)
        self.samples = [spec for spec in self.types for _ in range(repeats_per_type)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | int | float | str]:
        spec = self.samples[index]
        image = generate_disc_square_image(
            spec,
            image_size=self.image_size,
        )
        rotation_deg = 0.0
        if self.random_rotation:
            lo, hi = self.rotation_range_deg
            if self.generator is not None:
                u = torch.rand((), generator=self.generator)
            else:
                u = torch.rand(())
            rotation_deg = lo + (hi - lo) * float(u.item())
            image = rotate_image_2d(image, rotation_deg)
        return {
            "image": image,
            "type_index": spec.type_index,
            "radius": spec.radius,
            "near_density": spec.near_density,
            "far_density": spec.far_density,
            "square_size": spec.square_size,
            "rotation_deg": rotation_deg,
        }
