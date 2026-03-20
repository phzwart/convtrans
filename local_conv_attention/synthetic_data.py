"""Small synthetic test-data generators for local feature experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Literal, Sequence

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


def shift_image_2d_zero_pad(image_1hw: Tensor, dy: int, dx: int) -> Tensor:
    """Translate a single-channel map ``[1, H, W]`` by integer ``(dy, dx)`` pixels.

    Positive ``dy`` moves content **down**, positive ``dx`` moves content **right**.
    Out-of-bounds source is filled with **zero** (same border style as :func:`rotate_image_2d`).
    """
    if image_1hw.dim() != 3 or image_1hw.size(0) != 1:
        raise ValueError(f"shift_image_2d_zero_pad expects [1, H, W], got {tuple(image_1hw.shape)}.")
    _, h, w = image_1hw.shape
    out = torch.zeros_like(image_1hw)
    dy_i, dx_i = int(dy), int(dx)
    if dy_i >= 0:
        y_d0, y_d1 = dy_i, h
        y_s0, y_s1 = 0, h - dy_i
    else:
        y_d0, y_d1 = 0, h + dy_i
        y_s0, y_s1 = -dy_i, h
    if dx_i >= 0:
        x_d0, x_d1 = dx_i, w
        x_s0, x_s1 = 0, w - dx_i
    else:
        x_d0, x_d1 = 0, w + dx_i
        x_s0, x_s1 = -dx_i, w
    if y_d1 > y_d0 and x_d1 > x_d0:
        out[:, y_d0:y_d1, x_d0:x_d1] = image_1hw[:, y_s0:y_s1, x_s0:x_s1]
    return out


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


def rotate_tensor_nchw(
    x_bchw: Tensor,
    degrees: float,
    *,
    padding_mode: Literal["zeros", "border", "reflection"] = "zeros",
) -> Tensor:
    """Rotate a batch ``[B, C, H, W]`` by ``degrees`` CCW about the image center (shared warp).

    Uses ``grid_sample`` with the chosen ``padding_mode``.  ``reflection`` / ``border`` reduce
    dark wedges vs ``zeros`` when composing rotate → op → inverse rotate in view generation.
    """
    if x_bchw.dim() != 4:
        raise ValueError(f"rotate_tensor_nchw expects [B, C, H, W], got {tuple(x_bchw.shape)}.")
    b, _, height, width = x_bchw.shape
    if abs(float(degrees)) < 1e-8:
        return x_bchw
    device = x_bchw.device
    dtype = x_bchw.dtype
    angle = math.radians(float(degrees))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    ys, xs = torch.meshgrid(yy, xx, indexing="ij")
    x_src = xs * cos_a + ys * sin_a
    y_src = -xs * sin_a + ys * cos_a
    grid = torch.stack((x_src, y_src), dim=-1).expand(b, -1, -1, -1)

    return F.grid_sample(
        x_bchw,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


class DiscSquareDataset(Dataset):
    """Dataset that repeats the 8 canonical disc/two-square types."""

    def __init__(
        self,
        *,
        repeats_per_type: int = 1,
        image_size: int = 128,
        radii: Sequence[int] = (28, 40),
        square_sizes: Sequence[int] = (8, 14),
        random_shift: bool = True,
        shift_px_range: tuple[int, int] = (-32, 32),
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
        s_lo, s_hi = shift_px_range
        if s_hi < s_lo:
            raise ValueError("shift_px_range must satisfy low <= high.")
        self.image_size = image_size
        self.random_shift = random_shift
        self.shift_px_range = (int(s_lo), int(s_hi))
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
        shift_dy = 0
        shift_dx = 0
        if self.random_shift:
            lo, hi = self.shift_px_range
            high = hi + 1  # ``torch.randint`` upper bound is exclusive
            if self.generator is not None:
                shift_dy = int(torch.randint(lo, high, (1,), generator=self.generator).item())
                shift_dx = int(torch.randint(lo, high, (1,), generator=self.generator).item())
            else:
                shift_dy = int(torch.randint(lo, high, (1,)).item())
                shift_dx = int(torch.randint(lo, high, (1,)).item())
            image = shift_image_2d_zero_pad(image, shift_dy, shift_dx)

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
            "shift_dy": shift_dy,
            "shift_dx": shift_dx,
            "rotation_deg": rotation_deg,
        }
