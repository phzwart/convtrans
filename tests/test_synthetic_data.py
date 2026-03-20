from __future__ import annotations

import torch

from local_conv_attention import DiscSquareDataset, generate_disc_square_image, make_disc_square_types


def test_make_disc_square_types_returns_eight_combinations() -> None:
    types = make_disc_square_types()
    assert len(types) == 8
    assert len({(item.radius, item.square_size, item.near_density, item.far_density) for item in types}) == 8


def test_generated_image_has_expected_values() -> None:
    spec = make_disc_square_types()[0]
    image = generate_disc_square_image(spec, image_size=128)
    assert image.shape == (1, 128, 128)
    values = torch.unique(image)
    torch.testing.assert_close(values, torch.tensor([0.0, min(spec.near_density, spec.far_density), max(spec.near_density, spec.far_density), 1.0]))


def test_near_square_is_closer_to_disc_edge_than_far_square() -> None:
    spec = next(spec for spec in make_disc_square_types() if spec.radius == 28 and spec.near_density == 0.5 and spec.square_size == 8)
    image = generate_disc_square_image(spec, image_size=128)[0]

    far_positions = torch.nonzero(image == spec.far_density, as_tuple=False).float()
    near_positions = torch.nonzero(image == spec.near_density, as_tuple=False).float()
    center = torch.tensor([64.0, 64.0])
    far_radius = torch.linalg.vector_norm(far_positions - center, dim=1).max()
    near_radius = torch.linalg.vector_norm(near_positions - center, dim=1).max()
    assert near_radius > far_radius


def test_disc_square_dataset_repeats_types() -> None:
    dataset = DiscSquareDataset(repeats_per_type=2, random_rotation=False)
    assert len(dataset) == 16
    sample = dataset[0]
    assert sample["image"].shape == (1, 128, 128)
    assert sample["square_size"] in {8, 14}
    assert sample["rotation_deg"] == 0.0


def test_rotate_image_2d_zero_degrees_is_unchanged() -> None:
    from local_conv_attention import rotate_image_2d

    spec = make_disc_square_types()[0]
    image = generate_disc_square_image(spec, image_size=32)
    torch.testing.assert_close(rotate_image_2d(image, 0.0), image)


def test_disc_square_dataset_rotation_deterministic_with_generator() -> None:
    g = torch.Generator()
    g.manual_seed(12345)
    ds = DiscSquareDataset(repeats_per_type=1, random_rotation=True, generator=g)
    a = ds[0]["rotation_deg"]
    g.manual_seed(12345)
    ds2 = DiscSquareDataset(repeats_per_type=1, random_rotation=True, generator=g)
    b = ds2[0]["rotation_deg"]
    assert a == b
    assert 0.0 <= a < 360.0
