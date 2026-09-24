import numpy as np
import pytest

from fractal_detector import box_dimension, differential_dimension, estimate_dimension
from fractal_detector.fractal import mass_features


@pytest.mark.parametrize("dimension", [0, 1, 2])
def test_known_box_dimensions(dimension: int) -> None:
    mask = np.zeros((256, 256))
    if dimension == 0:
        mask[0, 0] = 1
    elif dimension == 1:
        mask[0, :] = 1
    else:
        mask[:] = 1
    result = box_dimension(mask, (1, 2, 4, 8, 16, 32), "python")
    assert result.value == pytest.approx(dimension, abs=1e-12)
    assert result.r_squared == pytest.approx(1)


def test_sierpinski_carpet() -> None:
    tile = np.ones((3, 3))
    tile[1, 1] = 0
    carpet = tile
    for _ in range(4):
        carpet = np.kron(carpet, tile)
    result = box_dimension(carpet, (1, 3, 9, 27, 81), "python")
    assert result.value == pytest.approx(np.log(8) / np.log(3), abs=1e-12)


@pytest.mark.parametrize("level", [0.0, 0.5, 1.0])
def test_flat_grayscale_surface_is_two_dimensional(level: float) -> None:
    result = differential_dimension(np.full((64, 64), level), (2, 4, 8, 16), "python")
    assert result.value == pytest.approx(2.0)


def test_random_surface_is_rougher_than_flat() -> None:
    result = differential_dimension(
        np.random.default_rng(1337).random((256, 256)), (2, 4, 8, 16, 32)
    )
    assert result.value is not None and 2.5 < result.value < 3.1


def test_empty_mask_has_no_dimension() -> None:
    result = box_dimension(np.zeros((32, 32)), (2, 4, 8))
    assert result.value is None
    assert result.reason


@pytest.mark.parametrize(
    ("scales", "counts"),
    [
        ((1, 2), (2.0, 1.0)),
        ((0, 2, 4), (3.0, 2.0, 1.0)),
        ((1, 1, 4), (3.0, 2.0, 1.0)),
        ((1, 2, 4), (-1.0, 2.0, 1.0)),
        ((1, 2, 4), (float("nan"), 2.0, 1.0)),
    ],
)
def test_invalid_regression_inputs(scales: tuple[int, ...], counts: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        estimate_dimension(scales, counts)


def test_invalid_fractal_inputs() -> None:
    with pytest.raises(ValueError, match="binary"):
        box_dimension(np.full((32, 32), 0.5), (2, 4, 8))
    with pytest.raises(ValueError, match="square"):
        differential_dimension(np.zeros((32, 64)), (2, 4, 8))


def test_uniform_mass_dimensions_and_lacunarity() -> None:
    features, _ = mass_features(np.ones((32, 32)), (2, 4, 8), "python")
    for q in range(3):
        assert features[f"residual_d{q}"] == pytest.approx(2.0)
    assert features["lacunarity_mean"] == pytest.approx(1.0)


def test_empty_mass_is_explicit() -> None:
    features, diagnostic = mass_features(np.zeros((32, 32)), (2, 4, 8), "python")
    assert diagnostic["mass_empty"]
    assert features["residual_d1"] == 0
