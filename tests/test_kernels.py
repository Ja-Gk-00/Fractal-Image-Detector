import importlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fractal_detector import kernels, native_available
from fractal_detector.compression import transform


@pytest.mark.native
@pytest.mark.skipif(not native_available(), reason="C extension unavailable")
@given(seed=st.integers(0, 10000), size=st.sampled_from([2, 4, 8]))
@settings(max_examples=20, deadline=None)
def test_native_and_python_agree(seed: int, size: int) -> None:
    rng = np.random.default_rng(seed)
    pixels = rng.random((32, 32))
    np.testing.assert_allclose(
        kernels.tile_stats(pixels, size, "native"),
        kernels.tile_stats(pixels, size, "python"),
        atol=1e-12,
    )
    ranges, domains = rng.random((8, 16)), rng.random((11, 16))
    np.testing.assert_allclose(
        kernels.affine_match(ranges, domains, 0.95, "native"),
        kernels.affine_match(ranges, domains, 0.95, "python"),
        atol=1e-12,
    )


@pytest.mark.parametrize("backend", ["python", "native"])
def test_affine_exact_fit_and_flat_domain(backend: str) -> None:
    if backend == "native" and not native_available():
        pytest.skip("C extension unavailable")
    domain = np.linspace(0, 1, 16).reshape(1, -1)
    result = kernels.affine_match(0.7 * domain + 0.1, domain, 0.95, backend)[0]
    np.testing.assert_allclose(result[:3], [0, 0.7, 0.1], atol=1e-14)
    result = kernels.affine_match(np.full((1, 16), 0.8), np.full((1, 16), 0.3), 0.95, backend)[0]
    np.testing.assert_allclose(result[:3], [0, 0, 0.8], atol=1e-14)


def test_backend_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernels, "_native", None)
    assert kernels.resolve_backend("auto") == "python"
    with pytest.raises(RuntimeError, match="unavailable"):
        kernels.resolve_backend("native")
    with pytest.raises(ValueError):
        kernels.resolve_backend("cuda")


@pytest.mark.parametrize("size", [0, -1, 3, 64, True])
def test_bad_tile_sizes(size: int) -> None:
    with pytest.raises(ValueError):
        kernels.tile_stats(np.ones((32, 32)), size)


def test_bad_matrices() -> None:
    for data in (np.array([]), np.full((2, 2), np.nan), np.full((2, 2), 2.0)):
        with pytest.raises(ValueError):
            kernels.tile_stats(data, 2)
    with pytest.raises(ValueError):
        kernels.affine_match(np.zeros((1, 4)), np.zeros((1, 8)), 0.9)
    with pytest.raises(ValueError):
        kernels.affine_match(np.zeros((1, 4)), np.zeros((1, 4)), 1.0)


@pytest.mark.native
@pytest.mark.skipif(not native_available(), reason="C extension unavailable")
def test_c_boundary_rejects_invalid_buffers() -> None:
    native = importlib.import_module("fractal_detector._native")
    for data in (
        np.ones((2, 2), dtype=np.float32),
        np.ones(4),
        np.empty((0, 2)),
        np.full((2, 2), np.inf),
        np.ones((2, 2), dtype=">f8"),
        np.ones((4, 4))[:, ::2],
    ):
        with pytest.raises((ValueError, BufferError)):
            native.tile_stats(data, 2)
    for size in (0, -1, 3, 2**62):
        with pytest.raises(ValueError):
            native.tile_stats(np.zeros((4, 4)), size)
    with pytest.raises(ValueError):
        native.affine_match(np.zeros((1, 4)), np.zeros((1, 8)), 0.9)
    for clip in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError):
            native.affine_match(np.zeros((1, 4)), np.zeros((1, 4)), clip)


def test_transform_order_matches_fastfractal() -> None:
    block = np.arange(9, dtype=np.float64).reshape(3, 3)
    expected = (
        block,
        block[::-1, :].T,
        block[::-1, ::-1],
        block.T[::-1, :],
        block[:, ::-1],
        block.T[::-1, ::-1],
        block[::-1, :],
        block.T,
    )
    for i in range(8):
        np.testing.assert_array_equal(transform(block, i), expected[i])
    with pytest.raises(ValueError):
        transform(block, 8)
