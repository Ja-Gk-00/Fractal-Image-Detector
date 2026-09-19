import importlib
from typing import Literal

import numpy as np

from fractal_detector.io import FloatArray

try:
    _native = importlib.import_module("fractal_detector._native")
except ImportError:
    _native = None


def native_available() -> bool:
    return _native is not None


def resolve_backend(backend: str) -> Literal["native", "python"]:
    if backend not in {"auto", "native", "python"}:
        raise ValueError("backend must be auto, native, or python")
    if backend == "native" and _native is None:
        raise RuntimeError(
            "native extension unavailable; install with a C compiler or use backend='python'"
        )
    return "native" if _native is not None and backend != "python" else "python"


def _matrix(array: FloatArray) -> FloatArray:
    array = np.ascontiguousarray(array, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all() or np.abs(array).max() > 1:
        raise ValueError("expected nonempty finite matrix in [-1,1]")
    return array


def tile_stats(image: FloatArray, size: int, backend: str = "auto") -> FloatArray:
    image = _matrix(image)
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or any(d % size for d in image.shape)
    ):
        raise ValueError("tile size must be a positive integer dividing the image dimensions")
    if resolve_backend(backend) == "native" and _native is not None:
        return np.frombuffer(_native.tile_stats(image, size), dtype=np.float64).reshape(-1, 3)
    tiles = image.reshape(image.shape[0] // size, size, image.shape[1] // size, size)
    tiles = tiles.transpose(0, 2, 1, 3).reshape(-1, size * size)
    return np.column_stack((tiles.min(axis=1), tiles.max(axis=1), tiles.sum(axis=1)))


def affine_match(
    ranges: FloatArray, domains: FloatArray, clip: float, backend: str = "auto"
) -> FloatArray:
    ranges, domains = _matrix(ranges), _matrix(domains)
    if ranges.shape[1] != domains.shape[1] or not 0 < clip < 1:
        raise ValueError("block widths must match and clip must be in (0,1)")
    if resolve_backend(backend) == "native" and _native is not None:
        return np.frombuffer(_native.affine_match(ranges, domains, clip), dtype=np.float64).reshape(
            -1, 4
        )
    means = domains.mean(axis=1)
    centered = domains - means[:, None]
    variances = np.mean(centered**2, axis=1)
    output = np.empty((len(ranges), 4), dtype=np.float64)
    for i, block in enumerate(ranges):
        mean = block.mean()
        cov = np.mean(centered * (block - mean), axis=1)
        contrast = np.divide(cov, variances, out=np.zeros_like(cov), where=variances > 1e-15).clip(
            -clip, clip
        )
        error = np.maximum(0, np.var(block) + contrast**2 * variances - 2 * contrast * cov)
        best = int(np.argmin(error))
        output[i] = (error[best], contrast[best], mean - contrast[best] * means[best], best)
    return output
