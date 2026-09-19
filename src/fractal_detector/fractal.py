import numpy as np

from fractal_detector.io import FloatArray
from fractal_detector.kernels import tile_stats
from fractal_detector.records import DimensionEstimate


def linear_fit(x: FloatArray, y: FloatArray) -> tuple[float, float, float]:
    dx, dy = x - x.mean(), y - y.mean()
    slope = float(dx @ dy / (dx @ dx))
    residual = dy - slope * dx
    sse, sst = float(residual @ residual), float(dy @ dy)
    r2 = 1.0 if sst < 1e-24 else max(0.0, 1.0 - sse / sst)
    stderr = float(np.sqrt(max(0.0, sse) / (len(x) - 2) / (dx @ dx)))
    return slope, r2, stderr


def estimate_dimension(scales: tuple[int, ...], counts: tuple[float, ...]) -> DimensionEstimate:
    if len(scales) != len(counts) or len(scales) < 3:
        raise ValueError("at least three scale/count pairs are required")
    if any(s <= 0 for s in scales) or tuple(sorted(set(scales))) != scales:
        raise ValueError("scales must be positive, distinct, and increasing")
    values = np.asarray(counts, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("counts must be finite and nonnegative")
    valid = values > 0
    if np.count_nonzero(valid) < 3:
        return DimensionEstimate(
            value=None,
            r_squared=None,
            slope_standard_error=None,
            scales=scales,
            counts=counts,
            reason="fewer than three occupied scales",
        )
    x = -np.log(np.asarray(scales, dtype=np.float64)[valid])
    slope, r2, stderr = linear_fit(x, np.log(values[valid]))
    return DimensionEstimate(
        value=slope, r_squared=r2, slope_standard_error=stderr, scales=scales, counts=counts
    )


def box_dimension(
    mask: FloatArray, scales: tuple[int, ...], backend: str = "auto"
) -> DimensionEstimate:
    if not np.isin(mask, [0.0, 1.0]).all():
        raise ValueError("box_dimension requires a binary mask")
    counts = tuple(float(np.count_nonzero(tile_stats(mask, s, backend)[:, 2])) for s in scales)
    return estimate_dimension(scales, counts)


def differential_dimension(
    image: FloatArray, scales: tuple[int, ...], backend: str = "auto"
) -> DimensionEstimate:
    if image.ndim != 2 or image.shape[0] != image.shape[1] or image.min() < 0:
        raise ValueError("differential_dimension requires a square image in [0,1]")
    counts: list[float] = []
    side = image.shape[0]
    for size in scales:
        stats = tile_stats(image, size, backend)
        height = size / side
        low = np.floor(np.minimum(stats[:, 0], np.nextafter(1.0, 0)) / height)
        high = np.floor(np.minimum(stats[:, 1], np.nextafter(1.0, 0)) / height)
        counts.append(float(np.sum(high - low + 1)))
    return estimate_dimension(scales, tuple(counts))


def mass_features(
    mass: FloatArray, scales: tuple[int, ...], backend: str
) -> tuple[dict[str, float], dict[str, object]]:
    lacunarity: list[float] = []
    moments: list[list[float]] = []
    for size in scales:
        boxes = tile_stats(mass, size, backend)[:, 2]
        total = float(boxes.sum())
        if total <= 1e-15:
            lacunarity.append(0.0)
            moments.append([0.0, 0.0, 0.0])
            continue
        p = boxes[boxes > 0] / total
        lacunarity.append(float(np.mean(boxes**2) / np.mean(boxes) ** 2))
        moments.append(
            [float(np.log(len(p))), float(-np.sum(p * np.log(p))), float(-np.log(np.sum(p**2)))]
        )
    curves = np.asarray(moments)
    x = -np.log(np.asarray(scales, dtype=np.float64))
    features = {f"residual_d{q}": linear_fit(x, curves[:, q])[0] for q in range(3)}
    features["lacunarity_mean"] = float(np.mean(lacunarity))
    features["lacunarity_std"] = float(np.std(lacunarity))
    return features, {
        "lacunarity": lacunarity,
        "renyi_entropy_curves": moments,
        "mass_empty": bool(mass.sum() <= 1e-15),
    }
