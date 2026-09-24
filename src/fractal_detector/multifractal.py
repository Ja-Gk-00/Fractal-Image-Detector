import numpy as np
from scipy.special import logsumexp

from fractal_detector.fractal import linear_fit
from fractal_detector.io import FloatArray
from fractal_detector.kernels import tile_stats


def multifractal_spectrum(
    mass: FloatArray,
    scales: tuple[int, ...],
    orders: tuple[float, ...],
    *,
    backend: str = "auto",
    mass_floor: float = 0.0,
) -> tuple[dict[str, float], dict[str, object]]:
    if mass.ndim != 2 or not mass.size or mass.shape[0] != mass.shape[1]:
        raise ValueError("mass must be square")
    if not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("mass must be finite and nonnegative")
    if len(scales) < 3 or tuple(sorted(set(scales))) != scales:
        raise ValueError("at least three increasing scales are required")
    if any(s <= 0 or mass.shape[0] % s for s in scales):
        raise ValueError("scales must be positive divisors of the image size")
    q = np.asarray(orders, dtype=np.float64)
    if len(q) < 3 or not np.isfinite(q).all() or np.any(np.diff(q) <= 0):
        raise ValueError("at least three finite increasing moment orders are required")
    if not np.isfinite(mass_floor) or not 0 <= mass_floor <= 0.01:
        raise ValueError("mass_floor must be in [0, 0.01]")
    amplitude = float(np.max(mass))
    empty = amplitude == 0.0
    if not empty:
        mass = mass / amplitude
    curves, information, occupancy = [], [], []
    for size in scales:
        boxes = tile_stats(mass, size, backend)[:, 2]
        positive = boxes[boxes > 0]
        occupancy.append(len(positive))
        if empty:
            curves.append(np.zeros_like(q))
            information.append(0.0)
            continue
        p = positive / positive.sum()
        if mass_floor:
            p = np.maximum(p, mass_floor)
            p /= p.sum()
        log_p = np.log(p)
        curves.append(logsumexp(q[:, None] * log_p, axis=1))
        information.append(float(np.sum(p * log_p)))
    x = np.log(np.asarray(scales, dtype=np.float64) / mass.shape[0])
    partition = np.asarray(curves)
    fits = np.asarray([linear_fit(x, partition[:, i]) for i in range(len(q))])
    tau, r2, stderr = fits.T
    d = np.asarray(
        [
            linear_fit(x, np.asarray(information))[0] if order == 1 else t / (order - 1)
            for order, t in zip(q, tau, strict=True)
        ]
    )
    alpha = np.gradient(tau, q, edge_order=2)
    spectrum = q * alpha - tau
    features = {
        "multifractal_width": float(np.ptp(alpha)),
        "multifractal_alpha_peak": float(alpha[np.argmax(spectrum)]),
        "multifractal_f_peak": float(np.max(spectrum)),
        "multifractal_d_span": float(np.ptp(d)),
        "multifractal_fit_min": float(np.min(r2)),
        "multifractal_valid": float(not empty),
    }
    features.update({f"multifractal_d_{i}": float(value) for i, value in enumerate(d)})
    return features, {
        "orders": list(orders),
        "scales": list(scales),
        "tau": tau.tolist(),
        "generalized_dimensions": d.tolist(),
        "alpha": alpha.tolist(),
        "spectrum": spectrum.tolist(),
        "r_squared": r2.tolist(),
        "slope_standard_errors": stderr.tolist(),
        "log_partition": partition.tolist(),
        "occupied_boxes": occupancy,
        "empty": empty,
        "empty_box_policy": "exclude",
        "mass_floor": mass_floor,
    }


def sliding_lacunarity(mass: FloatArray, scales: tuple[int, ...], stride: int = 1) -> list[float]:
    if (
        isinstance(stride, bool)
        or not isinstance(stride, int)
        or stride < 1
        or mass.ndim != 2
        or not mass.size
        or not np.isfinite(mass).all()
        or np.any(mass < 0)
    ):
        raise ValueError("sliding lacunarity requires nonnegative finite mass and positive stride")
    amplitude = float(np.max(mass))
    if amplitude:
        mass = mass / amplitude
    integral = np.pad(mass, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    values = []
    for size in scales:
        if size < 1 or size > min(mass.shape):
            raise ValueError("window sizes must fit the image")
        boxes = (
            integral[size:, size:]
            - integral[:-size, size:]
            - integral[size:, :-size]
            + integral[:-size, :-size]
        )[::stride, ::stride]
        mean = float(boxes.mean())
        values.append(float(1 + boxes.var() / mean**2) if mean > 1e-15 else 0.0)
    return values
