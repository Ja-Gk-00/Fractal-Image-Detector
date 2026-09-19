import logging
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter

from fractal_detector.compression import compression_features
from fractal_detector.config import FeatureConfig
from fractal_detector.fractal import box_dimension, differential_dimension, mass_features
from fractal_detector.io import FloatArray, ImageInput, load_image, source_name
from fractal_detector.kernels import resolve_backend
from fractal_detector.records import FeatureResult

logger = logging.getLogger("fractal_detector")


def feature_names(config: FeatureConfig) -> tuple[str, ...]:
    names = [
        "gray_dimension",
        "gray_dimension_r2",
        "residual_dimension",
        "residual_dimension_r2",
        "edge_dimension",
        "edge_dimension_r2",
        "edge_dimension_valid",
        "residual_d0",
        "residual_d1",
        "residual_d2",
        "lacunarity_mean",
        "lacunarity_std",
        "residual_std",
        "residual_entropy",
        "spectrum_similarity_half",
        "spectrum_similarity_quarter",
    ]
    if config.compression:
        names += [
            "compression_mse_mean",
            "compression_mse_std",
            "compression_mse_p90",
            "compression_contrast_mean",
            "compression_contrast_std",
        ]
    return tuple(names)


def noise_residual(gray: FloatArray, config: FeatureConfig) -> FloatArray:
    if config.residual == "gaussian":
        return gray - gaussian_filter(gray, config.gaussian_sigma, mode="reflect")
    centered = gray - gray.mean(axis=0, keepdims=True)
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    rank = config.svd_rank
    return centered - (u[:, :rank] * singular[:rank]) @ vt[:rank]


def _spectrum_features(residual: FloatArray) -> dict[str, float]:
    spectrum = np.log1p(np.abs(np.fft.fft2(residual)))
    spectrum[0, 0] = 0
    centered = spectrum - spectrum.mean()
    energy = float(np.sum(centered**2))
    results = {}
    for name, divisor in (("half", 2), ("quarter", 4)):
        correlations = [
            float(
                np.sum(centered * np.roll(centered, len(centered) // divisor, axis=axis)) / energy
            )
            if energy > 1e-20
            else 0.0
            for axis in (0, 1)
        ]
        results[f"spectrum_similarity_{name}"] = float(np.mean(correlations))
    return results


def extract(
    image: ImageInput, config: FeatureConfig | None = None, *, source: str | None = None
) -> FeatureResult:
    start = perf_counter()
    config = config or FeatureConfig()
    backend = resolve_backend(config.backend)
    gray, image_id, shape, notes = load_image(image, config)
    residual = noise_residual(gray, config)
    amplitude = float(np.max(np.abs(residual)))
    empty = amplitude <= 1e-12
    mass = np.zeros_like(residual) if empty else np.abs(residual) / amplitude
    surface = np.full_like(residual, 0.5) if empty else residual / (2 * amplitude) + 0.5
    grad_y, grad_x = np.gradient(gray)
    gradient = np.hypot(grad_y, grad_x)
    threshold = max(1e-12, float(np.quantile(gradient, config.edge_quantile)))
    edges = np.asarray(gradient > threshold, dtype=np.float64)
    dimensions = {
        "gray": differential_dimension(gray, config.scales, backend),
        "residual": differential_dimension(surface, config.scales, backend),
        "edge": box_dimension(edges, config.scales, backend),
    }
    features: dict[str, float] = {}
    for name, estimate in dimensions.items():
        features[f"{name}_dimension"] = estimate.value if estimate.value is not None else 0.0
        features[f"{name}_dimension_r2"] = (
            estimate.r_squared if estimate.r_squared is not None else 0.0
        )
        if estimate.value is None:
            notes.append(f"{name} dimension undefined: {estimate.reason}")
        elif estimate.r_squared is not None and estimate.r_squared < 0.95:
            notes.append(f"{name} dimension has weak scale fit (R2={estimate.r_squared:.3f})")
    features["edge_dimension_valid"] = float(dimensions["edge"].value is not None)
    mass_values, diagnostics = mass_features(mass, config.scales, backend)
    features.update(mass_values)
    features["residual_std"] = float(residual.std())
    histogram = np.histogram(surface, bins=64, range=(0, 1))[0]
    probabilities = histogram[histogram > 0] / histogram.sum()
    features["residual_entropy"] = float(-np.sum(probabilities * np.log2(probabilities)))
    features.update(_spectrum_features(residual))
    diagnostics.update(
        {
            "edge_threshold": threshold,
            "edge_density": float(edges.mean()),
            "residual_amplitude": amplitude,
            "residual_empty": empty,
            "config": config.model_dump(mode="json"),
        }
    )
    if empty:
        notes.append(
            "residual is numerically empty; residual surface dimension is the flat-plane convention"
        )
    if config.compression:
        values, matches = compression_features(gray, config, backend)
        features.update(values)
        diagnostics["compression"] = matches
    features = {name: features[name] for name in feature_names(config)}
    result = FeatureResult(
        image_id=image_id,
        source=source if source is not None else source_name(image),
        original_shape=shape,
        processed_shape=(config.image_size, config.image_size),
        backend=backend,
        features=features,
        dimensions=dimensions,
        diagnostics=diagnostics,
        warnings=tuple(notes),
        elapsed_seconds=perf_counter() - start,
    )
    logger.info(
        "image measured", extra={"event": "extraction", "data": result.model_dump(mode="json")}
    )
    return result
