import logging
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from fractal_detector.compression import compression_features
from fractal_detector.config import FeatureConfig
from fractal_detector.fractal import box_dimension, differential_dimension, mass_features
from fractal_detector.io import FloatArray, ImageInput, load_image, source_name
from fractal_detector.kernels import resolve_backend
from fractal_detector.multifractal import multifractal_spectrum, sliding_lacunarity
from fractal_detector.records import FeatureResult
from fractal_detector.sizing import image_side, resolve_features

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
    if config.multifractal:
        names += [
            "multifractal_width",
            "multifractal_alpha_peak",
            "multifractal_f_peak",
            "multifractal_d_span",
            "multifractal_fit_min",
            "multifractal_valid",
        ]
        names += [f"multifractal_d_{i}" for i in range(len(config.moment_orders))]
    if config.low_correlation:
        for rank in config.pca_ranks:
            names += [
                f"pca_{rank}_{name}"
                for name in (
                    "energy_fraction",
                    "mean",
                    "std",
                    "entropy",
                    "dimension",
                    "dimension_r2",
                    "lacunarity_mean",
                    "multifractal_width",
                    "multifractal_d_span",
                    "valid",
                )
            ]
    if config.families is not None:
        names = [
            name
            for name in names
            if feature_family(name) in config.families
            or (
                "dimensions" in config.families
                and feature_family(name) in {"gray_surface", "residual_surface", "edge_box"}
            )
        ]
    return tuple(names)


def feature_family(name: str) -> str:
    if name.startswith("pca_"):
        return "low_correlation"
    if name.startswith("multifractal_"):
        return "multifractal"
    if "dimension" in name:
        return {"gray": "gray_surface", "residual": "residual_surface", "edge": "edge_box"}[
            name.split("_")[0]
        ]
    if name in {"residual_d0", "residual_d1", "residual_d2"}:
        return "renyi"
    if name.startswith("lacunarity_"):
        return "lacunarity"
    if name.startswith("spectrum_"):
        return "spectral"
    if name.startswith("compression_"):
        return "compression"
    return "statistics"


def noise_residual(gray: FloatArray, config: FeatureConfig) -> FloatArray:
    if config.residual == "gaussian":
        return gray - gaussian_filter(gray, config.gaussian_sigma, mode="reflect")
    if config.residual == "median":
        return gray - median_filter(gray, size=config.median_size, mode="reflect")
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
    config = resolve_features(
        config, image_side(image, config.max_input_pixels) if config.image_size == "auto" else 0
    )
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
    histogram = np.histogram(surface, bins=config.entropy_bins, range=(0, 1))[0]
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
    if config.lacunarity_mode == "sliding":
        curve = sliding_lacunarity(mass, config.scales, config.lacunarity_stride)
        features.update(lacunarity_mean=float(np.mean(curve)), lacunarity_std=float(np.std(curve)))
        diagnostics["lacunarity"] = curve
    if config.multifractal:
        values, spectrum = multifractal_spectrum(
            mass, config.scales, config.moment_orders, backend=backend, mass_floor=config.mass_floor
        )
        features.update(values)
        diagnostics["multifractal"] = spectrum
    if config.low_correlation:
        values, pca_diagnostics = _low_correlation(gray, config, backend)
        features.update(values)
        diagnostics["low_correlation"] = pca_diagnostics
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


def _low_correlation(
    gray: FloatArray, config: FeatureConfig, backend: str
) -> tuple[dict[str, float], dict[str, object]]:
    centered = gray - gray.mean(axis=0, keepdims=True)
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    energy = float(np.sum(singular**2))
    features: dict[str, float] = {}
    diagnostics: dict[str, object] = {}
    for rank in config.pca_ranks:
        residual = centered - (u[:, :rank] * singular[:rank]) @ vt[:rank]
        amplitude = float(np.max(np.abs(residual)))
        valid = amplitude > 1e-12
        mass = np.abs(residual) / amplitude if valid else np.zeros_like(residual)
        surface = residual / (2 * amplitude) + 0.5 if valid else np.full_like(residual, 0.5)
        estimate = differential_dimension(surface, config.scales, backend)
        spectrum, details = multifractal_spectrum(
            mass, config.scales, config.moment_orders, backend=backend, mass_floor=config.mass_floor
        )
        lac = (
            sliding_lacunarity(mass, config.scales, config.lacunarity_stride)
            if config.lacunarity_mode == "sliding"
            else mass_features(mass, config.scales, backend)[1]["lacunarity"]
        )
        counts = np.histogram(surface, bins=config.entropy_bins, range=(0, 1))[0]
        p = counts[counts > 0] / counts.sum()
        values = {
            "energy_fraction": float(np.sum(singular[rank:] ** 2) / energy)
            if energy > 1e-24
            else 0.0,
            "mean": float(residual.mean()),
            "std": float(residual.std()),
            "entropy": float(-np.sum(p * np.log2(p))),
            "dimension": estimate.value or 0.0,
            "dimension_r2": estimate.r_squared or 0.0,
            "lacunarity_mean": float(np.mean(np.asarray(lac, dtype=np.float64))),
            "multifractal_width": spectrum["multifractal_width"],
            "multifractal_d_span": spectrum["multifractal_d_span"],
            "valid": float(valid),
        }
        features.update({f"pca_{rank}_{key}": value for key, value in values.items()})
        diagnostics[str(rank)] = {
            "dimension": estimate.model_dump(mode="json"),
            "lacunarity": lac,
            "spectrum": details,
            "residual_amplitude": amplitude,
        }
    return features, diagnostics
