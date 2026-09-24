import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from fractal_detector import FeatureConfig, extract, extract_dataset, fit_features
from fractal_detector.config import SearchCandidate, SearchConfig, TaskConfig
from fractal_detector.features import feature_names, noise_residual
from fractal_detector.model import Detector, evaluate_features
from fractal_detector.multifractal import multifractal_spectrum, sliding_lacunarity
from fractal_detector.parallel import resolve_workers
from fractal_detector.runner import run_task
from fractal_detector.search import search_features


def test_uniform_and_point_mass_spectra() -> None:
    q = (-4.0, -1.0, 0.0, 1.0, 2.0, 4.0)
    for expected, mass in [
        (2, np.ones((32, 32))),
        (0, np.pad(np.ones((1, 1)), ((0, 31), (0, 31)))),
    ]:
        features, diagnostics = multifractal_spectrum(mass, (2, 4, 8), q)
        np.testing.assert_allclose(
            np.asarray(diagnostics["generalized_dimensions"], dtype=float), expected, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(diagnostics["alpha"], dtype=float), expected, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(diagnostics["spectrum"], dtype=float), expected, atol=1e-12
        )
        assert features["multifractal_width"] == pytest.approx(0, abs=1e-12)
    features, diagnostics = multifractal_spectrum(np.zeros((32, 32)), (2, 4, 8), q)
    assert features["multifractal_valid"] == 0
    assert diagnostics["empty"]


def test_multiplicative_cascade_matches_analytic_tau() -> None:
    probabilities = np.array([[0.1, 0.2], [0.3, 0.4]])
    cascade = np.ones((1, 1))
    for _ in range(6):
        cascade = np.kron(cascade, probabilities)
    q = (-4.0, -2.0, 0.0, 1.0, 2.0, 4.0)
    features, details = multifractal_spectrum(cascade, (2, 4, 8, 16), q)
    expected = -np.log2(np.sum(probabilities.reshape(-1)[None, :] ** np.array(q)[:, None], axis=1))
    np.testing.assert_allclose(np.asarray(details["tau"], dtype=float), expected, atol=1e-12)
    assert features["multifractal_width"] > 0.5
    assert features["multifractal_fit_min"] > 0.99999
    floor, _ = multifractal_spectrum(cascade, (2, 4, 8, 16), q, mass_floor=1e-3)
    assert all(np.isfinite(list(floor.values())))


@pytest.mark.parametrize(
    "mass,scales,orders,floor",
    [
        (np.ones((3, 4)), (1, 2, 3), (-1.0, 0.0, 1.0), 0.0),
        (np.ones((32, 32)), (2, 3, 4), (-1.0, 0.0, 1.0), 0.0),
        (np.ones((32, 32)), (2, 4, 8), (0.0, 0.0, 1.0), 0.0),
        (np.ones((32, 32)), (2, 4, 8), (-1.0, 0.0, 1.0), -1.0),
        (np.full((32, 32), np.nan), (2, 4, 8), (-1.0, 0.0, 1.0), 0.0),
    ],
)
def test_invalid_spectrum(mass, scales, orders, floor) -> None:
    with pytest.raises(ValueError):
        multifractal_spectrum(mass, scales, orders, mass_floor=floor)


def test_sliding_lacunarity_against_brute_force() -> None:
    mass = np.random.default_rng(1337).random((32, 32))
    measured = sliding_lacunarity(mass, (2, 4, 8), 3)
    for size, result in zip((2, 4, 8), measured, strict=True):
        boxes = np.array(
            [
                mass[y : y + size, x : x + size].sum()
                for y in range(0, 33 - size, 3)
                for x in range(0, 33 - size, 3)
            ]
        )
        assert result == pytest.approx(1 + boxes.var() / boxes.mean() ** 2, abs=1e-12)
    assert sliding_lacunarity(np.zeros((32, 32)), (2, 4, 8)) == [0.0, 0.0, 0.0]
    for scales, stride in [((2,), 0), ((33,), 1)]:
        with pytest.raises(ValueError):
            sliding_lacunarity(mass, scales, stride)


def test_mass_estimators_are_intensity_scale_invariant() -> None:
    mass = np.random.default_rng(1337).random((32, 32))
    expected, _ = multifractal_spectrum(mass, (2, 4, 8), (-4.0, 0.0, 1.0, 4.0))
    expected_lac = sliding_lacunarity(mass, (2, 4, 8))
    for factor in (1e-200, 255.0, 1e200):
        actual, _ = multifractal_spectrum(mass * factor, (2, 4, 8), (-4.0, 0.0, 1.0, 4.0))
        np.testing.assert_allclose(list(actual.values()), list(expected.values()), atol=1e-12)
        np.testing.assert_allclose(
            sliding_lacunarity(mass * factor, (2, 4, 8)), expected_lac, atol=1e-12
        )


def test_extended_features_and_pca_energy() -> None:
    config = FeatureConfig(
        image_size=64,
        scales=(2, 4, 8, 16),
        compression=False,
        residual="median",
        low_correlation=True,
        pca_ranks=(4, 8, 16),
        multifractal=True,
        lacunarity_mode="sliding",
    )
    image = np.random.default_rng(1337).random((64, 64))
    measured = extract(image, config)
    assert tuple(measured.features) == feature_names(config)
    assert (
        measured.features["pca_4_energy_fraction"]
        > measured.features["pca_8_energy_fraction"]
        > measured.features["pca_16_energy_fraction"]
    )
    json.dumps(measured.model_dump(mode="json"), allow_nan=False)
    flat = extract(np.ones((64, 64)), config)
    assert flat.features["pca_4_valid"] == 0
    assert np.all(noise_residual(np.ones((64, 64)), config) == 0)
    tiled = extract(image, config.model_copy(update={"lacunarity_mode": "tiled"}))
    assert tiled.features["pca_4_lacunarity_mean"] >= 1
    selected = config.model_copy(update={"families": ("multifractal", "low_correlation")})
    assert set(feature_names(selected)) == {
        k for k in measured.features if k.startswith(("multifractal_", "pca_"))
    }


def test_legacy_measurements_and_model_still_load(
    dataset, config: FeatureConfig, tmp_path: Path
) -> None:
    samples, _ = dataset
    rows = extract_dataset(samples, config)
    old_names = {
        "image_size",
        "scales",
        "residual",
        "gaussian_sigma",
        "svd_rank",
        "edge_quantile",
        "backend",
        "compression",
        "block_size",
        "stride",
        "max_domains",
        "s_clip",
        "transform_ids",
        "max_input_pixels",
    }
    for row in rows:
        stored = row.diagnostics["config"]
        assert isinstance(stored, dict)
        row.diagnostics["config"] = {k: v for k, v in stored.items() if k in old_names}
    model = fit_features(samples, rows, config)
    artifact = model.artifact.model_dump(mode="json")
    artifact["feature_config"] = rows[0].diagnostics["config"]
    artifact.pop("selection_ids")
    artifact.pop("selection_groups")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(artifact))
    loaded = Detector.load(path)
    assert (
        loaded.predict_features(rows[0]).synthetic_probability
        == model.predict_features(rows[0]).synthetic_probability
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"median_size": 4},
        {"moment_orders": (1.0, 0.0, 2.0)},
        {"moment_orders": (-17.0, 0.0, 1.0)},
        {"low_correlation": True, "pca_ranks": (128,)},
        {"families": ()},
        {"families": ("multifractal",)},
        {"families": ("dimensions", "dimensions")},
    ],
)
def test_research_config_validation(updates) -> None:
    with pytest.raises(ValidationError):
        FeatureConfig(image_size=128, **updates)


def test_search_holds_out_selection_ids(dataset, config: FeatureConfig, tmp_path: Path) -> None:
    samples, _ = dataset
    train = [s for s in samples if s.group in {"source0", "source1", "source2", "source3"}]
    validation = [s for s in samples if s.group == "source4"]
    heldout = [s for s in samples if s.group == "source5"]
    search = SearchConfig(
        candidates=(SearchCandidate(name="baseline", features=config),), regularization_c=(0.1, 1.0)
    )
    result = search_features(train, validation, search, workers=2)
    assert len(result.trials) == 2
    assert result.report["test_used"] is False
    with pytest.raises(ValueError):
        evaluate_features(result.detector, validation, extract_dataset(validation, config))
    assert (
        evaluate_features(result.detector, heldout, extract_dataset(heldout, config)).samples == 4
    )
    with pytest.raises(ValueError, match="duplicate"):
        search_features(train, train, search)
    with pytest.raises(ValueError, match="group"):
        search_features(train, [s.model_copy(update={"group": None}) for s in validation], search)

    def write(name, rows):
        path = tmp_path / f"{name}.json"
        import csv

        path = path.with_suffix(".csv")
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "group"])
            writer.writeheader()
            writer.writerows({"path": s.path, "label": s.label, "group": s.group} for s in rows)
        return str(path)

    task = TaskConfig.model_validate_json(
        json.dumps(
            {
                "task": "search",
                "data": {
                    "train": [write("train", train)],
                    "validation": [write("validation", validation)],
                    "test": ["never-read.csv"],
                },
                "search": search.model_dump(mode="json"),
                "output_dir": str(tmp_path / "search"),
                "workers": "auto",
            }
        )
    )
    summary = run_task(task)
    assert summary["samples"] == len(train) + len(validation)
    assert (tmp_path / "search" / "model.json").is_file()


def test_auto_workers(monkeypatch) -> None:
    monkeypatch.setattr("os.process_cpu_count", lambda: 24)
    assert resolve_workers("auto") == 24
    for value in (0, True, "bad", 2.5):
        with pytest.raises(ValueError):
            resolve_workers(value)  # ty: ignore[invalid-argument-type]


def test_all_method_presets(tmp_path: Path) -> None:
    from fractal_detector.config import load_config
    from fractal_detector.presets import METHODS, method_preset, write_method_configs

    paths = write_method_configs(tmp_path, test="test.csv")
    assert len(paths) == 2 * len(METHODS) == 36
    for path in paths:
        config = load_config(path)
        assert config.workers == "auto"
        assert config.data is not None and config.data.test == ("test.csv",)
    for method in (
        "gray_surface",
        "edge_box",
        "residual_surface",
        "compression",
        "statistics",
        "spectral",
        "renyi",
        "lacunarity",
    ):
        values = method_preset(method)["features"]
        config = FeatureConfig.model_validate_json(json.dumps(values))
        result = extract(np.random.default_rng(1337).random((32, 32)), config)
        assert tuple(result.features) == feature_names(config)
        assert result.features
    with pytest.raises(ValueError, match="unknown"):
        method_preset("unknown")


@pytest.mark.parametrize(
    "values",
    [
        {"task": "search", "input_dir": "images"},
        {"task": "train", "input_dir": "images", "search": {"candidates": []}},
        {"task": "extract", "input_dir": "images", "neural": {}},
        {"task": "train", "inputs": ["x.csv"], "neural": {}},
    ],
)
def test_task_extensions_require_valid_inputs(values) -> None:
    with pytest.raises(ValueError):
        TaskConfig.model_validate_json(json.dumps(values))


def test_search_rejects_unbounded_and_duplicate_trials(config: FeatureConfig) -> None:
    candidate = SearchCandidate(name="baseline", features=config)
    for candidates, cs in [
        ((), (1.0,)),
        ((candidate,), ()),
        ((candidate, candidate), (1.0,)),
        ((candidate,), tuple(float(i + 1) for i in range(101))),
    ]:
        with pytest.raises(ValueError):
            SearchConfig(candidates=candidates, regularization_c=cs)
