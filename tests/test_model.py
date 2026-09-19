import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.special import expit

from fractal_detector import (
    Detector,
    FeatureConfig,
    Sample,
    TrainingConfig,
    evaluate,
    evaluate_features,
    extract,
    extract_dataset,
    fit_features,
    train,
)


def test_training_group_split_and_explanation(
    dataset, config: FeatureConfig, tmp_path: Path
) -> None:
    samples, _ = dataset
    result = train(samples, config, workers=2)
    assert {samples[i].group for i in result.train_indices}.isdisjoint(
        {samples[i].group for i in result.test_indices}
    )
    assert len(result.train_indices) + len(result.test_indices) == len(samples)
    assert result.evaluation.metrics["roc_auc"] >= 0.9
    prediction = result.evaluation.predictions[0]
    expected = expit(prediction.intercept + sum(c.log_odds for c in prediction.contributions))
    assert prediction.synthetic_probability == pytest.approx(expected, abs=1e-14)
    assert "fractal dimension" in prediction.explanation
    path = tmp_path / "model.json"
    result.detector.save(path)
    loaded = Detector.load(path)
    before = result.detector.predict(samples[0].path)
    after = loaded.predict(samples[0].path)
    assert before.synthetic_probability == after.synthetic_probability
    assert before.contributions == after.contributions
    assert loaded.predict_many(samples[:2])[0].synthetic_probability == after.synthetic_probability


def test_seed_reproducibility_and_scaler_has_no_holdout_leakage(
    dataset, config: FeatureConfig
) -> None:
    samples, _ = dataset
    samples = [s.model_copy(update={"group": None}) for s in samples]
    a, b = train(samples, config), train(samples, config)
    assert a.train_indices == b.train_indices
    assert a.detector.artifact.coefficients == b.detector.artifact.coefficients
    train_rows = np.array([list(a.measurements[i].features.values()) for i in a.train_indices])
    np.testing.assert_allclose(a.detector.artifact.means, train_rows.mean(axis=0), atol=1e-14)


def test_evaluation_rejects_training_images_and_groups(dataset, config: FeatureConfig) -> None:
    samples, _ = dataset
    result = train(samples, config)
    with pytest.raises(ValueError, match="used for fitting"):
        evaluate(result.detector, samples)
    test_samples = [samples[i] for i in result.test_indices]
    changed = [
        s.model_copy(update={"group": result.detector.artifact.training_groups[0]})
        for s in test_samples
    ]
    with pytest.raises(ValueError, match="group used"):
        evaluate(result.detector, changed)
    evaluation = evaluate(result.detector, test_samples)
    assert evaluation.metrics == result.evaluation.metrics


def test_duplicates_detected_even_with_different_paths(
    dataset, config: FeatureConfig, tmp_path: Path
) -> None:
    samples, _ = dataset
    duplicate = tmp_path / "copied.png"
    duplicate.write_bytes(Path(samples[0].path).read_bytes())
    with pytest.raises(ValueError, match="duplicate decoded"):
        train([*samples, Sample(path=str(duplicate), label=0, group="extra")], config)


def test_bad_training_data(dataset, config: FeatureConfig) -> None:
    samples, _ = dataset
    with pytest.raises(ValueError, match="both real"):
        train(samples[:4], config)
    with pytest.raises(ValueError, match="at least two"):
        train([samples[0], samples[-1]], config)
    with pytest.raises(ValueError, match="every sample"):
        train([samples[0].model_copy(update={"group": None}), *samples[1:]], config)
    with pytest.raises(ValueError):
        train([s.model_copy(update={"group": "one"}) for s in samples], config)
    with pytest.raises(ValueError, match="group-disjoint"):
        train([s.model_copy(update={"group": str(s.label)}) for s in samples], config)


def test_feature_schema_and_config_mismatches(dataset, config: FeatureConfig) -> None:
    samples, _ = dataset
    results = extract_dataset(samples, config)
    with pytest.raises(ValueError, match="lengths"):
        fit_features(samples, results[:2], config)
    fitted = fit_features(samples, results, config)
    wrong = results[0].model_copy(update={"features": {"nonsense": 1.0}})
    with pytest.raises(ValueError, match="schema"):
        fitted.predict_features(wrong)
    wrong = results[0].model_copy(update={"diagnostics": {}})
    with pytest.raises(ValueError, match="configuration"):
        fitted.predict_features(wrong)
    wrong = extract(samples[0].path, config.model_copy(update={"gaussian_sigma": 2.0}))
    with pytest.raises(ValueError, match="configuration differs"):
        fitted.predict_features(wrong)
    with pytest.raises(ValueError, match="lengths"):
        evaluate_features(fitted, samples, results[:1])


def test_model_tampering_is_rejected(dataset, config: FeatureConfig, tmp_path: Path) -> None:
    samples, _ = dataset
    trained = train(samples, config)
    for field, value in (
        ("schema_version", 999),
        ("means", []),
        ("feature_names", ["bad"]),
        ("intercept", float("nan")),
        ("scales", [-1] * len(trained.detector.artifact.scales)),
    ):
        data = trained.detector.artifact.model_dump(mode="json")
        data[field] = value
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValidationError):
            Detector.load(path)


def test_threshold_is_preserved(dataset, config: FeatureConfig) -> None:
    samples, _ = dataset
    trained = train(samples, config, TrainingConfig(threshold=0.8))
    assert trained.detector.artifact.threshold == 0.8
    for p in trained.evaluation.predictions:
        assert (p.label == "synthetic") == (p.synthetic_probability >= 0.8)
