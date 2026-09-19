import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fractal_detector import (
    DataSource,
    DataSplits,
    Detector,
    FeatureConfig,
    TaskConfig,
    TrackingConfig,
    evaluate,
    extract_dataset,
    fit_features,
    run_task,
    train_splits,
)


def split_samples(samples):
    return (
        [s for s in samples if s.group in {"source0", "source1", "source2"}],
        [s for s in samples if s.group == "source3"],
        [s for s in samples if s.group in {"source4", "source5"}],
    )


def test_explicit_training_fits_only_train(dataset, config: FeatureConfig) -> None:
    samples, _ = dataset
    training, validation, test = split_samples(samples)
    result = train_splits(training, validation, test, feature_config=config)
    reference = fit_features(training, extract_dataset(training, config), config)
    assert result.detector.artifact.means == reference.artifact.means
    assert result.detector.artifact.coefficients == reference.artifact.coefficients
    assert set(result.evaluations) == {"validation", "test"}
    assert result.evaluations["test"].metrics == evaluate(reference, test).metrics
    assert len(result.detector.artifact.training_ids) == len(training)
    alone = train_splits(training, feature_config=config)
    assert alone.evaluations == {} and set(alone.measurements) == {"train"}
    assert alone.detector.artifact.coefficients == reference.artifact.coefficients


def test_explicit_splits_reject_leakage_and_missing_labels(dataset, config: FeatureConfig) -> None:
    samples, _ = dataset
    training, validation, test = split_samples(samples)
    with pytest.raises(ValueError, match="group from another split"):
        train_splits(training, [*validation, training[0]], test, feature_config=config)
    ungrouped = [
        [s.model_copy(update={"group": None}) for s in part]
        for part in (training, validation, test)
    ]
    with pytest.raises(ValueError, match="image from another split"):
        train_splits(
            ungrouped[0], ungrouped[1], ungrouped[2] + [ungrouped[1][0]], feature_config=config
        )
    with pytest.raises(ValueError, match="every sample"):
        train_splits(training, ungrouped[1], feature_config=config)
    with pytest.raises(ValueError, match="both real"):
        train_splits(training, [], feature_config=config)
    with pytest.raises(ValueError, match="both real"):
        train_splits(
            [s.model_copy(update={"label": None}) for s in training], feature_config=config
        )


def test_explicit_training_config_and_split_reports(
    dataset, config: FeatureConfig, tmp_path: Path
) -> None:
    samples, _ = dataset
    parts = dict(zip(("train", "validation", "test"), split_samples(samples), strict=True))
    sources = {}
    for name, rows in parts.items():
        path = tmp_path / f"{name}.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [{"path": Path(s.path).name, "label": s.label, "group": s.group} for s in rows]
            ),
            path,
        )
        sources[name] = (str(path),)
    data = DataSplits(**sources)
    output = tmp_path / "training"
    summary = run_task(TaskConfig(task="train", data=data, features=config, output_dir=str(output)))
    assert summary["split_counts"] == {k: len(v) for k, v in parts.items()}
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    assert set(metrics) == {"validation", "test"}
    assert json.loads((output / "split.json").read_text())["mode"] == "explicit"
    stored = Detector.load(output / "model.json")
    assert len(stored.artifact.training_ids) == len(parts["train"])
    for name in ("validation", "test"):
        report = json.loads((output / f"{name}-report.json").read_text())
        assert report["samples"] == len(parts[name])
    evaluated = run_task(
        TaskConfig(
            task="evaluate",
            data=data,
            split="test",
            model=str(output / "model.json"),
            output_dir=str(tmp_path / "evaluation"),
        )
    )
    assert evaluated["metrics"] == metrics["test"]


def test_labeled_multiple_sources_support_automatic_holdout(
    dataset, config: FeatureConfig, tmp_path: Path
) -> None:
    samples, _ = dataset
    inputs = tuple(
        DataSource(paths=tuple(s.path for s in samples if s.label == label), label=label)
        for label in (0, 1)
    )
    summary = run_task(
        TaskConfig(
            task="train", inputs=inputs, features=config, output_dir=str(tmp_path / "automatic")
        )
    )
    assert summary["samples"] == 24
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    assert np.isfinite(metrics["roc_auc"])


@pytest.mark.integration
@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="MLflow extra not installed")
def test_explicit_training_logs_separate_mlflow_metrics(
    dataset, config: FeatureConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlflow.tracking import MlflowClient

    from fractal_detector import TrackingRun

    monkeypatch.chdir(tmp_path)
    samples, _ = dataset
    result = train_splits(*split_samples(samples), feature_config=config)
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    with TrackingRun(TrackingConfig(enabled=True, uri=uri)) as tracking:
        tracking.log_training(result)
        run_id = tracking.run_id
    assert run_id is not None
    client = MlflowClient(tracking_uri=uri)
    stored = client.get_run(run_id)
    assert "validation.roc_auc" in stored.data.metrics and "test.roc_auc" in stored.data.metrics
    assert "holdout.roc_auc" not in stored.data.metrics
    assert stored.info.status == "FINISHED"
    assert len(client.list_artifacts(run_id, "measurements")) == len(samples)
