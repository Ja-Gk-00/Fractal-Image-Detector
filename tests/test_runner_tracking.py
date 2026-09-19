import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from fractal_detector import (
    FeatureConfig,
    TaskConfig,
    TrackingConfig,
    TrackingRun,
    run_config,
    run_task,
    train,
)
from fractal_detector.cli import main
from fractal_detector.logging import configure_logging


def test_full_configuration_workflow(dataset, config: FeatureConfig, tmp_path: Path) -> None:
    samples, manifest = dataset
    task = TaskConfig(task="train", manifest=manifest.name, output_dir="training", features=config)
    path = tmp_path / "train.json"
    path.write_text(task.model_dump_json())
    summary = run_config(path)
    assert "metrics" in summary
    output = tmp_path / "training"
    assert json.loads((output / "model.json").read_text())["schema_version"] == 1
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "extraction" and "dimensions" in e["data"] for e in events)
    summary = run_task(
        TaskConfig(
            task="predict",
            manifest=manifest.name,
            model="training/model.json",
            output_dir="predictions",
        ),
        base_dir=tmp_path,
    )
    assert summary["samples"] == len(samples)
    assert len((tmp_path / "predictions/predictions.jsonl").read_text().splitlines()) == len(
        samples
    )
    split = json.loads((output / "split.json").read_text())
    holdout = tmp_path / "holdout.csv"
    holdout.write_text(
        "path,label,group\n"
        + "\n".join(
            f"{samples[i].path},{samples[i].label},{samples[i].group}"
            for i in split["test_indices"]
        )
    )
    summary = run_task(
        TaskConfig(
            task="evaluate",
            manifest=holdout.name,
            model="training/model.json",
            output_dir="evaluation",
        ),
        base_dir=tmp_path,
    )
    assert "metrics" in summary
    run_task(
        TaskConfig(task="extract", manifest=manifest.name, output_dir="features", features=config),
        base_dir=tmp_path,
    )
    assert len((tmp_path / "features/features.jsonl").read_text().splitlines()) == len(samples)


def test_failures_are_logged_and_handler_removed(
    dataset, config: FeatureConfig, tmp_path: Path
) -> None:
    _, manifest = dataset
    logger = logging.getLogger("fractal_detector")
    handlers = list(logger.handlers)
    with pytest.raises(FileNotFoundError):
        run_task(
            TaskConfig(
                task="predict",
                manifest=str(manifest),
                model="missing.json",
                output_dir=str(tmp_path / "failed"),
            )
        )
    assert logger.handlers == handlers
    assert "task_failed" in (tmp_path / "failed/events.jsonl").read_text()
    run_task(
        TaskConfig(
            task="train",
            manifest=str(manifest),
            features=config,
            output_dir=str(tmp_path / "trained"),
        )
    )
    with pytest.raises(ValueError, match="preprocessing"):
        run_task(
            TaskConfig(
                task="predict",
                manifest=str(manifest),
                model=str(tmp_path / "trained/model.json"),
                features=FeatureConfig(),
                output_dir=str(tmp_path / "wrong"),
            )
        )
    with pytest.raises(ValueError, match="overwrite an input"):
        run_task(
            TaskConfig(
                task="predict",
                manifest=str(manifest),
                model=str(tmp_path / "trained/model.json"),
                output_dir=str(tmp_path / "trained"),
            )
        )


def test_cli(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, dataset, config: FeatureConfig
) -> None:
    assert main(["doctor"]) == 0
    assert "native_available" in capsys.readouterr().out
    assert main(["run", str(tmp_path / "missing.toml")]) == 2
    assert "fractal-detector:" in capsys.readouterr().err
    _, manifest = dataset
    path = tmp_path / "extract.json"
    path.write_text(
        TaskConfig(
            task="extract",
            manifest=str(manifest),
            features=config,
            output_dir=str(tmp_path / "cli"),
        ).model_dump_json()
    )
    assert main(["run", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["task"] == "extract"


def test_explicit_json_logging(tmp_path: Path) -> None:
    handler = configure_logging(tmp_path / "log.jsonl")
    logger = logging.getLogger("fractal_detector")
    try:
        logger.info("measurement", extra={"event": "test", "data": {"dimension": 2.4}})
    finally:
        logger.removeHandler(handler)
        handler.close()
    assert json.loads((tmp_path / "log.jsonl").read_text())["data"]["dimension"] == 2.4


def test_disabled_and_missing_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    with TrackingRun(TrackingConfig(enabled=False)) as run:
        assert run.run_id is None
        run.log_dict({}, "test.json")
        run.log_metrics({"value": 1.0})
        run.log_artifact("missing.json")

    def missing(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="requires"), TrackingRun():
        pass


@pytest.mark.integration
@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="MLflow extra not installed")
def test_real_mlflow_tracking_lifecycle(
    dataset, config: FeatureConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    client = MlflowClient(tracking_uri=uri)
    tracking_config = TrackingConfig(
        enabled=True, uri=uri, experiment="test-fractal", run_name="training"
    )
    samples, _ = dataset
    result = train(samples, config)
    previous_uri = mlflow.get_tracking_uri()
    previous_print_setting = os.environ.get("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT")
    with TrackingRun(tracking_config) as run:
        run.log_training(result)
        run_id = run.run_id
    assert run_id is not None
    stored = client.get_run(run_id)
    assert stored.info.status == "FINISHED"
    assert "gray_dimension" in stored.data.metrics
    assert "holdout.roc_auc" in stored.data.metrics
    assert stored.data.params["features.image_size"] == "32"
    assert stored.data.params["training.seed"] == "42"
    assert mlflow.get_tracking_uri() == previous_uri
    assert os.environ.get("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT") == previous_print_setting
    artifact = client.download_artifacts(run_id, "model.json", str(tmp_path))
    assert json.loads(Path(artifact).read_text())["schema_version"] == 1
    failed_id = None
    with pytest.raises(ValueError, match="intentional"), TrackingRun(tracking_config) as failed:
        failed_id = failed.run_id
        raise ValueError("intentional failure")
    assert failed_id is not None
    assert client.get_run(failed_id).info.status == "FAILED"
    assert os.environ.get("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT") == previous_print_setting
    owned = client.create_run(stored.info.experiment_id)
    with TrackingRun(TrackingConfig(enabled=True, uri=uri, run_id=owned.info.run_id)) as attached:
        attached.log_evaluation(result.evaluation)
    assert client.get_run(owned.info.run_id).info.status == "RUNNING"
    client.set_terminated(owned.info.run_id)
    with (
        pytest.raises(ValueError, match="RUNNING"),
        TrackingRun(TrackingConfig(enabled=True, uri=uri, run_id=owned.info.run_id)),
    ):
        pass


@pytest.mark.integration
@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="MLflow extra not installed")
def test_configured_mlflow_extraction(
    dataset, config: FeatureConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _, manifest = dataset
    summary = run_task(
        TaskConfig(
            task="extract",
            manifest=str(manifest),
            features=config,
            output_dir=str(tmp_path / "tracked"),
            tracking=TrackingConfig(
                enabled=True, uri="sqlite:///" + (tmp_path / "task.db").as_posix()
            ),
        )
    )
    assert summary["run_id"]


def test_tracking_rejects_logging_without_context() -> None:
    with pytest.raises(RuntimeError, match="with block"):
        TrackingRun().log_metrics({"dimension": 2.1})


@pytest.mark.parametrize("previous", [None, "false", "true"])
@pytest.mark.parametrize("fail", [False, True])
def test_tracking_finalization_restores_console_setting(
    previous: str | None, fail: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"
    if previous is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, previous)
    statuses = []

    class Client:
        def __init__(self, *, tracking_uri: str | None) -> None:
            pass

        def get_experiment_by_name(self, name: str) -> SimpleNamespace:
            return SimpleNamespace(experiment_id="experiment")

        def create_run(self, experiment_id: str, *, tags: dict[str, str]) -> SimpleNamespace:
            return SimpleNamespace(info=SimpleNamespace(run_id="run"))

        def set_terminated(self, run_id: str, *, status: str) -> None:
            assert os.environ[name] == "true"
            statuses.append(status)
            if fail:
                raise OSError("tracking unavailable")

    monkeypatch.setattr(importlib, "import_module", lambda _: SimpleNamespace(MlflowClient=Client))
    tracking = TrackingRun()
    if fail:
        with pytest.raises(OSError, match="tracking unavailable"), tracking:
            pass
    else:
        with tracking:
            pass
    assert statuses == ["FINISHED"]
    assert os.environ.get(name) == previous
    with pytest.raises(RuntimeError, match="with block"):
        tracking.log_metrics({"dimension": 2.1})
