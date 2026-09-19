import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

from fractal_detector import FeatureConfig, TaskConfig, TrackingConfig, run_task
from fractal_detector.project import run_project


def test_project_requires_mlflow_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    with pytest.raises(RuntimeError, match="mlflow run"):
        run_project()


@pytest.mark.integration
@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="MLflow extra not installed")
def test_batched_mlflow_preserves_every_result(
    tmp_path: Path, config: FeatureConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    from mlflow.tracking import MlflowClient

    images = tmp_path / "images"
    images.mkdir()
    for i in range(5):
        Image.new("RGB", (32, 32), (i * 40, 0, 0)).save(images / f"image{i}.png")
    (images / "broken.png").touch()
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    summary = run_task(
        TaskConfig(
            task="extract",
            input_dir=str(images),
            output_dir=str(tmp_path / "results"),
            features=config,
            batch_size=2,
            on_error="skip",
            tracking=TrackingConfig(enabled=True, uri=uri, image_logging="batch"),
        )
    )
    client = MlflowClient(tracking_uri=uri)
    run_id = summary["run_id"]
    assert isinstance(run_id, str)
    artifacts = client.list_artifacts(run_id, "batches")
    assert len(artifacts) == 3
    rows = []
    for artifact in artifacts:
        downloaded = client.download_artifacts(run_id, artifact.path)
        rows.extend(json.loads(line) for line in Path(downloaded).read_text().splitlines())
    assert sorted(row["index"] for row in rows) == list(range(6))
    assert sum(row["kind"] == "error" for row in rows) == 1
    assert all("dimensions" in row["result"] for row in rows if row["kind"] == "extract")
    stored = client.get_run(run_id)
    assert stored.data.metrics["progress.failed"] == 1
    assert "batch.gray_dimension.mean" in stored.data.metrics


@pytest.mark.integration
@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="MLflow extra not installed")
def test_full_tracking_steps_do_not_restart_between_batches(
    tmp_path: Path, config: FeatureConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    from mlflow.tracking import MlflowClient

    images = tmp_path / "images"
    images.mkdir()
    for i in range(5):
        Image.new("RGB", (32, 32), (i * 40, 0, 0)).save(images / f"image{i}.png")
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    summary = run_task(
        TaskConfig(
            task="extract",
            input_dir=str(images),
            output_dir=str(tmp_path / "results"),
            features=config,
            batch_size=2,
            tracking=TrackingConfig(enabled=True, uri=uri),
        )
    )
    client = MlflowClient(tracking_uri=uri)
    run_id = summary["run_id"]
    assert isinstance(run_id, str)
    assert {m.step for m in client.get_metric_history(run_id, "gray_dimension")} == set(range(5))
    assert len(client.list_artifacts(run_id, "measurements")) == 5


@pytest.mark.integration
@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="MLflow extra not installed")
def test_project_attaches_to_launcher_run(
    tmp_path: Path, config: FeatureConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    from mlflow.tracking import MlflowClient

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32)).save(images / "image.png")
    task_path = tmp_path / "config with spaces.json"
    task_path.write_text(
        TaskConfig(
            task="extract", input_dir="images", output_dir="result", features=config
        ).model_dump_json()
    )
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment("project")
    owned = client.create_run(experiment_id)
    monkeypatch.setenv("MLFLOW_RUN_ID", owned.info.run_id)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    with pytest.raises(ValueError, match="config parameter"):
        run_project()
    client.log_param(owned.info.run_id, "config", str(task_path))
    summary = run_project()
    assert summary["run_id"] == owned.info.run_id
    assert client.get_run(owned.info.run_id).info.status == "RUNNING"
    assert len(client.search_runs([experiment_id])) == 1
    client.set_terminated(owned.info.run_id)


@pytest.mark.integration
@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="MLflow extra not installed")
def test_actual_mlflow_project_launch(
    tmp_path: Path, config: FeatureConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    project_root = Path(__file__).resolve().parents[1]
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32)).save(images / "image.png")
    task_path = tmp_path / "task with spaces.json"
    output = tmp_path / "results"
    task_path.write_text(
        TaskConfig(
            task="extract", input_dir=str(images), output_dir=str(output), features=config
        ).model_dump_json()
    )
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    previous = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(uri)
        client = MlflowClient(tracking_uri=uri)
        client.create_experiment("launched", artifact_location=(tmp_path / "artifacts").as_uri())
        submitted = mlflow.projects.run(
            str(project_root),
            env_manager="local",
            experiment_name="launched",
            parameters={"config": str(task_path)},
            synchronous=True,
        )
        assert client.get_run(submitted.run_id).info.status == "FINISHED"
        summary = json.loads((output / "summary.json").read_text())
        assert summary["run_id"] == submitted.run_id
        assert summary["succeeded"] == 1
    finally:
        mlflow.set_tracking_uri(previous)
