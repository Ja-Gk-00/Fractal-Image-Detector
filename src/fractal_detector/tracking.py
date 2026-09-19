import importlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import RLock
from time import time
from types import TracebackType
from typing import Self

from fractal_detector.config import TrackingConfig
from fractal_detector.model import Evaluation, SplitTrainingResult, TrainingResult
from fractal_detector.records import FeatureResult, Prediction

_termination_lock = RLock()


class TrackingRun:
    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig(enabled=True)
        self.run_id: str | None = None
        self._client = None
        self._owns_run = False

    def __enter__(self) -> Self:
        if self._client is not None:
            raise RuntimeError("this TrackingRun is already active")
        if not self.config.enabled:
            return self
        try:
            tracking = importlib.import_module("mlflow.tracking")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MLflow tracking requires: pip install 'fractal-image-detector[mlflow]'"
            ) from exc
        self._client = tracking.MlflowClient(tracking_uri=self.config.uri)
        if self.config.run_id:
            run = self._client.get_run(self.config.run_id)
            if run.info.status != "RUNNING":
                raise ValueError("an attached MLflow run must be RUNNING")
            self.run_id = self.config.run_id
        else:
            experiment = self._client.get_experiment_by_name(self.config.experiment)
            experiment_id = (
                experiment.experiment_id
                if experiment is not None
                else self._client.create_experiment(self.config.experiment)
            )
            tags = {"fractal.package_version": "0.1.0"}
            if self.config.run_name:
                tags["mlflow.runName"] = self.config.run_name
            self.run_id = self._client.create_run(experiment_id, tags=tags).info.run_id
            self._owns_run = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self._owns_run and self._client is not None and self.run_id is not None:
                with _termination_lock:
                    name = "MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"
                    previous = os.environ.get(name)
                    os.environ[name] = "true"
                    try:
                        self._client.set_terminated(
                            self.run_id, status="FAILED" if exc_type else "FINISHED"
                        )
                    finally:
                        if previous is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = previous
        finally:
            self._client = None
            self._owns_run = False

    def log_dict(self, data: Mapping[str, object], artifact_file: str) -> None:
        self._ensure_active()
        if self._client is not None and self.run_id is not None:
            self._client.log_dict(self.run_id, dict(data), artifact_file)

    def log_metrics(self, metrics: dict[str, float], *, step: int = 0) -> None:
        self._ensure_active()
        if self._client is not None and self.run_id is not None:
            from mlflow.entities import Metric

            timestamp = int(time() * 1000)
            items = [Metric(name, value, timestamp, step) for name, value in metrics.items()]
            for start in range(0, len(items), 1000):
                self._client.log_batch(self.run_id, metrics=items[start : start + 1000])

    def log_parameters(self, parameters: Mapping[str, object]) -> None:
        self._ensure_active()
        if self._client is not None and self.run_id is not None:
            from mlflow.entities import Param

            flattened: dict[str, str] = {}

            def flatten(values: Mapping[str, object], prefix: str = "") -> None:
                for key, value in values.items():
                    name = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        flatten(value, name)
                    else:
                        flattened[name] = (
                            json.dumps(value, allow_nan=False)
                            if not isinstance(value, str)
                            else value
                        )

            flatten(parameters)
            items = [Param(name, value) for name, value in flattened.items()]
            for start in range(0, len(items), 100):
                self._client.log_batch(self.run_id, params=items[start : start + 100])

    def _ensure_active(self) -> None:
        if self.config.enabled and self._client is None:
            raise RuntimeError("use TrackingRun inside a with block before logging")

    def log_artifact(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        self._ensure_active()
        if self._client is not None and self.run_id is not None:
            self._client.log_artifact(self.run_id, str(path), artifact_path=artifact_path)

    def log_features(self, results: Sequence[FeatureResult], *, start_step: int = 0) -> None:
        if results:
            config = results[0].diagnostics.get("config")
            if isinstance(config, dict):
                self.log_parameters({"features": config})
        for step, result in enumerate(results, start=start_step):
            self.log_metrics(
                result.features | {"extraction_seconds": result.elapsed_seconds}, step=step
            )
            self.log_dict(result.model_dump(mode="json"), f"measurements/{step:08d}.json")

    def log_predictions(self, predictions: Sequence[Prediction], *, start_step: int = 0) -> None:
        self.log_features([p.measurement for p in predictions], start_step=start_step)
        for step, prediction in enumerate(predictions, start=start_step):
            self.log_metrics({"synthetic_probability": prediction.synthetic_probability}, step=step)
            self.log_dict(prediction.model_dump(mode="json"), f"predictions/{step:08d}.json")

    def log_evaluation(self, evaluation: Evaluation) -> None:
        self.log_metrics({f"evaluation.{k}": v for k, v in evaluation.metrics.items()})
        self.log_dict(
            evaluation.model_dump(mode="json", exclude={"predictions"}), "evaluation.json"
        )
        self.log_predictions(evaluation.predictions)

    def log_training(self, result: TrainingResult | SplitTrainingResult) -> None:
        self.log_parameters(
            {"training": result.detector.artifact.training_config.model_dump(mode="json")}
        )
        self.log_dict(result.detector.artifact.model_dump(mode="json"), "model.json")
        if isinstance(result, SplitTrainingResult):
            self.log_parameters({"training.split_mode": "explicit"})
            self.log_dict(
                {"image_ids": {k: [r.image_id for r in v] for k, v in result.measurements.items()}},
                "split.json",
            )
            self.log_features([r for values in result.measurements.values() for r in values])
            for name, evaluation in result.evaluations.items():
                self.log_metrics({f"{name}.{k}": v for k, v in evaluation.metrics.items()})
                self.log_dict(evaluation.model_dump(mode="json"), f"{name}.json")
            return
        self.log_dict(
            {"train_indices": result.train_indices, "test_indices": result.test_indices},
            "split.json",
        )
        self.log_features(result.measurements)
        self.log_metrics({f"holdout.{k}": v for k, v in result.evaluation.metrics.items()})
        self.log_dict(result.evaluation.model_dump(mode="json"), "holdout.json")
