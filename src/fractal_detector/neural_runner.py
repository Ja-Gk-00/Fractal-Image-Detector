import json
from collections.abc import Iterable
from pathlib import Path

from fractal_detector.config import TaskConfig
from fractal_detector.dataset import ImageFailure, Sample
from fractal_detector.io import atomic_json
from fractal_detector.neural import NeuralDetector, evaluate_neural, train_neural
from fractal_detector.sizing import resolve_features
from fractal_detector.tracking import TrackingRun


def run_neural_task(
    config: TaskConfig,
    base: Path,
    output: Path,
    prepared: dict[str, Iterable[Sample]],
    tracking: TrackingRun,
) -> dict[str, object]:
    if config.neural is None:
        raise ValueError("neural configuration is required")
    summary: dict[str, object] = {
        "task": config.task,
        "output_dir": str(output),
        "run_id": tracking.run_id,
    }
    tracking.log_parameters(
        {
            "neural": config.neural.model_dump(mode="json"),
            "training": config.training.model_dump(mode="json"),
        }
    )
    if config.task == "train":
        splits = {name: list(rows) for name, rows in prepared.items()}
        result = train_neural(
            splits["train"],
            splits["validation"],
            splits.get("test"),
            config.features,
            config.training,
            config.neural,
            workers=config.workers,
            epoch_callback=lambda metrics, step: tracking.log_metrics(
                {f"neural.{k}": v for k, v in metrics.items()}, step=step
            ),
        )
        tracking.log_parameters(
            {
                "resolved_neural": result.detector.artifact.neural_config.model_dump(mode="json"),
                "resolved_features": result.detector.artifact.feature_config.model_dump(
                    mode="json"
                ),
            }
        )
        result.detector.save(output / "model.json")
        tracking.log_artifact(output / "model.json")
        atomic_json(output / "history.json", result.detector.artifact.history)
        tracking.log_artifact(output / "history.json")
        reports = {
            name: report.model_dump(mode="json") for name, report in result.evaluations.items()
        }
        for name, report in result.evaluations.items():
            tracking.log_metrics({f"{name}.{k}": v for k, v in report.metrics.items()})
        for name, rows in result.measurements.items():
            path = output / f"{name}-features.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(row.model_dump_json() + "\n")
            tracking.log_artifact(path)
        atomic_json(output / "report.json", reports)
        tracking.log_artifact(output / "report.json")
        summary.update(
            samples=sum(map(len, splits.values())),
            split_counts={k: len(v) for k, v in splits.items()},
            best_epoch=result.detector.artifact.best_epoch,
            metrics={k: v.metrics for k, v in result.evaluations.items()},
        )
    else:
        if config.model is None:
            raise ValueError("model is required")
        detector = NeuralDetector.load(base / config.model, device=config.neural.device)
        if (
            "features" in config.model_fields_set
            and resolve_features(config.features, detector.artifact.feature_config.image_size)
            != detector.artifact.feature_config
        ):
            raise ValueError(
                "prediction/evaluation preprocessing must match the saved model; omit features"
            )
        if config.task == "evaluate":
            report = evaluate_neural(
                detector,
                list(prepared["input"]),
                workers=config.workers,
                allow_validation=config.split == "validation",
            )
            atomic_json(output / "report.json", report.model_dump(mode="json"))
            tracking.log_artifact(output / "report.json")
            tracking.log_metrics(report.metrics)
            summary.update(samples=report.samples, metrics=report.metrics)
        else:
            succeeded = failed = 0
            with (
                (output / "predictions.jsonl").open("w", encoding="utf-8") as results,
                (output / "errors.jsonl").open("w", encoding="utf-8") as errors,
            ):
                for item in detector.iter_predict(
                    prepared["input"],
                    workers=config.workers,
                    batch_size=config.batch_size,
                    on_error=config.on_error,
                ):
                    if isinstance(item, ImageFailure):
                        errors.write(item.model_dump_json() + "\n")
                        failed += 1
                    else:
                        results.write(item.model_dump_json() + "\n")
                        if config.tracking.image_logging == "full":
                            tracking.log_predictions([item], start_step=succeeded)
                        succeeded += 1
            for name in ("predictions.jsonl", "errors.jsonl"):
                tracking.log_artifact(output / name)
            summary.update(samples=succeeded + failed, succeeded=succeeded, failed=failed)
    atomic_json(output / "summary.json", summary)
    tracking.log_dict(
        json.loads((output / "summary.json").read_text(encoding="utf-8")), "summary.json"
    )
    return summary
