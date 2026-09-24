import importlib.metadata
import json
import logging
import platform
from collections.abc import Iterable
from pathlib import Path

from fractal_detector.config import DataSource, SourceSpec, TaskConfig, load_config
from fractal_detector.dataset import Sample, read_manifest
from fractal_detector.io import atomic_json
from fractal_detector.logging import configure_logging
from fractal_detector.model import Detector, SplitTrainingResult, evaluate, train, train_splits
from fractal_detector.parallel import resolve_workers
from fractal_detector.sizing import resolve_features
from fractal_detector.sources import iter_sources
from fractal_detector.streaming import stream_task
from fractal_detector.tracking import TrackingRun


def run_config(path: str | Path) -> dict[str, object]:
    path = Path(path).resolve()
    return run_task(load_config(path), base_dir=path.parent)


def _source_groups(config: TaskConfig) -> dict[str, tuple[SourceSpec, ...]]:
    if config.data is not None:
        if config.task in {"train", "search"}:
            return {
                name: getattr(config.data, name)
                for name in ("train", "validation", "test")
                if getattr(config.data, name) and (config.task != "search" or name != "test")
            }
        if config.split is None:
            raise ValueError("data requires a split selector")
        return {"input": getattr(config.data, config.split)}
    if config.inputs is not None:
        return {"input": config.inputs}
    path = config.manifest if config.manifest is not None else config.input_dir
    if path is None:
        raise ValueError("no input sources configured")
    return {"input": (path,)}


def run_task(config: TaskConfig, *, base_dir: str | Path = ".") -> dict[str, object]:
    config = config.model_copy(update={"workers": resolve_workers(config.workers)})
    base = Path(base_dir).resolve()
    output = (base / config.output_dir).resolve()
    source_groups = _source_groups(config)
    source_paths = {
        (base / path).resolve()
        for sources in source_groups.values()
        for spec in sources
        for path in (spec.paths if isinstance(spec, DataSource) else (spec,))
    }
    for path in source_paths:
        if path.is_dir() and path.is_relative_to(output):
            raise ValueError("output_dir must not equal or contain input_dir or a source directory")
    if config.input_dir is not None and not (base / config.input_dir).is_dir():
        raise ValueError("input directory does not exist")
    outputs = {
        output / name
        for name in (
            "events.jsonl",
            "config.json",
            "environment.json",
            "manifest.json",
            "features.jsonl",
            "model.json",
            "report.json",
            "split.json",
            "predictions.jsonl",
            "manifest.jsonl",
            "errors.jsonl",
            "summary.json",
            "input.json",
            "split-manifest.json",
            "train-features.jsonl",
            "validation-features.jsonl",
            "test-features.jsonl",
            "validation-report.json",
            "test-report.json",
            "history.json",
            "search.json",
            "resolved-features.json",
        )
    }
    inputs = set(source_paths)
    if config.model:
        inputs.add((base / config.model).resolve())
    if inputs & outputs:
        raise ValueError("output files would overwrite an input; choose a different output_dir")
    prepared: dict[str, Iterable[Sample]] = {}
    supervised = config.task in {"train", "evaluate", "search"}
    for name, sources in source_groups.items():
        iterator = iter_sources(
            sources,
            base_dir=base,
            recursive=config.recursive,
            extensions=config.extensions,
            exclude_dirs=(output,),
            require_labels=supervised,
            batch_size=config.batch_size,
        )
        if config.manifest is not None and Path(config.manifest).suffix.lower() == ".csv":
            prepared[name] = read_manifest(base / config.manifest, require_labels=supervised)
        else:
            prepared[name] = list(iterator) if supervised else iterator
    samples = prepared.get("input", ())
    manifest_samples = list(samples) if isinstance(samples, list) else []
    split_samples = (
        {name: list(items) for name, items in prepared.items()}
        if (config.task in {"train", "search"} and config.data is not None)
        else {}
    )
    inputs.update(
        Path(s.path).resolve()
        for rows in [manifest_samples, *split_samples.values()]
        for s in rows
        if s.image_bytes is None
    )
    if inputs & outputs:
        raise ValueError("output files would overwrite an input; choose a different output_dir")
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fractal_detector")
    old_level = logger.level
    handler = configure_logging(output / "events.jsonl", config.log_level)
    try:
        atomic_json(output / "config.json", config.model_dump(mode="json"))
        if manifest_samples:
            atomic_json(
                output / "manifest.json", [s.model_dump(mode="json") for s in manifest_samples]
            )
        provenance = {
            name: [s.model_dump(mode="json") for s in rows] for name, rows in split_samples.items()
        }
        if split_samples:
            atomic_json(output / "split-manifest.json", provenance)
        environment = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "fractal-image-detector",
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "pillow",
                    "pydantic",
                    "pyarrow",
                )
            },
        }
        atomic_json(output / "environment.json", environment)
        with TrackingRun(config.tracking) as tracking:
            tracking.log_parameters({"task": config.task, "workers": config.workers})
            tracking.log_dict(config.model_dump(mode="json"), "config.json")
            tracking.log_dict(environment, "environment.json")
            if manifest_samples:
                tracking.log_dict(
                    {"samples": [s.model_dump(mode="json") for s in manifest_samples]},
                    "manifest.json",
                )
            if split_samples:
                tracking.log_dict(provenance, "split-manifest.json")
            else:
                tracking.log_dict(
                    {
                        "paths": sorted(str(p) for p in source_paths),
                        "split": config.split,
                        "recursive": config.recursive,
                        "extensions": config.extensions,
                    },
                    "input.json",
                )
            summary: dict[str, object] = {
                "task": config.task,
                "samples": len(manifest_samples),
                "output_dir": str(output),
                "run_id": tracking.run_id,
            }
            match "neural" if config.neural is not None else config.task:
                case "search":
                    from fractal_detector.search import search_features

                    if config.search is None:
                        raise ValueError("search configuration is required")
                    searched = search_features(
                        split_samples["train"],
                        split_samples["validation"],
                        config.search,
                        config.training,
                        workers=config.workers,
                    )
                    searched.detector.save(output / "model.json")
                    atomic_json(output / "search.json", searched.report)
                    tracking.log_dict(searched.report, "search.json")
                    tracking.log_artifact(output / "model.json")
                    for index, trial in enumerate(searched.trials):
                        tracking.log_metrics(
                            {f"search.{k}": v for k, v in trial.metrics.items()}, step=index
                        )
                    summary.update(
                        samples=sum(map(len, split_samples.values())), search=searched.report
                    )
                case "neural":
                    from fractal_detector.neural_runner import run_neural_task

                    summary = run_neural_task(config, base, output, prepared, tracking)
                case "extract":
                    summary = stream_task(samples, config, output, tracking)
                case "train":
                    if config.data is not None:
                        explicit = train_splits(
                            split_samples["train"],
                            split_samples.get("validation"),
                            split_samples.get("test"),
                            config.features,
                            config.training,
                            workers=config.workers,
                        )
                        _write_split_training(output, explicit)
                        tracking.log_training(explicit)
                        summary["split_counts"] = {k: len(v) for k, v in split_samples.items()}
                        summary["samples"] = sum(len(v) for v in split_samples.values())
                        summary["metrics"] = {k: v.metrics for k, v in explicit.evaluations.items()}
                    else:
                        trained = train(
                            manifest_samples,
                            config.features,
                            config.training,
                            workers=config.workers,
                        )
                        trained.detector.save(output / "model.json")
                        _write_jsonl(
                            output / "features.jsonl",
                            [r.model_dump(mode="json") for r in trained.measurements],
                        )
                        atomic_json(
                            output / "split.json",
                            {
                                "train_indices": trained.train_indices,
                                "test_indices": trained.test_indices,
                                "image_ids": [r.image_id for r in trained.measurements],
                            },
                        )
                        atomic_json(
                            output / "report.json", trained.evaluation.model_dump(mode="json")
                        )
                        tracking.log_training(trained)
                        summary["metrics"] = trained.evaluation.metrics
                case "evaluate" | "predict":
                    if config.model is None:
                        raise ValueError("model is required")
                    detector = Detector.load(base / config.model)
                    if (
                        "features" in config.model_fields_set
                        and resolve_features(
                            config.features, detector.artifact.feature_config.image_size
                        )
                        != detector.artifact.feature_config
                    ):
                        raise ValueError(
                            "prediction/evaluation preprocessing must match the saved model; omit features"
                        )
                    if config.task == "evaluate":
                        evaluated = evaluate(detector, manifest_samples, workers=config.workers)
                        atomic_json(output / "report.json", evaluated.model_dump(mode="json"))
                        tracking.log_evaluation(evaluated)
                        summary["metrics"] = evaluated.metrics
                    else:
                        summary = stream_task(samples, config, output, tracking, detector)
            logger.info("task completed", extra={"event": "task_completed", "data": summary})
            tracking.log_artifact(output / "events.jsonl")
            return summary
    except Exception:
        logger.exception("task failed", extra={"event": "task_failed"})
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(old_level)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n")


def _write_split_training(output: Path, result: SplitTrainingResult) -> None:
    result.detector.save(output / "model.json")
    atomic_json(
        output / "split.json",
        {
            "mode": "explicit",
            "image_ids": {k: [r.image_id for r in v] for k, v in result.measurements.items()},
        },
    )
    for name, rows in result.measurements.items():
        _write_jsonl(output / f"{name}-features.jsonl", [r.model_dump(mode="json") for r in rows])
    reports = {name: report.model_dump(mode="json") for name, report in result.evaluations.items()}
    for name, report in reports.items():
        atomic_json(output / f"{name}-report.json", report)
    atomic_json(output / "report.json", reports)
