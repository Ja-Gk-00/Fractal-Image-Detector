import json
import logging
from collections.abc import Iterable
from contextlib import ExitStack
from itertools import batched
from pathlib import Path
from time import perf_counter
from typing import TextIO

from fractal_detector.config import TaskConfig
from fractal_detector.dataset import ImageFailure, Sample, measure_batch
from fractal_detector.io import atomic_json
from fractal_detector.model import Detector
from fractal_detector.sizing import auto_sized_stream
from fractal_detector.tracking import TrackingRun

logger = logging.getLogger("fractal_detector")


def _write(stream: TextIO, record: object) -> None:
    stream.write(json.dumps(record, allow_nan=False, ensure_ascii=False) + "\n")


def stream_task(
    samples: Iterable[Sample],
    config: TaskConfig,
    output: Path,
    tracking: TrackingRun,
    detector: Detector | None = None,
) -> dict[str, object]:
    if detector is None and config.features.image_size == "auto":
        with auto_sized_stream(samples, config.features, on_error=config.on_error) as (
            resolved,
            replay,
        ):
            atomic_json(output / "resolved-features.json", resolved.model_dump(mode="json"))
            tracking.log_dict(resolved.model_dump(mode="json"), "resolved-features.json")
            return stream_task(
                replay, config.model_copy(update={"features": resolved}), output, tracking
            )
    features = config.features if detector is None else detector.artifact.feature_config
    tracking.log_parameters(
        {
            "features": features.model_dump(mode="json"),
            "batch_size": config.batch_size,
            "on_error": config.on_error,
        }
    )
    start = perf_counter()
    seen = succeeded = failed = 0
    summary: dict[str, object] = {
        "task": config.task,
        "output_dir": str(output),
        "run_id": tracking.run_id,
    }
    filename = "features.jsonl" if detector is None else "predictions.jsonl"

    def checkpoint(status: str) -> None:
        summary.update(
            status=status,
            samples=seen,
            succeeded=succeeded,
            failed=failed,
            elapsed_seconds=perf_counter() - start,
        )
        atomic_json(output / "summary.json", summary)

    try:
        with ExitStack() as stack:
            results_file = stack.enter_context((output / filename).open("w", encoding="utf-8"))
            errors_file = stack.enter_context((output / "errors.jsonl").open("w", encoding="utf-8"))
            manifest_file = stack.enter_context(
                (output / "manifest.jsonl").open("w", encoding="utf-8")
            )
            checkpoint("running")
            for batch_number, batch in enumerate(batched(samples, config.batch_size, strict=False)):
                sums: dict[str, float] = {}
                batch_succeeded = 0
                shard_rows: list[dict[str, object]] = []
                for sample, result in zip(
                    batch, measure_batch(batch, features, workers=config.workers), strict=True
                ):
                    index = seen
                    seen += 1
                    _write(manifest_file, {"index": index, **sample.model_dump(mode="json")})
                    if isinstance(result, ImageFailure):
                        failed += 1
                        error = {"index": index, **result.model_dump(mode="json")}
                        _write(errors_file, error)
                        shard_rows.append({"index": index, "kind": "error", "result": error})
                        logger.warning(
                            "image processing failed", extra={"event": "image_error", "data": error}
                        )
                        if config.on_error == "raise":
                            raise ValueError(f"cannot measure {result.path}: {result.message}")
                        continue
                    if detector is None:
                        record = result.model_dump(mode="json")
                        if config.tracking.image_logging == "full":
                            tracking.log_features([result], start_step=index)
                    else:
                        prediction = detector.predict_features(result)
                        record = prediction.model_dump(mode="json")
                        if config.tracking.image_logging == "full":
                            tracking.log_predictions([prediction], start_step=index)
                        sums["synthetic_probability"] = (
                            sums.get("synthetic_probability", 0.0)
                            + prediction.synthetic_probability
                        )
                    _write(results_file, record)
                    shard_rows.append({"index": index, "kind": config.task, "result": record})
                    succeeded += 1
                    batch_succeeded += 1
                    for name, value in result.features.items():
                        sums[name] = sums.get(name, 0.0) + value
                for stream in (results_file, errors_file, manifest_file):
                    stream.flush()
                if config.tracking.enabled and config.tracking.image_logging == "batch":
                    shard_dir = output / "batches"
                    shard_dir.mkdir(exist_ok=True)
                    shard = shard_dir / f"batch-{batch_number:08d}.jsonl"
                    with shard.open("w", encoding="utf-8") as stream:
                        for row in shard_rows:
                            _write(stream, row)
                    tracking.log_artifact(shard, artifact_path="batches")
                    if batch_succeeded:
                        tracking.log_metrics(
                            {f"batch.{k}.mean": v / batch_succeeded for k, v in sums.items()},
                            step=batch_number,
                        )
                tracking.log_metrics(
                    {
                        "progress.processed": float(seen),
                        "progress.succeeded": float(succeeded),
                        "progress.failed": float(failed),
                    },
                    step=batch_number,
                )
                checkpoint("running")
                logger.info(
                    "batch completed", extra={"event": "batch_completed", "data": dict(summary)}
                )
            if seen == 0:
                raise ValueError("no matching images found")
            if succeeded == 0:
                raise ValueError("no images were processed successfully; see errors.jsonl")
        checkpoint("completed_with_errors" if failed else "completed")
    except BaseException:
        checkpoint("failed")
        raise
    tracking.log_dict(summary, "summary.json")
    return summary
