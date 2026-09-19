import json
from collections.abc import Generator
from itertools import islice
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from fractal_detector import (
    FeatureConfig,
    ImageFailure,
    Prediction,
    TaskConfig,
    iter_directory,
    run_task,
    train,
)
from fractal_detector.dataset import measure_batch


def test_directory_scanning(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    excluded = tmp_path / "excluded"
    excluded.mkdir()
    for path in (tmp_path / "first.PNG", nested / "second.jpg", excluded / "third.png"):
        path.touch()
    (tmp_path / "readme.txt").touch()
    samples = list(iter_directory(tmp_path, exclude_dirs=(excluded,)))
    assert {Path(s.path).name for s in samples} == {"first.PNG", "second.jpg"}
    assert len(list(iter_directory(tmp_path, recursive=False))) == 1
    assert len(list(iter_directory(tmp_path, extensions=(".jpg",)))) == 1
    assert list(iter_directory(tmp_path, exclude_dirs=(tmp_path,))) == []
    with pytest.raises(ValueError, match="does not exist"):
        list(iter_directory(tmp_path / "missing"))


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"manifest": "a.csv", "input_dir": "images"},
        {"input_dir": ""},
        {"input_dir": "images", "extensions": ()},
        {"input_dir": "images", "extensions": ("png",)},
        {"input_dir": "images", "extensions": (".png", ".png")},
        {"input_dir": "images", "batch_size": 0},
        {"input_dir": "images", "batch_size": 4097},
    ],
)
def test_invalid_directory_configuration(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TaskConfig.model_validate({"task": "extract", **values})


def test_supervised_tasks_require_manifest() -> None:
    with pytest.raises(ValidationError, match="labeled manifest"):
        TaskConfig(task="train", input_dir="images")
    with pytest.raises(ValidationError, match="on_error"):
        TaskConfig(task="evaluate", manifest="images.csv", model="model.json", on_error="skip")


def test_directory_predictions_match_single_images(
    dataset, config: FeatureConfig, tmp_path: Path
) -> None:
    samples, _ = dataset
    detector = train(samples, config).detector
    detector.save(tmp_path / "model.json")
    output = tmp_path / "results"
    output.mkdir()
    Image.new("RGB", (32, 32)).save(output / "do-not-scan.png")
    task = TaskConfig(
        task="predict",
        input_dir=str(tmp_path),
        model=str(tmp_path / "model.json"),
        output_dir=str(output),
        workers=2,
        batch_size=5,
    )
    summary = run_task(task)
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert summary["samples"] == len(samples)
    assert summary["succeeded"] == len(samples)
    assert summary["status"] == "completed"
    assert summary == json.loads((output / "summary.json").read_text())
    assert [
        r["index"] for r in map(json.loads, (output / "manifest.jsonl").read_text().splitlines())
    ] == list(range(len(samples)))
    for row in rows:
        assert (
            row["synthetic_probability"]
            == detector.predict(row["measurement"]["source"]).synthetic_probability
        )


def test_directory_iterator_is_bounded(dataset, config: FeatureConfig) -> None:
    samples, _ = dataset
    detector = train(samples, config).detector
    consumed = 0

    def repeated():
        nonlocal consumed
        for _ in range(100_000):
            consumed += 1
            yield samples[0]

    iterator = detector.iter_predict(repeated(), batch_size=3, workers=2)
    first = next(iterator)
    assert isinstance(first, Prediction)
    assert consumed == 3
    assert isinstance(iterator, Generator)
    iterator.close()


def test_directory_api_and_validation(dataset, config: FeatureConfig, tmp_path: Path) -> None:
    samples, _ = dataset
    detector = train(samples, config).detector
    assert len(list(islice(detector.predict_directory(tmp_path, batch_size=2), 3))) == 3
    for size in (0, True, 4097):
        with pytest.raises(ValueError, match="batch_size"):
            next(detector.iter_predict(samples, batch_size=size))
    with pytest.raises(ValueError, match="on_error"):
        next(detector.iter_predict(samples, on_error="ignore"))  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="workers"):
        measure_batch(samples, config, workers=True)


def test_corrupt_images_are_reported(tmp_path: Path, config: FeatureConfig) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32), "blue").save(images / "good.png")
    (images / "broken.png").write_text("invalid image")
    output = tmp_path / "result"
    task = TaskConfig(
        task="extract",
        input_dir=str(images),
        output_dir=str(output),
        features=config,
        on_error="skip",
        batch_size=1,
    )
    summary = run_task(task)
    assert summary["status"] == "completed_with_errors"
    assert summary["succeeded"] == 1 and summary["failed"] == 1
    errors = [json.loads(line) for line in (output / "errors.jsonl").read_text().splitlines()]
    assert len(errors) == 1
    assert errors[0]["path"].endswith("broken.png")
    assert "gray_dimension" in (output / "features.jsonl").read_text()


def test_failure_retains_partial_output(dataset, config: FeatureConfig, tmp_path: Path) -> None:
    samples, _ = dataset
    corrupt = tmp_path / "broken.png"
    corrupt.write_text("broken")
    manifest = tmp_path / "ordered.csv"
    manifest.write_text(f"path\n{samples[0].path}\n{corrupt}\n")
    output = tmp_path / "aborted"
    with pytest.raises(ValueError, match="cannot measure"):
        run_task(
            TaskConfig(
                task="extract",
                manifest=str(manifest),
                output_dir=str(output),
                features=config,
                batch_size=1,
            )
        )
    assert len((output / "features.jsonl").read_text().splitlines()) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["failed"] == 1
    assert len((output / "errors.jsonl").read_text().splitlines()) == 1
    detector = train(samples, config).detector
    failures = list(detector.predict_directory(tmp_path, extensions=(".png",), on_error="skip"))
    assert any(isinstance(r, ImageFailure) for r in failures)
    with pytest.raises(ValueError, match="cannot measure"):
        list(detector.predict_directory(tmp_path))


def test_empty_and_unreadable_directories(tmp_path: Path, config: FeatureConfig) -> None:
    images = tmp_path / "images"
    images.mkdir()
    task = TaskConfig(
        task="extract", input_dir=str(images), output_dir=str(tmp_path / "out"), features=config
    )
    with pytest.raises(ValueError, match="no matching"):
        run_task(task)
    (images / "broken.png").touch()
    with pytest.raises(ValueError, match="no images were processed"):
        run_task(task.model_copy(update={"on_error": "skip"}))
    with pytest.raises(ValueError, match="must not equal"):
        run_task(task.model_copy(update={"output_dir": str(images)}))
