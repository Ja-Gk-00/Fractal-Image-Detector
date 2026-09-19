import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
from pydantic import ValidationError

from fractal_detector import (
    DataSource,
    DataSplits,
    FeatureConfig,
    ImageFailure,
    TaskConfig,
    extract,
    extract_dataset,
    iter_manifest,
    iter_sources,
    read_manifest,
    run_task,
    train,
)


def write_parquet(path: Path, rows: list[dict[str, object]]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    return path


def test_parquet_paths_and_custom_columns(dataset, tmp_path: Path) -> None:
    samples, _ = dataset
    path = write_parquet(
        tmp_path / "images.parquet",
        [
            {"path": Path(s.path).name, "label": s.label, "group": s.group, "metadata": "ignored"}
            for s in samples
        ],
    )
    assert read_manifest(path, require_labels=True) == samples
    assert list(iter_manifest(path, batch_size=3)) == samples
    custom = write_parquet(
        tmp_path / "custom.parquet",
        [{"filename": Path(samples[0].path).name, "category": "Camera", "subject": 7}],
    )
    source = DataSource(
        paths=(str(custom),),
        path_column="filename",
        label_column="category",
        group_column="subject",
        label_map={"camera": 0, "generator": 1},
    )
    rows = list(iter_sources((source,), require_labels=True))
    assert rows[0].label == 0 and rows[0].group == "7"


@pytest.mark.parametrize("nested", [False, True])
def test_embedded_images_preserve_measurements_and_hide_bytes(
    dataset, config: FeatureConfig, tmp_path: Path, nested: bool
) -> None:
    samples, _ = dataset
    payload = Path(samples[0].path).read_bytes()
    image = {"bytes": payload, "path": "original.png"} if nested else payload
    path = write_parquet(tmp_path / "embedded.parquet", [{"image": image, "label": "real"}])
    loaded = read_manifest(path, require_labels=True)
    assert loaded[0].image_bytes == payload
    assert "image_bytes" not in loaded[0].model_dump(mode="json")
    measurement = extract_dataset(loaded, config)[0]
    reference = extract(samples[0].path, config)
    assert measurement.features == reference.features
    assert measurement.image_id == reference.image_id
    assert measurement.source == str(path) + "#row=0"
    assert extract(payload, config).features == reference.features


def test_nullable_nested_paths_and_source_labels(dataset, tmp_path: Path) -> None:
    samples, _ = dataset
    path = write_parquet(
        tmp_path / "paths.parquet",
        [{"image": {"bytes": None, "path": Path(samples[0].path).name}, "label": None}],
    )
    loaded = list(iter_sources((DataSource(paths=(str(path),), label=0),), require_labels=True))
    assert loaded[0].path == samples[0].path and loaded[0].label == 0


@pytest.mark.parametrize(
    "row, message",
    [
        ({"wrong": "x"}, "requires a path or image"),
        ({"path": None}, "missing image path"),
        ({"path": "absent.png"}, "image not found"),
        ({"image": "not encoded"}, "encoded bytes"),
        ({"image": b"x", "label": True}, "label"),
        ({"image": b"x", "label": 2}, "label"),
        ({"image": b"x", "label": 0.5}, "label"),
        ({"image": b"x", "group": 1.5}, "group"),
        ({"image": b"x", "group": True}, "group"),
    ],
)
def test_rejects_invalid_parquet_rows(tmp_path: Path, row: dict[str, object], message: str) -> None:
    path = write_parquet(tmp_path / "bad.parquet", [row])
    with pytest.raises(ValueError, match=message):
        read_manifest(path)


def test_empty_parquet_missing_labels_and_conflicting_labels(tmp_path: Path) -> None:
    path = tmp_path / "empty.parquet"
    pq.write_table(pa.table({"path": pa.array([], type=pa.string())}), path)
    with pytest.raises(ValueError, match="empty"):
        read_manifest(path)
    write_parquet(path, [{"image": b"x"}])
    with pytest.raises(ValueError, match="missing label"):
        list(iter_manifest(path, require_labels=True))
    write_parquet(path, [{"image": b"x", "label": 1}])
    with pytest.raises(ValueError, match="conflicts"):
        list(iter_sources((DataSource(paths=(str(path),), label=0),)))
    for size in (0, True):
        with pytest.raises(ValueError, match="batch_size"):
            list(iter_manifest(path, batch_size=size))
    with pytest.raises(ValueError, match="CSV or Parquet"):
        read_manifest(tmp_path / "a.json")


def test_multiple_nested_sources_deduplicate_and_exclude_outputs(
    tmp_path: Path, config: FeatureConfig
) -> None:
    images = tmp_path / "images"
    nested = images / "nested" / "deeper"
    nested.mkdir(parents=True)
    a = images / "a.PNG"
    b = nested / "b.jpg"
    Image.new("RGB", (32, 32), "blue").save(a)
    Image.new("RGB", (32, 32), "red").save(b)
    write_parquet(nested / "shard.parquet", [{"path": b.name}])
    csv = tmp_path / "images.csv"
    csv.write_text("path\nimages/a.PNG\n", encoding="utf-8")
    paths = (str(images), str(nested), str(a), str(csv))
    assert {s.path for s in iter_sources(paths)} == {str(a), str(b)}
    assert [s.path for s in iter_sources((str(images),), recursive=False)] == [str(a)]
    output = images / "out"
    output.mkdir()
    Image.new("RGB", (32, 32)).save(output / "ignored.png")
    summary = run_task(
        TaskConfig(
            task="extract", inputs=paths, output_dir=str(output), features=config, batch_size=1
        )
    )
    assert summary["succeeded"] == 2
    assert len((output / "manifest.jsonl").read_text().splitlines()) == 2


def test_duplicate_source_metadata_conflicts(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    path.touch()
    with pytest.raises(ValueError, match="conflicting labels/groups"):
        list(
            iter_sources(
                (DataSource(paths=(str(path),), label=0), DataSource(paths=(str(path),), label=1))
            )
        )
    with pytest.raises(ValueError, match="missing label"):
        list(iter_sources((str(path),), require_labels=True))
    for paths in ((), (str(tmp_path / "missing"),)):
        with pytest.raises(ValueError):
            list(iter_sources(paths))


def test_parquet_prediction_api_and_skip_errors(
    dataset, config: FeatureConfig, tmp_path: Path
) -> None:
    samples, _ = dataset
    detector = train(samples, config).detector
    path = write_parquet(
        tmp_path / "prediction.parquet",
        [{"image": Path(samples[0].path).read_bytes()}, {"image": b"invalid"}],
    )
    results = list(detector.predict_sources((str(path),), batch_size=1, on_error="skip"))
    assert not isinstance(results[0], ImageFailure)
    assert (
        results[0].synthetic_probability == detector.predict(samples[0].path).synthetic_probability
    )
    assert isinstance(results[1], ImageFailure)
    detector.save(tmp_path / "model.json")
    task = TaskConfig(
        task="predict",
        manifest=str(path),
        model=str(tmp_path / "model.json"),
        output_dir=str(tmp_path / "predictions"),
        batch_size=1,
        on_error="skip",
    )
    summary = run_task(task)
    assert summary["succeeded"] == summary["failed"] == 1
    assert "image_bytes" not in (tmp_path / "predictions/manifest.jsonl").read_text()


@pytest.mark.parametrize(
    "values",
    [
        {"inputs": ()},
        {"inputs": ("",)},
        {"inputs": ("a",), "manifest": "b.csv"},
        {"inputs": ("a",), "split": "test"},
        {"data": {}},
        {"data": {"test": [""]}},
        {"data": {"test": ["a"]}},
        {"data": {"test": ["a"]}, "split": "validation"},
    ],
)
def test_invalid_source_configurations(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TaskConfig.model_validate_json(json.dumps({"task": "extract", **values}))


@pytest.mark.parametrize(
    "values",
    [
        {"paths": ()},
        {"paths": (" ",)},
        {"paths": ("a",), "path_column": "label"},
        {"paths": ("a",), "path_column": ""},
        {"paths": ("a",), "label_map": {}},
        {"paths": ("a",), "label_map": {"REAL": 0}},
        {"paths": ("a",), "label_map": {"": 0}},
    ],
)
def test_invalid_source_options(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        DataSource.model_validate(values)


def test_explicit_split_selector(tmp_path: Path, config: FeatureConfig) -> None:
    path = tmp_path / "a.png"
    Image.new("RGB", (32, 32)).save(path)
    data = DataSplits(train=("missing-train.parquet",), test=(str(path),))
    summary = run_task(
        TaskConfig(
            task="extract",
            data=data,
            split="test",
            features=config,
            output_dir=str(tmp_path / "output"),
        )
    )
    assert summary["samples"] == 1
    for values in (
        {"data": DataSplits(test=("test.csv",))},
        {"data": DataSplits(train=("train.csv",)), "split": "train"},
    ):
        with pytest.raises(ValidationError, match=r"data\.train"):
            TaskConfig.model_validate({"task": "train", **values})
