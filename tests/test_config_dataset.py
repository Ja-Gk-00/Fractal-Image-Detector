import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fractal_detector import (
    FeatureConfig,
    Sample,
    TaskConfig,
    extract_dataset,
    load_config,
    read_manifest,
)


@pytest.mark.parametrize(
    "values",
    [
        {"scales": (2, 4)},
        {"scales": (4, 2, 8)},
        {"scales": (2, 3, 8)},
        {"image_size": 30},
        {"block_size": 3},
        {"svd_rank": 128},
        {"transform_ids": ()},
        {"transform_ids": (0, 0)},
        {"transform_ids": (8,)},
        {"gaussian_sigma": float("nan")},
        {"backend": "gpu"},
        {"unknown": 1},
        {"max_domains": True},
    ],
)
def test_reject_invalid_feature_configuration(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        FeatureConfig.model_validate({"image_size": 128, **values})


@pytest.mark.parametrize(
    ("extension", "content"),
    [
        ("toml", 'task="extract"\nmanifest="images.csv"'),
        ("json", '{"task":"extract","manifest":"images.csv"}'),
        ("yaml", "task: extract\nmanifest: images.csv"),
    ],
)
def test_load_configuration(tmp_path: Path, extension: str, content: str) -> None:
    path = tmp_path / f"config.{extension}"
    path.write_text(content)
    assert load_config(path).task == "extract"


def test_configuration_errors(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="require a model"):
        TaskConfig(task="predict", manifest="data.csv")
    path = tmp_path / "config.txt"
    path.write_text("task=extract")
    with pytest.raises(ValueError, match="configuration must"):
        load_config(path)
    path = tmp_path / "config.yaml"
    path.write_text('!!python/object/apply:os.system ["echo unsafe"]')
    with pytest.raises(Exception, match="constructor"):
        load_config(path)


def test_manifest_resolves_relative_paths(dataset) -> None:
    samples, manifest = dataset
    assert read_manifest(manifest, require_labels=True) == samples


@pytest.mark.parametrize(
    "content",
    [
        "",
        "label\nreal\n",
        "path,path\na,a\n",
        "path,bad\na,b\n",
        "path\nmissing.png\n",
        "path,label\n,0\n",
        "path,label\na.png,wrong\n",
        "path,label\na.png,\n",
        "path\na.png\na.png\n",
        "path,label\na.png\n",
        "path,label\na.png,0,extra\n",
    ],
)
def test_manifest_rejects_bad_data(tmp_path: Path, content: str) -> None:
    (tmp_path / "a.png").touch()
    manifest = tmp_path / "data.csv"
    manifest.write_text(content)
    with pytest.raises(ValueError):
        read_manifest(manifest, require_labels=True)


def test_unlabeled_manifest_and_corrupt_file(tmp_path: Path, config: FeatureConfig) -> None:
    image = tmp_path / "corrupt.png"
    image.write_text("not an image")
    manifest = tmp_path / "data.csv"
    manifest.write_text("path\ncorrupt.png\n")
    samples = read_manifest(manifest)
    assert samples[0].label is None
    with pytest.raises(ValueError, match="cannot measure"):
        extract_dataset(samples, config)


def test_json_config_preserves_strict_types(tmp_path: Path) -> None:
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"task": "extract", "manifest": "data.csv", "workers": "2"}))
    with pytest.raises(ValidationError):
        load_config(path)
    with pytest.raises(ValidationError):
        Sample.model_validate({"path": "image.png", "label": 2})
