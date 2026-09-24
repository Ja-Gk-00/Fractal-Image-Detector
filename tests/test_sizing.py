import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fractal_detector import (
    Detector,
    FeatureConfig,
    Sample,
    TaskConfig,
    TrainingConfig,
    extract,
    extract_dataset,
    fit_features,
    run_task,
    train,
    train_splits,
)
from fractal_detector.config import NeuralConfig, ResolvedFeatureConfig, ResolvedNeuralConfig
from fractal_detector.sizing import (
    auto_sized_stream,
    image_side,
    largest_side,
    resolve_features,
    resolve_neural,
)


def encoded(width: int, height: int) -> bytes:
    buffer = BytesIO()
    pixels = np.random.default_rng(1337).integers(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def test_default_seed_and_image_sizes() -> None:
    assert TrainingConfig().seed == 1337
    assert FeatureConfig().image_size == "auto"
    assert NeuralConfig().image_size == "auto"
    with pytest.raises(ValueError):
        ResolvedFeatureConfig(image_size="auto")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        ResolvedNeuralConfig(image_size="auto")  # ty: ignore[invalid-argument-type]


def test_auto_uses_longest_dimension_and_shared_geometry() -> None:
    samples = [
        Sample(path="small", image_bytes=encoded(32, 32)),
        Sample(path="large", image_bytes=encoded(90, 45)),
    ]
    features = FeatureConfig(compression=False)
    measured = extract_dataset(samples, features, workers=2)
    assert [r.processed_shape for r in measured] == [(96, 96), (96, 96)]
    assert measured[1].original_shape == (45, 90)
    stored = measured[0].diagnostics["config"]
    assert isinstance(stored, dict) and stored["image_size"] == 96
    assert extract(encoded(90, 45), features).processed_shape == (96, 96)
    assert extract(np.zeros((45, 90)), features).processed_shape == (96, 96)
    fixed = extract_dataset(samples, features.model_copy(update={"image_size": 64}))
    assert {r.processed_shape for r in fixed} == {(64, 64)}


def test_geometry_and_limits() -> None:
    assert resolve_features(FeatureConfig(scales=(2, 3, 8), block_size=3), 90).image_size == 96
    assert (
        resolve_features(FeatureConfig(low_correlation=True, pca_ranks=(128,)), 32).image_size
        == 160
    )
    assert resolve_features(FeatureConfig(), 1500).image_size == 1504
    for architecture in ("srm_fractal", "residual_fractal"):
        config = NeuralConfig.model_validate({"architecture": architecture})
        result = resolve_neural(config, 641, 40_000_000)
        assert result.image_size >= 641
        assert result.image_size % 8 == 0
    assert resolve_neural(NeuralConfig(levels=1), 33, 40_000_000).image_size == 40
    with pytest.raises(ValueError, match="max_input_pixels"):
        resolve_features(FeatureConfig(max_input_pixels=100_000), 500)
    with pytest.raises(ValueError, match="max_input_pixels"):
        resolve_neural(NeuralConfig(), 500, 100_000)


@pytest.mark.parametrize(
    "image", [np.zeros((0, 32)), np.zeros((32, 32, 2)), np.zeros((32,)), encoded(16, 16)]
)
def test_metadata_limits(image) -> None:
    with pytest.raises(ValueError):
        image_side(image, 100)


def test_metadata_rejects_high_depth_and_animation() -> None:
    for mode, format_name in (("F", "TIFF"), ("RGB", "GIF")):
        buffer = BytesIO()
        image = Image.new(mode, (32, 32))
        if format_name == "GIF":
            image.save(
                buffer,
                format=format_name,
                save_all=True,
                append_images=[Image.new(mode, (32, 32), "white")],
            )
        else:
            image.save(buffer, format=format_name)
        with pytest.raises(ValueError):
            image_side(buffer.getvalue(), 40_000_000)


def test_stream_resolution_replays_bytes_order_and_errors(tmp_path: Path) -> None:
    rows = [
        Sample(path="small", image_bytes=encoded(32, 32)),
        Sample(path="broken", image_bytes=b"broken"),
        Sample(path="large", image_bytes=encoded(96, 64)),
    ]
    with auto_sized_stream(iter(rows), FeatureConfig(compression=False), on_error="skip") as (
        config,
        replay,
    ):
        assert config.image_size == 96
        assert list(replay) == rows
    with (
        pytest.raises(ValueError, match="cannot measure"),
        auto_sized_stream(iter(rows), FeatureConfig()),
    ):
        pass
    with auto_sized_stream(iter(rows), FeatureConfig(image_size=64)) as (config, replay):
        assert config.image_size == 64
        assert list(replay) == rows
    assert largest_side([], 40_000_000) == 0
    for i, row in enumerate(rows):
        assert row.image_bytes is not None
        (tmp_path / f"{i}.png").write_bytes(row.image_bytes)
    task = TaskConfig(
        task="extract",
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / "result"),
        features=FeatureConfig(compression=False),
        on_error="skip",
        batch_size=1,
    )
    summary = run_task(task)
    assert summary["succeeded"] == 2 and summary["failed"] == 1
    extracted = [
        json.loads(row) for row in (tmp_path / "result/features.jsonl").read_text().splitlines()
    ]
    assert [r["processed_shape"] for r in extracted] == [[96, 96], [96, 96]]
    assert json.loads((tmp_path / "result/resolved-features.json").read_text())["image_size"] == 96


def test_training_sizing_ignores_validation_and_persists(dataset, tmp_path: Path) -> None:
    samples, _ = dataset
    training = [s for s in samples if s.group in {"source0", "source1", "source2", "source3"}]
    validation = [s for s in samples if s.group in {"source4", "source5"}]
    for sample in validation:
        with Image.open(sample.path) as image:
            image.resize((96, 80)).save(sample.path)
    config = FeatureConfig(scales=(2, 4, 8), compression=False)
    result = train_splits(training, validation, feature_config=config)
    assert result.detector.artifact.feature_config.image_size == 32
    assert {r.processed_shape for rows in result.measurements.values() for r in rows} == {(32, 32)}
    model = tmp_path / "model.json"
    result.detector.save(model)
    loaded = Detector.load(model)
    assert loaded.predict(encoded(128, 96)).measurement.processed_shape == (32, 32)
    assert loaded.artifact.training_config.seed == 1337
    fitted = fit_features(training, extract_dataset(training, config), config)
    assert fitted.artifact.feature_config.image_size == 32


def test_implicit_split_resolves_training_only(dataset) -> None:
    samples, _ = dataset
    for i, sample in enumerate(samples):
        with Image.open(sample.path) as image:
            image.resize((32 + 8 * i, 32 + 8 * i)).save(sample.path)
    result = train(samples, FeatureConfig(scales=(2, 4, 8), compression=False))
    expected = max(32 + 8 * i for i in result.train_indices)
    assert result.detector.artifact.feature_config.image_size == expected


def test_toml_presets_roundtrip_with_quoted_paths(tmp_path: Path) -> None:
    from fractal_detector import load_config, write_method_configs

    source = 'D:\\images with spaces\\a"b.csv'
    paths = write_method_configs(tmp_path, train=source)
    assert len(paths) == 36
    assert {p.suffix for p in paths} == {".toml"}
    for path in paths:
        config = load_config(path)
        assert config.features.image_size == "auto"
        assert config.data is not None and config.data.train == (source,)
        assert config.training.seed == 1337
        if config.neural is not None:
            assert config.neural.image_size == "auto"
