import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fractal_detector import FeatureConfig, extract, extract_dataset
from fractal_detector.features import feature_names, noise_residual
from fractal_detector.io import atomic_json, load_image


def test_features_are_finite_and_serializable(config: FeatureConfig) -> None:
    result = extract(np.random.default_rng(1337).random((32, 32, 3)), config)
    assert tuple(result.features) == feature_names(config)
    assert all(np.isfinite(v) for v in result.features.values())
    assert result.dimensions["gray"].value is not None
    json.dumps(result.model_dump(mode="json"), allow_nan=False)
    compression = result.diagnostics["compression"]
    assert isinstance(compression, dict)
    assert len(compression["matches_mse_contrast_offset_domain"]) == 64


def test_flat_image_reports_degeneracy(config: FeatureConfig) -> None:
    result = extract(np.ones((32, 32)), config)
    assert result.features["edge_dimension_valid"] == 0
    assert result.diagnostics["residual_empty"]
    assert result.warnings
    assert result.features["gray_dimension"] == pytest.approx(2)


def test_svd_removes_low_rank_signal(config: FeatureConfig) -> None:
    config = config.model_copy(update={"residual": "svd", "svd_rank": 1, "compression": False})
    matrix = np.outer(np.linspace(0, 1, 32), np.linspace(0, 1, 32))
    assert np.max(np.abs(noise_residual(matrix, config))) < 1e-12
    result = extract(matrix, config)
    assert "compression_mse_mean" not in result.features


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((4, 4)),
        np.zeros((32, 32, 2)),
        np.zeros((32, 32), dtype=np.int64),
        np.ones((32, 32)) * 256,
        np.full((32, 32), np.nan),
        np.empty((0, 32)),
    ],
)
def test_invalid_arrays(array: np.ndarray, config: FeatureConfig) -> None:
    with pytest.raises(ValueError):
        extract(array, config)


def test_rgba_crop_resize_and_limits(tmp_path: Path, config: FeatureConfig) -> None:
    rgba = np.zeros((48, 64, 4), dtype=np.uint8)
    path = tmp_path / "alpha.png"
    Image.fromarray(rgba).save(path)
    array_result = extract(rgba, config)
    file_result = extract(path, config)
    assert array_result.original_shape == (48, 64)
    assert array_result.processed_shape == (32, 32)
    np.testing.assert_allclose(
        list(array_result.features.values()), list(file_result.features.values())
    )
    assert any("alpha" in w for w in file_result.warnings)
    assert any("crop" in w for w in file_result.warnings)
    with pytest.raises(ValueError, match="max_input_pixels"):
        load_image(path, config.model_copy(update={"max_input_pixels": 10}))


def test_unsupported_files(tmp_path: Path, config: FeatureConfig) -> None:
    high = tmp_path / "depth.tiff"
    Image.fromarray(np.ones((32, 32), dtype=np.float32)).save(high)
    with pytest.raises(ValueError, match="high-bit-depth"):
        extract(high, config)
    animation = tmp_path / "movie.gif"
    Image.new("RGB", (32, 32), "red").save(
        animation, save_all=True, append_images=[Image.new("RGB", (32, 32), "blue")]
    )
    with pytest.raises(ValueError, match="multi-frame"):
        extract(animation, config)


def test_exif_orientation_applied(tmp_path: Path, config: FeatureConfig) -> None:
    path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (32, 48))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)
    assert extract(path, config).original_shape == (32, 48)


def test_atomic_json_preserves_old_file_on_invalid_data(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_json(path, {"value": 1})
    with pytest.raises(ValueError):
        atomic_json(path, {"value": float("nan")})
    assert json.loads(path.read_text())["value"] == 1


def test_threaded_extraction_matches_serial(dataset, config: FeatureConfig) -> None:
    samples, _ = dataset
    serial = extract_dataset(samples[:3], config)
    parallel = extract_dataset(samples[:3], config, workers=2)
    assert [r.features for r in serial] == [r.features for r in parallel]
    with pytest.raises(ValueError, match="workers"):
        extract_dataset(samples, config, workers=0)
