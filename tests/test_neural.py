import base64
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.special import expit

from fractal_detector import FeatureConfig, Sample, TrainingConfig
from fractal_detector.config import NeuralConfig, TaskConfig
from fractal_detector.dataset import ImageFailure
from fractal_detector.runner import run_task

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from fractal_detector.networks import FractalNetwork, quadrants  # noqa: E402
from fractal_detector.neural import (  # noqa: E402
    NeuralDetector,
    _device,
    _threads,
    evaluate_neural,
    train_neural,
)


@pytest.fixture
def splits(dataset):
    samples, _ = dataset
    return tuple(
        [s for s in samples if s.group in groups]
        for groups in ({"source0", "source1"}, {"source2"}, {"source3"})
    )


def test_quadrants_product_and_average() -> None:
    x = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    parts = quadrants(x)
    assert parts.shape == (4, 1, 1, 4, 4)
    assert parts.prod(dim=0)[0, 0, 1, 2] == 10 * 14 * 42 * 46
    assert parts.mean(dim=0)[0, 0, 1, 2] == 28
    with pytest.raises(ValueError):
        quadrants(torch.ones(1, 1, 7, 8))


@pytest.mark.parametrize(
    "architecture,residual",
    [
        ("srm_fractal", "median"),
        ("residual_fractal", "median"),
        ("residual_fractal", "gaussian"),
        ("residual_fractal", "dncnn"),
    ],
)
def test_neural_training_roundtrip_and_gradient(
    splits,
    tmp_path: Path,
    architecture: Literal["srm_fractal", "residual_fractal"],
    residual: Literal["median", "gaussian", "dncnn"],
) -> None:
    train, validation, test = splits
    config = NeuralConfig(
        architecture=architecture,
        residual=residual,
        image_size=32,
        channels=4,
        levels=2,
        backbone_blocks=1,
        dncnn_depth=3,
        dncnn_channels=4,
        epochs=2,
        batch_size=4,
        device="cpu",
        augment_probability=1.0,
        patience=1,
    )
    features = FeatureConfig(image_size=32, scales=(2, 4, 8), compression=False)
    with _threads(2):
        network = FractalNetwork(config)
        loss = network(torch.rand(2, 1, 32, 32)).square().sum()
        loss.backward()
        if residual == "dncnn":
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in network.denoiser.parameters()
            )
    events = []
    result = train_neural(
        train,
        validation,
        test,
        features,
        TrainingConfig(class_weight="balanced"),
        config,
        workers=2,
        epoch_callback=lambda m, s: events.append((m, s)),
    )
    assert len(events) >= 1
    assert result.detector.artifact.best_epoch >= 1
    assert result.evaluations["test"].samples == len(test)
    prediction = result.evaluations["test"].predictions[0]
    assert prediction.synthetic_probability == pytest.approx(
        expit(prediction.intercept + sum(c.log_odds for c in prediction.contributions)), abs=1e-6
    )
    path = tmp_path / "network.json"
    result.detector.save(path)
    loaded = NeuralDetector.load(path, device="cpu")
    assert loaded.predict(test[0].path).synthetic_probability == pytest.approx(
        prediction.synthetic_probability, abs=1e-5
    )
    with pytest.raises(ValueError, match="overlaps"):
        evaluate_neural(loaded, validation, workers=2)
    assert evaluate_neural(loaded, validation, workers=2, allow_validation=True).samples == len(
        validation
    )
    missing = Sample(path=str(tmp_path / "missing.jpg"))
    skipped = list(loaded.iter_predict([missing, test[0]], workers=2, on_error="skip"))
    assert isinstance(skipped[0], ImageFailure)
    with pytest.raises(ValueError, match="cannot measure"):
        list(loaded.iter_predict([missing], workers=2))
    with pytest.raises(ValueError, match="invalid"):
        list(loaded.iter_predict(test, batch_size=0))
    with pytest.raises(ValueError, match="checksum"):
        NeuralDetector(loaded.artifact.model_copy(update={"weights_sha256": "bad"}))
    from safetensors.torch import save

    tensors = loaded.network.state_dict()
    tensors["head.weight"][0, 0] = float("nan")
    payload = save(tensors)
    with pytest.raises(ValueError, match="nonfinite"):
        NeuralDetector(
            loaded.artifact.model_copy(
                update={
                    "weights": base64.b64encode(payload).decode(),
                    "weights_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        )


def test_split_guards_and_configuration(splits) -> None:
    train, validation, _ = splits
    features = FeatureConfig(image_size=32, scales=(2, 4, 8), compression=False)
    config = NeuralConfig(image_size=32, channels=4, levels=2, epochs=1, device="cpu")
    with pytest.raises(ValueError, match="overlap"):
        train_neural(train, train, feature_config=features, neural_config=config, workers=2)
    with pytest.raises(ValueError, match="group"):
        train_neural(
            train,
            [s.model_copy(update={"group": None}) for s in validation],
            feature_config=features,
            neural_config=config,
        )
    for options in (
        {"levels": 6, "image_size": 32},
        {"median_size": 4},
        {"architecture": "residual_fractal", "levels": 4, "image_size": 128},
        {"architecture": "residual_fractal", "levels": 3, "image_size": 48},
    ):
        with pytest.raises(ValidationError):
            NeuralConfig.model_validate(options)
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA"):
            _device(NeuralConfig(device="cuda"))
    assert _device(NeuralConfig(device="cpu")).type == "cpu"


def test_neural_runner_and_streaming(splits, tmp_path: Path) -> None:
    import csv

    data = {}
    for name, samples in zip(("train", "validation", "test"), splits, strict=True):
        path = tmp_path / f"{name}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "label", "group"])
            writer.writeheader()
            writer.writerows({"path": s.path, "label": s.label, "group": s.group} for s in samples)
        data[name] = [str(path)]
    config = {
        "task": "train",
        "data": data,
        "features": {"image_size": 32, "scales": [2, 4, 8], "compression": False},
        "neural": {
            "image_size": 32,
            "channels": 4,
            "levels": 2,
            "backbone_blocks": 1,
            "epochs": 1,
            "batch_size": 4,
            "device": "cpu",
        },
        "workers": 2,
        "output_dir": str(tmp_path / "train"),
    }
    summary = run_task(TaskConfig.model_validate_json(json.dumps(config)))
    assert summary["samples"] == sum(map(len, splits))
    model = str(tmp_path / "train" / "model.json")
    for task in ("evaluate", "predict"):
        task_config = {
            "task": task,
            "data": data,
            "split": "test",
            "model": model,
            "workers": 2,
            "neural": {"device": "cpu"},
            "output_dir": str(tmp_path / task),
        }
        assert run_task(TaskConfig.model_validate_json(json.dumps(task_config)))["samples"] == len(
            splits[2]
        )
    config["task"] = "predict"
    config["split"] = "test"
    config["model"] = model
    config["features"]["image_size"] = 64
    config["output_dir"] = str(tmp_path / "bad")
    with pytest.raises(ValueError, match="preprocessing"):
        run_task(TaskConfig.model_validate_json(json.dumps(config)))


def test_neural_auto_size_uses_training_only(splits) -> None:
    from PIL import Image

    train, validation, _ = splits
    for sample in validation:
        with Image.open(sample.path) as image:
            image.resize((96, 80)).save(sample.path)
    config = NeuralConfig(channels=4, levels=2, backbone_blocks=1, epochs=1, device="cpu")
    features = FeatureConfig(scales=(2, 4, 8), compression=False)
    result = train_neural(
        train, validation, feature_config=features, neural_config=config, workers=2
    )
    assert result.detector.artifact.neural_config.image_size == 32
    assert result.detector.artifact.feature_config.image_size == 32
    assert result.evaluations["validation"].predictions[0].measurement.processed_shape == (32, 32)


def test_neural_seed_reproducibility(splits) -> None:
    train, validation, _ = splits
    features = FeatureConfig(image_size=32, scales=(2, 4, 8), compression=False)
    config = NeuralConfig(
        image_size=32, levels=2, channels=4, backbone_blocks=1, epochs=1, device="cpu"
    )
    a = train_neural(train, validation, feature_config=features, neural_config=config, workers=1)
    b = train_neural(train, validation, feature_config=features, neural_config=config, workers=1)
    assert a.detector.artifact.weights_sha256 == b.detector.artifact.weights_sha256
    assert np.isfinite(a.detector.artifact.history[0]["validation_loss"])
