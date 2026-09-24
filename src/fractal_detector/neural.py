import base64
import hashlib
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from itertools import batched
from pathlib import Path
from typing import Literal, Self

import numpy as np
from PIL import Image, ImageFilter
from pydantic import Field

try:
    import torch
    from safetensors.torch import load as load_tensors
    from safetensors.torch import save as save_tensors
except ImportError as exc:
    raise ImportError(
        "Neural methods require: pip install 'fractal-image-detector[neural]'"
    ) from exc

from fractal_detector.config import (
    FeatureConfig,
    NeuralConfig,
    ResolvedFeatureConfig,
    ResolvedNeuralConfig,
    Schema,
    TrainingConfig,
)
from fractal_detector.dataset import ImageFailure, Sample, extract_dataset
from fractal_detector.features import extract
from fractal_detector.io import ImageInput, atomic_json, load_image
from fractal_detector.model import Evaluation, _check_unique, _labels
from fractal_detector.networks import FractalNetwork
from fractal_detector.parallel import Workers, resolve_workers
from fractal_detector.records import Contribution, FeatureResult, Prediction
from fractal_detector.sizing import largest_side, resolve_dataset_features, resolve_neural

logger = logging.getLogger("fractal_detector")


class NeuralArtifact(Schema):
    schema_version: Literal[2] = 2
    kind: Literal["fractal_cnn"] = "fractal_cnn"
    neural_config: ResolvedNeuralConfig
    feature_config: ResolvedFeatureConfig
    training_config: TrainingConfig
    training_ids: tuple[str, ...]
    training_groups: tuple[str, ...]
    validation_ids: tuple[str, ...]
    validation_groups: tuple[str, ...]
    weights: str = Field(repr=False)
    weights_sha256: str
    best_epoch: int
    history: tuple[dict[str, float], ...]


def _device(config: NeuralConfig) -> torch.device:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but is unavailable; install a compatible CUDA PyTorch wheel"
        )
    return torch.device("cuda" if config.device != "cpu" and torch.cuda.is_available() else "cpu")


@contextmanager
def _threads(workers: Workers) -> Iterator[int]:
    count, previous = resolve_workers(workers), torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield count
    finally:
        torch.set_num_threads(previous)


def _image_config(config: ResolvedNeuralConfig, features: FeatureConfig) -> FeatureConfig:
    return FeatureConfig(
        image_size=config.image_size,
        scales=(2, 4, 8),
        compression=False,
        max_input_pixels=features.max_input_pixels,
    )


def _batch(
    samples: Sequence[Sample],
    config: ResolvedNeuralConfig,
    features: FeatureConfig,
    pool: ThreadPoolExecutor,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    image_config = _image_config(config, features)

    def read(sample: Sample) -> np.ndarray:
        gray = load_image(
            sample.image_bytes if sample.image_bytes is not None else sample.path, image_config
        )[0]
        return gray.astype(np.float32)

    images = list(pool.map(read, samples))
    if rng is not None and config.augment_probability:
        for index, gray in enumerate(images):
            im = Image.fromarray(np.round(gray * 255).astype(np.uint8))
            if rng.random() < config.augment_probability:
                buffer = BytesIO()
                im.save(
                    buffer, format="JPEG", quality=int(rng.integers(config.jpeg_min_quality, 101))
                )
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    im = decoded.copy()
            if rng.random() < config.augment_probability:
                im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.1, 1.0))))
            if rng.random() < config.augment_probability:
                side = config.image_size
                im = im.resize((side // 2, side // 2), Image.Resampling.BILINEAR).resize(
                    (side, side), Image.Resampling.BILINEAR
                )
            images[index] = np.asarray(im, dtype=np.float32) / 255
    return torch.from_numpy(np.stack(images)[:, None])


class NeuralDetector:
    def __init__(
        self, artifact: NeuralArtifact, *, device: Literal["auto", "cpu", "cuda"] = "auto"
    ) -> None:
        self.artifact = artifact
        self.device = _device(artifact.neural_config.model_copy(update={"device": device}))
        payload = base64.b64decode(artifact.weights, validate=True)
        if hashlib.sha256(payload).hexdigest() != artifact.weights_sha256:
            raise ValueError("neural weights checksum does not match")
        with torch.random.fork_rng(devices=[]):
            self.network = FractalNetwork(artifact.neural_config)
        self.network.load_state_dict(load_tensors(payload), strict=True)
        if any(not torch.isfinite(t).all() for t in self.network.state_dict().values()):
            raise ValueError("neural weights contain nonfinite values")
        self.network.to(self.device).eval()

    @classmethod
    def load(cls, path: str | Path, *, device: Literal["auto", "cpu", "cuda"] = "auto") -> Self:
        return cls(
            NeuralArtifact.model_validate_json(Path(path).read_text(encoding="utf-8")),
            device=device,
        )

    def save(self, path: str | Path) -> None:
        atomic_json(path, self.artifact.model_dump(mode="json"))

    def _predictions(
        self, images: torch.Tensor, measurements: Sequence[FeatureResult]
    ) -> list[Prediction]:
        with torch.inference_mode():
            hidden = self.network.encode(images.to(self.device)).float()
            logits = self.network.head(hidden).squeeze(-1)
            probabilities = logits.sigmoid().cpu().numpy()
            terms = (hidden * self.network.head.weight).cpu().numpy()
            values = hidden.cpu().numpy()
            intercept = float(self.network.head.bias.item())
        if not np.isfinite(probabilities).all() or not np.isfinite(terms).all():
            raise ValueError("nonfinite network prediction")
        results = []
        for i, measurement in enumerate(measurements):
            contributions = tuple(
                sorted(
                    (
                        Contribution(
                            feature=f"neural_hidden_{j}",
                            value=float(v),
                            standardized_value=float(v),
                            log_odds=float(t),
                        )
                        for j, (v, t) in enumerate(zip(values[i], terms[i], strict=True))
                    ),
                    key=lambda c: abs(c.log_odds),
                    reverse=True,
                )
            )
            threshold = self.artifact.training_config.threshold
            label = "synthetic" if probabilities[i] >= threshold else "real"
            dimension = measurement.dimensions["gray"].value
            prediction = Prediction(
                label=label,
                synthetic_probability=float(probabilities[i]),
                threshold=threshold,
                intercept=intercept,
                contributions=contributions,
                measurement=measurement,
                explanation=f"{self.artifact.neural_config.architecture} classified as {label}. Grayscale surface dimension: {dimension}. Contributions exactly decompose the final neural head log-odds; hidden activations are not measured fractal dimensions or causal pixel explanations. Probabilities are not independently calibrated.",
            )
            logger.info(
                "neural image classified",
                extra={"event": "prediction", "data": prediction.model_dump(mode="json")},
            )
            results.append(prediction)
        return results

    def predict(self, image: ImageInput) -> Prediction:
        measurement = extract(image, self.artifact.feature_config)
        gray = load_image(
            image, _image_config(self.artifact.neural_config, self.artifact.feature_config)
        )[0]
        tensor = torch.from_numpy(gray.astype(np.float32)[None, None])
        return self._predictions(tensor, [measurement])[0]

    def iter_predict(
        self,
        samples: Iterable[Sample],
        *,
        workers: Workers = "auto",
        batch_size: int | None = None,
        on_error: Literal["raise", "skip"] = "raise",
    ) -> Iterator[Prediction | ImageFailure]:
        size = self.artifact.neural_config.batch_size if batch_size is None else batch_size
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 1 <= size <= 4096
            or on_error not in {"raise", "skip"}
        ):
            raise ValueError("invalid batch_size or on_error")
        with _threads(workers) as count, ThreadPoolExecutor(max_workers=count) as pool:
            for batch in batched(samples, size, strict=False):
                from fractal_detector.dataset import measure_batch

                measured = measure_batch(batch, self.artifact.feature_config, workers=count)
                valid = [
                    (s, r)
                    for s, r in zip(batch, measured, strict=True)
                    if isinstance(r, FeatureResult)
                ]
                failures = [r for r in measured if isinstance(r, ImageFailure)]
                if failures and on_error == "raise":
                    raise ValueError(f"cannot measure {failures[0].path}: {failures[0].message}")
                predicted = iter(
                    self._predictions(
                        _batch(
                            [s for s, _ in valid],
                            self.artifact.neural_config,
                            self.artifact.feature_config,
                            pool,
                        ),
                        [r for _, r in valid],
                    )
                    if valid
                    else []
                )
                for result in measured:
                    yield result if isinstance(result, ImageFailure) else next(predicted)

    def predict_many(
        self, samples: Sequence[Sample], *, workers: Workers = "auto"
    ) -> list[Prediction]:
        return [p for p in self.iter_predict(samples, workers=workers) if isinstance(p, Prediction)]


@dataclass
class NeuralTrainingResult:
    detector: NeuralDetector
    evaluations: dict[str, Evaluation]
    measurements: dict[str, tuple[FeatureResult, ...]]


def _report(samples: Sequence[Sample], predictions: Sequence[Prediction]) -> Evaluation:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        log_loss,
        roc_auc_score,
    )

    labels = _labels(samples)
    p = np.asarray([r.synthetic_probability for r in predictions])
    y = p >= predictions[0].threshold
    cm = confusion_matrix(labels, y, labels=[0, 1])
    return Evaluation(
        metrics={
            "accuracy": float(accuracy_score(labels, y)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, y)),
            "roc_auc": float(roc_auc_score(labels, p)),
            "average_precision": float(average_precision_score(labels, p)),
            "log_loss": float(log_loss(labels, np.column_stack((1 - p, p)), labels=[0, 1])),
            "brier_score": float(brier_score_loss(labels, p)),
        },
        confusion_matrix=((int(cm[0, 0]), int(cm[0, 1])), (int(cm[1, 0]), int(cm[1, 1]))),
        samples=len(samples),
        predictions=tuple(predictions),
    )


def evaluate_neural(
    detector: NeuralDetector,
    samples: Sequence[Sample],
    *,
    workers: Workers = "auto",
    allow_validation: bool = False,
) -> Evaluation:
    _labels(samples)
    predictions = detector.predict_many(samples, workers=workers)
    measurements = [p.measurement for p in predictions]
    _check_unique(measurements)
    ids = set(detector.artifact.training_ids)
    groups = set(detector.artifact.training_groups)
    if not allow_validation:
        ids.update(detector.artifact.validation_ids)
        groups.update(detector.artifact.validation_groups)
    if ids & {r.image_id for r in measurements} or groups & {
        s.group for s in samples if s.group is not None
    }:
        raise ValueError("evaluation overlaps fitting or model-selection data")
    return _report(samples, predictions)


def train_neural(
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    test_samples: Sequence[Sample] | None = None,
    feature_config: FeatureConfig | None = None,
    training_config: TrainingConfig | None = None,
    neural_config: NeuralConfig | None = None,
    *,
    workers: Workers = "auto",
    epoch_callback: Callable[[dict[str, float], int], None] | None = None,
) -> NeuralTrainingResult:
    features, training, config = (
        feature_config or FeatureConfig(compression=False),
        training_config or TrainingConfig(),
        neural_config or NeuralConfig(),
    )
    features = resolve_dataset_features(features, train_samples)
    config = resolve_neural(
        config,
        largest_side(train_samples, features.max_input_pixels)
        if config.image_size == "auto"
        else 0,
        features.max_input_pixels,
    )
    splits = {"train": train_samples, "validation": validation_samples}
    if test_samples is not None:
        splits["test"] = test_samples
    measurements, ids, groups = {}, set(), set()
    all_groups = [s.group for rows in splits.values() for s in rows]
    if any(g is not None for g in all_groups) and any(g is None for g in all_groups):
        raise ValueError("group must be supplied for every sample or none")
    for name, samples in splits.items():
        _labels(samples)
        rows = tuple(extract_dataset(samples, features, workers=workers))
        _check_unique(rows)
        current_ids = {r.image_id for r in rows}
        current_groups = {s.group for s in samples if s.group is not None}
        if ids & current_ids or groups & current_groups:
            raise ValueError("neural splits overlap by decoded image or group")
        ids.update(current_ids)
        groups.update(current_groups)
        measurements[name] = rows
    device = _device(config)
    with (
        _threads(workers) as count,
        torch.random.fork_rng(
            devices=[torch.cuda.current_device()] if device.type == "cuda" else []
        ),
        ThreadPoolExecutor(max_workers=count) as pool,
    ):
        torch.manual_seed(training.seed)
        network = FractalNetwork(config).to(device)
        history, weights, best_epoch = _fit(
            network, splits, features, training, config, pool, device, epoch_callback
        )
    payload = save_tensors(weights)
    artifact = NeuralArtifact(
        neural_config=config,
        feature_config=features,
        training_config=training,
        training_ids=tuple(r.image_id for r in measurements["train"]),
        training_groups=tuple(sorted({s.group for s in train_samples if s.group is not None})),
        validation_ids=tuple(r.image_id for r in measurements["validation"]),
        validation_groups=tuple(
            sorted({s.group for s in validation_samples if s.group is not None})
        ),
        weights=base64.b64encode(payload).decode("ascii"),
        weights_sha256=hashlib.sha256(payload).hexdigest(),
        best_epoch=best_epoch,
        history=tuple(history),
    )
    detector = NeuralDetector(artifact, device=config.device)
    reports = {
        name: evaluate_neural(
            detector, samples, workers=workers, allow_validation=name == "validation"
        )
        for name, samples in splits.items()
        if name != "train"
    }
    return NeuralTrainingResult(detector, reports, measurements)


def _fit(
    network: FractalNetwork,
    splits: dict[str, Sequence[Sample]],
    features: FeatureConfig,
    training: TrainingConfig,
    config: ResolvedNeuralConfig,
    pool: ThreadPoolExecutor,
    device: torch.device,
    callback: Callable[[dict[str, float], int], None] | None,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor], int]:
    main, denoiser = [], []
    for name, parameter in network.named_parameters():
        (denoiser if name.startswith("denoiser.") else main).append(parameter)
    parameters = [{"params": main, "lr": config.learning_rate}]
    if denoiser:
        parameters.append(
            {"params": denoiser, "lr": config.learning_rate * config.denoiser_lr_ratio}
        )
    optimizer = torch.optim.Adam(parameters, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.restart_period,
        T_mult=2,
        eta_min=min(1e-7, config.learning_rate * config.denoiser_lr_ratio),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
    rng = np.random.default_rng(training.seed)
    best_loss, best_epoch, history, best = float("inf"), 0, [], {}
    counts = np.bincount(_labels(splits["train"]), minlength=2)
    for epoch in range(config.epochs):
        network.train()
        total = 0.0
        order = rng.permutation(len(splits["train"]))
        for indices in batched(order.tolist(), config.batch_size, strict=False):
            samples = [splits["train"][i] for i in indices]
            images = _batch(samples, config, features, pool, rng).to(device)
            targets = torch.tensor([s.label for s in samples], device=device, dtype=torch.float32)
            smoothed = targets * (1 - config.label_smoothing) + 0.5 * config.label_smoothing
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = network(images)
                losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), smoothed, reduction="none"
                )
                if training.class_weight == "balanced":
                    sample_weights = torch.tensor(
                        [len(splits["train"]) / (2 * counts[int(s.label)]) for s in samples],
                        device=device,
                    )
                    losses = losses * sample_weights
                loss = losses.mean()
            if not torch.isfinite(loss):
                raise ValueError(
                    "nonfinite training loss; reduce learning rate or disable mixed precision"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                network.parameters(), config.clip_norm, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(samples)
        network.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for samples in batched(splits["validation"], config.batch_size, strict=False):
                logits = network(_batch(samples, config, features, pool).to(device))
                target = torch.tensor(
                    [s.label for s in samples], device=device, dtype=torch.float32
                )
                validation_loss += float(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, target, reduction="sum"
                    )
                )
        validation_loss /= len(splits["validation"])
        if not np.isfinite(validation_loss):
            raise ValueError("nonfinite validation loss")
        metrics = {
            "train_loss": total / len(splits["train"]),
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(metrics)
        logger.info(
            "neural epoch completed",
            extra={"event": "neural_epoch", "data": {"epoch": epoch + 1, **metrics}},
        )
        if callback is not None:
            callback(metrics, epoch + 1)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch + 1
            best = {
                name: value.detach().cpu().clone().contiguous()
                for name, value in network.state_dict().items()
            }
        elif epoch + 1 - best_epoch >= config.patience:
            break
        scheduler.step(epoch + 1)
    return history, best, best_epoch
