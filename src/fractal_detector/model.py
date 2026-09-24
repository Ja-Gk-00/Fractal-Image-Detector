import hashlib
import json
import logging
import warnings
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from itertools import batched
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import Field, model_validator
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from fractal_detector.config import (
    IMAGE_EXTENSIONS,
    FeatureConfig,
    ResolvedFeatureConfig,
    Schema,
    SourceSpec,
    TrainingConfig,
)
from fractal_detector.dataset import (
    ImageFailure,
    Sample,
    extract_dataset,
    iter_directory,
    measure_batch,
)
from fractal_detector.features import extract, feature_names
from fractal_detector.io import FloatArray, ImageInput, atomic_json
from fractal_detector.parallel import Workers
from fractal_detector.records import Contribution, FeatureResult, Prediction
from fractal_detector.sizing import resolve_dataset_features, resolve_features

logger = logging.getLogger("fractal_detector")


class ModelArtifact(Schema):
    schema_version: Literal[1] = 1
    package_version: Literal["0.1.0"] = "0.1.0"
    feature_config: ResolvedFeatureConfig
    training_config: TrainingConfig
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[Annotated[float, Field(gt=0)], ...]
    coefficients: tuple[float, ...]
    intercept: float
    threshold: Annotated[float, Field(gt=0, lt=1)]
    training_ids: tuple[str, ...]
    training_groups: tuple[str, ...]
    selection_ids: tuple[str, ...] = ()
    selection_groups: tuple[str, ...] = ()
    training_digest: str
    created_at: str

    @model_validator(mode="after")
    def check_schema(self) -> Self:
        if self.feature_names != feature_names(self.feature_config):
            raise ValueError("model feature schema does not match this package")
        if any(
            len(x) != len(self.feature_names) for x in (self.means, self.scales, self.coefficients)
        ):
            raise ValueError("model weight dimensions do not match the feature schema")
        return self


def _matrix(results: Sequence[FeatureResult], config: FeatureConfig) -> FloatArray:
    if not results:
        raise ValueError("at least one image is required")
    config = resolve_features(config, results[0].processed_shape[0])
    names = feature_names(config)
    expected = config.model_dump(mode="json", exclude={"backend", "max_input_pixels"})
    for result in results:
        if tuple(result.features) != names:
            raise ValueError("feature schema mismatch")
        stored = result.diagnostics.get("config")
        if not isinstance(stored, dict):
            raise ValueError("missing feature extraction configuration")
        actual = FeatureConfig.model_validate_json(json.dumps(stored)).model_dump(
            mode="json", exclude={"backend", "max_input_pixels"}
        )
        if actual != expected:
            raise ValueError("feature extraction configuration differs from the model")
    return np.asarray(
        [[result.features[name] for name in names] for result in results], dtype=np.float64
    )


def _labels(samples: Sequence[Sample]) -> np.ndarray:
    if any(s.label is None for s in samples) or {s.label for s in samples} != {0, 1}:
        raise ValueError("training/evaluation require both real (0) and synthetic (1) labels")
    return np.asarray([s.label for s in samples], dtype=np.int64)


class Detector:
    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls(ModelArtifact.model_validate_json(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        atomic_json(path, self.artifact.model_dump(mode="json"))

    def predict_features(self, result: FeatureResult) -> Prediction:
        artifact = self.artifact
        row = _matrix([result], artifact.feature_config)[0]
        standardized = (row - np.asarray(artifact.means)) / np.asarray(artifact.scales)
        terms = standardized * np.asarray(artifact.coefficients)
        probability = float(expit(artifact.intercept + terms.sum()))
        contributions = tuple(
            sorted(
                (
                    Contribution(
                        feature=name,
                        value=float(value),
                        standardized_value=float(z),
                        log_odds=float(term),
                    )
                    for name, value, z, term in zip(
                        artifact.feature_names, row, standardized, terms, strict=True
                    )
                ),
                key=lambda c: abs(c.log_odds),
                reverse=True,
            )
        )
        dimension = result.dimensions["gray"].value
        leaders = "; ".join(f"{c.feature} {c.log_odds:+.3f}" for c in contributions[:3])
        label = "synthetic" if probability >= artifact.threshold else "real"
        explanation = (
            f"Classified as {label} at threshold {artifact.threshold:.3f}. "
            f"Grayscale surface fractal dimension: {dimension:.4f}. "
            f"Largest log-odds contributions: {leaders}. Positive terms favor synthetic. "
            "This is a learned statistical estimate, not proof of image provenance; "
            "probabilities are not independently calibrated."
        )
        prediction = Prediction(
            label=label,
            synthetic_probability=probability,
            threshold=artifact.threshold,
            intercept=artifact.intercept,
            contributions=contributions,
            explanation=explanation,
            measurement=result,
        )
        logger.info(
            "image classified",
            extra={"event": "prediction", "data": prediction.model_dump(mode="json")},
        )
        return prediction

    def predict(self, image: ImageInput) -> Prediction:
        return self.predict_features(extract(image, self.artifact.feature_config))

    def predict_many(self, samples: Sequence[Sample], *, workers: Workers = 1) -> list[Prediction]:
        return [
            self.predict_features(r)
            for r in extract_dataset(samples, self.artifact.feature_config, workers=workers)
        ]

    def iter_predict(
        self,
        samples: Iterable[Sample],
        *,
        workers: Workers = 1,
        batch_size: int = 64,
        on_error: Literal["raise", "skip"] = "raise",
    ) -> Iterator[Prediction | ImageFailure]:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= 4096
        ):
            raise ValueError("batch_size must be in 1..4096")
        if on_error not in {"raise", "skip"}:
            raise ValueError("on_error must be raise or skip")
        for batch in batched(samples, batch_size, strict=False):
            for result in measure_batch(batch, self.artifact.feature_config, workers=workers):
                if isinstance(result, ImageFailure):
                    if on_error == "raise":
                        raise ValueError(f"cannot measure {result.path}: {result.message}")
                    yield result
                else:
                    yield self.predict_features(result)

    def predict_directory(
        self,
        path: str | Path,
        *,
        recursive: bool = True,
        extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
        workers: Workers = 1,
        batch_size: int = 64,
        on_error: Literal["raise", "skip"] = "raise",
    ) -> Iterator[Prediction | ImageFailure]:
        return self.iter_predict(
            iter_directory(path, recursive=recursive, extensions=extensions),
            workers=workers,
            batch_size=batch_size,
            on_error=on_error,
        )

    def predict_sources(
        self,
        sources: Sequence[SourceSpec | Path],
        *,
        base_dir: str | Path = ".",
        recursive: bool = True,
        extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
        workers: Workers = 1,
        batch_size: int = 64,
        on_error: Literal["raise", "skip"] = "raise",
    ) -> Iterator[Prediction | ImageFailure]:
        from fractal_detector.sources import iter_sources

        return self.iter_predict(
            iter_sources(
                sources,
                base_dir=base_dir,
                recursive=recursive,
                extensions=extensions,
                batch_size=batch_size,
            ),
            workers=workers,
            batch_size=batch_size,
            on_error=on_error,
        )


class Evaluation(Schema):
    metrics: dict[str, float]
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    samples: int
    predictions: tuple[Prediction, ...]


def evaluate_features(
    detector: Detector, samples: Sequence[Sample], results: Sequence[FeatureResult]
) -> Evaluation:
    if len(samples) != len(results):
        raise ValueError("sample and feature lengths differ")
    labels = _labels(samples)
    _check_unique(results)
    if set(detector.artifact.training_ids + detector.artifact.selection_ids) & {
        r.image_id for r in results
    }:
        raise ValueError("evaluation contains an image used for fitting")
    if set(detector.artifact.training_groups + detector.artifact.selection_groups) & {
        s.group for s in samples if s.group is not None
    }:
        raise ValueError("evaluation contains a group used for fitting")
    predictions = tuple(detector.predict_features(result) for result in results)
    p = np.asarray([pred.synthetic_probability for pred in predictions])
    predicted = p >= detector.artifact.threshold
    metrics = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, p)),
        "average_precision": float(average_precision_score(labels, p)),
        "log_loss": float(log_loss(labels, np.column_stack((1 - p, p)), labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, p)),
    }
    cm = confusion_matrix(labels, predicted, labels=[0, 1])
    return Evaluation(
        metrics=metrics,
        confusion_matrix=((int(cm[0, 0]), int(cm[0, 1])), (int(cm[1, 0]), int(cm[1, 1]))),
        samples=len(samples),
        predictions=predictions,
    )


def evaluate(detector: Detector, samples: Sequence[Sample], *, workers: Workers = 1) -> Evaluation:
    return evaluate_features(
        detector,
        samples,
        extract_dataset(samples, detector.artifact.feature_config, workers=workers),
    )


def _check_unique(results: Sequence[FeatureResult]) -> None:
    if len({r.image_id for r in results}) != len(results):
        raise ValueError(
            "duplicate decoded images detected; remove copies before splitting or evaluating"
        )


def fit_features(
    samples: Sequence[Sample],
    results: Sequence[FeatureResult],
    feature_config: FeatureConfig,
    training_config: TrainingConfig | None = None,
) -> Detector:
    config = training_config or TrainingConfig()
    if len(samples) != len(results):
        raise ValueError("sample and feature lengths differ")
    feature_config = resolve_features(
        feature_config, results[0].processed_shape[0] if results else 0
    )
    labels = _labels(samples)
    _check_unique(results)
    matrix = _matrix(results, feature_config)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix)
    estimator = LogisticRegression(
        C=config.regularization_c,
        max_iter=config.max_iter,
        random_state=config.seed,
        class_weight=config.class_weight,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(standardized, labels)
    provenance = sorted(
        (r.image_id, s.label, s.group) for s, r in zip(samples, results, strict=True)
    )
    digest = hashlib.sha256(json.dumps(provenance).encode()).hexdigest()
    return Detector(
        ModelArtifact(
            feature_config=feature_config,
            training_config=config,
            feature_names=feature_names(feature_config),
            means=tuple(np.asarray(scaler.mean_).tolist()),
            scales=tuple(np.asarray(scaler.scale_).tolist()),
            coefficients=tuple(estimator.coef_[0]),
            intercept=float(estimator.intercept_[0]),
            threshold=config.threshold,
            training_ids=tuple(r.image_id for r in results),
            training_groups=tuple(sorted({s.group for s in samples if s.group is not None})),
            training_digest=digest,
            created_at=datetime.now(UTC).isoformat(),
        )
    )


class TrainingResult:
    def __init__(
        self,
        detector: Detector,
        evaluation: Evaluation,
        train_indices: tuple[int, ...],
        test_indices: tuple[int, ...],
        measurements: tuple[FeatureResult, ...],
    ) -> None:
        self.detector = detector
        self.evaluation = evaluation
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.measurements = measurements


class SplitTrainingResult:
    def __init__(
        self,
        detector: Detector,
        evaluations: dict[str, Evaluation],
        measurements: dict[str, tuple[FeatureResult, ...]],
    ) -> None:
        self.detector = detector
        self.evaluations = evaluations
        self.measurements = measurements


def train_splits(
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample] | None = None,
    test_samples: Sequence[Sample] | None = None,
    feature_config: FeatureConfig | None = None,
    training_config: TrainingConfig | None = None,
    *,
    workers: Workers = 1,
) -> SplitTrainingResult:
    config = feature_config or FeatureConfig()
    config = resolve_dataset_features(config, train_samples)
    splits = {"train": train_samples}
    if validation_samples is not None:
        splits["validation"] = validation_samples
    if test_samples is not None:
        splits["test"] = test_samples
    for samples in splits.values():
        _labels(samples)
    groups = [sample.group for samples in splits.values() for sample in samples]
    if any(g is not None for g in groups) and any(g is None for g in groups):
        raise ValueError("group must be supplied for every sample in every split or none")
    measurements: dict[str, tuple[FeatureResult, ...]] = {}
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    for name, samples in splits.items():
        current_groups = {s.group for s in samples if s.group is not None}
        if seen_groups & current_groups:
            raise ValueError(f"{name} split contains a group from another split")
        results = tuple(extract_dataset(samples, config, workers=workers))
        _check_unique(results)
        ids = {r.image_id for r in results}
        if seen_ids & ids:
            raise ValueError(f"{name} split contains an image from another split")
        seen_groups.update(current_groups)
        seen_ids.update(ids)
        measurements[name] = results
    detector = fit_features(train_samples, measurements["train"], config, training_config)
    evaluations = {
        name: evaluate_features(detector, samples, measurements[name])
        for name, samples in splits.items()
        if name != "train"
    }
    logger.info(
        "training completed",
        extra={
            "event": "training",
            "data": {
                "split_mode": "explicit",
                "samples": {k: len(v) for k, v in splits.items()},
                "metrics": {k: v.metrics for k, v in evaluations.items()},
                "training_digest": detector.artifact.training_digest,
            },
        },
    )
    return SplitTrainingResult(detector, evaluations, measurements)


def train(
    samples: Sequence[Sample],
    feature_config: FeatureConfig | None = None,
    training_config: TrainingConfig | None = None,
    *,
    workers: Workers = 1,
) -> TrainingResult:
    features = feature_config or FeatureConfig()
    config = training_config or TrainingConfig()
    labels = _labels(samples)
    if min(np.bincount(labels)) < 2:
        raise ValueError("at least two distinct images per class are required for a holdout")
    indices = np.arange(len(samples))
    groups = [s.group for s in samples]
    if any(group is not None for group in groups):
        if any(group is None for group in groups):
            raise ValueError("group must be supplied for every sample or none")
        splitter = GroupShuffleSplit(
            n_splits=64, test_size=config.test_size, random_state=config.seed
        )
        split = next(
            (
                (a, b)
                for a, b in splitter.split(indices, labels, groups)
                if len(np.unique(labels[a])) == 2 and len(np.unique(labels[b])) == 2
            ),
            None,
        )
        if split is None:
            raise ValueError(
                "cannot create a group-disjoint split with both classes; revise groups or test_size"
            )
        train_idx, test_idx = split
    else:
        train_idx, test_idx = train_test_split(
            indices, test_size=config.test_size, random_state=config.seed, stratify=labels
        )
    features = resolve_dataset_features(features, [samples[i] for i in train_idx])
    results = extract_dataset(samples, features, workers=workers)
    _check_unique(results)
    fitted = fit_features(
        [samples[i] for i in train_idx], [results[i] for i in train_idx], features, config
    )
    evaluation = evaluate_features(
        fitted, [samples[i] for i in test_idx], [results[i] for i in test_idx]
    )
    logger.info(
        "training completed",
        extra={
            "event": "training",
            "data": {
                "metrics": evaluation.metrics,
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
                "training_digest": fitted.artifact.training_digest,
            },
        },
    )
    return TrainingResult(
        fitted,
        evaluation,
        tuple(int(i) for i in train_idx),
        tuple(int(i) for i in test_idx),
        tuple(results),
    )
