from collections.abc import Sequence
from dataclasses import dataclass

from fractal_detector.config import SearchConfig, TrainingConfig
from fractal_detector.dataset import Sample, extract_dataset
from fractal_detector.model import Detector, _check_unique, _labels, evaluate_features, fit_features
from fractal_detector.parallel import Workers
from fractal_detector.sizing import resolve_dataset_features


@dataclass
class SearchTrial:
    candidate: str
    regularization_c: float
    metrics: dict[str, float]
    feature_count: int


@dataclass
class SearchResult:
    detector: Detector
    trials: tuple[SearchTrial, ...]
    best_index: int

    @property
    def report(self) -> dict[str, object]:
        return {
            "selection_metric": "validation.roc_auc",
            "best_index": self.best_index,
            "test_used": False,
            "trials": [
                {
                    "candidate": t.candidate,
                    "regularization_c": t.regularization_c,
                    "feature_count": t.feature_count,
                    "metrics": t.metrics,
                }
                for t in self.trials
            ],
            "selected_features": self.detector.artifact.feature_config.model_dump(mode="json"),
        }


def search_features(
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    config: SearchConfig,
    training_config: TrainingConfig | None = None,
    *,
    workers: Workers = "auto",
) -> SearchResult:
    training = training_config or TrainingConfig()
    _labels(train_samples)
    _labels(validation_samples)
    groups = [s.group for s in (*train_samples, *validation_samples)]
    if any(g is not None for g in groups) and any(g is None for g in groups):
        raise ValueError("group must be supplied for every sample or none")
    trials: list[SearchTrial] = []
    best: Detector | None = None
    best_key = (-float("inf"), -float("inf"), -float("inf"))
    best_index = 0
    for candidate in config.candidates:
        features = resolve_dataset_features(candidate.features, train_samples)
        train_rows = extract_dataset(train_samples, features, workers=workers)
        validation_rows = extract_dataset(validation_samples, features, workers=workers)
        _check_unique([*train_rows, *validation_rows])
        for c in config.regularization_c:
            detector = fit_features(
                train_samples,
                train_rows,
                features,
                training.model_copy(update={"regularization_c": c}),
            )
            evaluation = evaluate_features(detector, validation_samples, validation_rows)
            trial = SearchTrial(
                candidate.name, c, evaluation.metrics, len(detector.artifact.feature_names)
            )
            trials.append(trial)
            key = (trial.metrics["roc_auc"], -trial.metrics["log_loss"], -trial.feature_count)
            if key > best_key:
                best, best_key, best_index = detector, key, len(trials) - 1
                best.artifact = best.artifact.model_copy(
                    update={
                        "selection_ids": tuple(r.image_id for r in validation_rows),
                        "selection_groups": tuple(
                            sorted({s.group for s in validation_samples if s.group is not None})
                        ),
                    }
                )
    if best is None:
        raise ValueError("no search trials")
    return SearchResult(best, tuple(trials), best_index)
