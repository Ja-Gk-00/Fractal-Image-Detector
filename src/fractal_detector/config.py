import json
import tomllib
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class FeatureConfig(Schema):
    image_size: Annotated[int, Field(ge=32, le=32768)] | Literal["auto"] = "auto"
    scales: tuple[PositiveInt, ...] = (2, 4, 8, 16, 32)
    residual: Literal["gaussian", "svd", "median"] = "gaussian"
    gaussian_sigma: Annotated[float, Field(gt=0, le=10)] = 1.0
    median_size: Annotated[int, Field(ge=3, le=15)] = 3
    svd_rank: PositiveInt = 8
    entropy_bins: Annotated[int, Field(ge=8, le=1024)] = 64
    families: (
        tuple[
            Literal[
                "dimensions",
                "gray_surface",
                "residual_surface",
                "edge_box",
                "renyi",
                "lacunarity",
                "statistics",
                "spectral",
                "compression",
                "multifractal",
                "low_correlation",
            ],
            ...,
        ]
        | None
    ) = None
    multifractal: bool = False
    moment_orders: tuple[float, ...] = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
    mass_floor: Annotated[float, Field(ge=0, le=0.01)] = 0.0
    lacunarity_mode: Literal["tiled", "sliding"] = "tiled"
    lacunarity_stride: PositiveInt = 1
    low_correlation: bool = False
    pca_ranks: tuple[PositiveInt, ...] = (24, 28, 32)
    edge_quantile: Annotated[float, Field(gt=0, lt=1)] = 0.75
    backend: Literal["auto", "native", "python"] = "auto"
    compression: bool = True
    block_size: Annotated[int, Field(ge=2, le=32)] = 8
    stride: PositiveInt = 8
    max_domains: Annotated[int, Field(ge=1, le=1024)] = 64
    s_clip: Annotated[float, Field(gt=0, lt=1)] = 0.95
    transform_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    max_input_pixels: PositiveInt = 40_000_000

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        if self.median_size % 2 == 0:
            raise ValueError("median_size must be odd")
        if (
            len(self.moment_orders) < 3
            or tuple(sorted(set(self.moment_orders))) != self.moment_orders
        ):
            raise ValueError("moment_orders must contain at least three increasing distinct values")
        if any(abs(q) > 16 for q in self.moment_orders):
            raise ValueError("moment_orders must be within [-16, 16]")
        if self.low_correlation and (
            not self.pca_ranks
            or tuple(sorted(set(self.pca_ranks))) != self.pca_ranks
            or (isinstance(self.image_size, int) and max(self.pca_ranks) >= self.image_size)
        ):
            raise ValueError("pca_ranks must be increasing, distinct and smaller than image_size")
        if self.families is not None:
            if not self.families or len(set(self.families)) != len(self.families):
                raise ValueError("families must be nonempty and distinct")
            for family, enabled in (
                ("compression", self.compression),
                ("multifractal", self.multifractal),
                ("low_correlation", self.low_correlation),
            ):
                if family in self.families and not enabled:
                    raise ValueError(f"enable {family} before selecting its family")
        if len(self.scales) < 3 or tuple(sorted(set(self.scales))) != self.scales:
            raise ValueError("scales must contain at least three increasing, distinct sizes")
        if isinstance(self.image_size, int) and any(
            self.image_size % s or s > self.image_size // 2 for s in self.scales
        ):
            raise ValueError("scales must divide image_size and be at most image_size / 2")
        if isinstance(self.image_size, int) and (
            self.image_size % self.block_size or 2 * self.block_size > self.image_size
        ):
            raise ValueError("block_size must divide image_size and allow a double-sized domain")
        if isinstance(self.image_size, int) and self.svd_rank >= self.image_size:
            raise ValueError("svd_rank must be smaller than image_size")
        if not self.transform_ids or len(set(self.transform_ids)) != len(self.transform_ids):
            raise ValueError("transform_ids must be nonempty and distinct")
        if any(t not in range(8) for t in self.transform_ids):
            raise ValueError("transform_ids must be in 0..7 (FastFractal D4 order)")
        return self


class ResolvedFeatureConfig(FeatureConfig):
    image_size: Annotated[int, Field(ge=32, le=32768)]


class TrainingConfig(Schema):
    test_size: Annotated[float, Field(gt=0, lt=1)] = 0.25
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 1337
    regularization_c: Annotated[float, Field(gt=0)] = 1.0
    max_iter: PositiveInt = 2000
    threshold: Annotated[float, Field(gt=0, lt=1)] = 0.5
    class_weight: Literal["balanced"] | None = None


class TrackingConfig(Schema):
    enabled: bool = False
    uri: str | None = None
    experiment: str = "fractal-image-detector"
    run_name: str | None = None
    run_id: str | None = None
    image_logging: Literal["full", "batch"] = "full"


class NeuralConfig(Schema):
    architecture: Literal["srm_fractal", "residual_fractal"] = "srm_fractal"
    residual: Literal["median", "gaussian", "dncnn"] = "median"
    image_size: Annotated[int, Field(ge=32, le=32768)] | Literal["auto"] = "auto"
    channels: Annotated[int, Field(ge=4, le=128)] = 32
    levels: Annotated[int, Field(ge=1, le=6)] = 3
    backbone_blocks: Annotated[int, Field(ge=1, le=8)] = 2
    dncnn_depth: Annotated[int, Field(ge=3, le=25)] = 17
    dncnn_channels: Annotated[int, Field(ge=4, le=128)] = 64
    median_size: Annotated[int, Field(ge=3, le=15)] = 3
    gaussian_sigma: Annotated[float, Field(gt=0, le=10)] = 1.0
    dropout: Annotated[float, Field(ge=0, lt=1)] = 0.5
    epochs: Annotated[int, Field(ge=1, le=1000)] = 50
    batch_size: Annotated[int, Field(ge=2, le=512)] = 16
    learning_rate: Annotated[float, Field(gt=0, le=1)] = 0.00005
    denoiser_lr_ratio: Annotated[float, Field(gt=0, le=1)] = 0.1
    weight_decay: Annotated[float, Field(ge=0, le=1)] = 0.0001
    label_smoothing: Annotated[float, Field(ge=0, lt=0.5)] = 0.1
    patience: Annotated[int, Field(ge=1, le=1000)] = 15
    clip_norm: Annotated[float, Field(gt=0)] = 1.0
    restart_period: PositiveInt = 10
    augment_probability: Annotated[float, Field(ge=0, le=1)] = 0.1
    jpeg_min_quality: Annotated[int, Field(ge=10, le=100)] = 70
    device: Literal["auto", "cpu", "cuda"] = "auto"
    mixed_precision: bool = False

    @model_validator(mode="after")
    def geometry(self) -> Self:
        if self.median_size % 2 == 0:
            raise ValueError("median_size must be odd")
        if self.image_size == "auto":
            return self
        if self.image_size % (2**self.levels) or self.image_size // (2**self.levels) < 2:
            raise ValueError(
                "image_size must support all fractal levels with at least 2 pixels left"
            )
        if self.architecture == "residual_fractal" and self.image_size % (
            4 ** (self.levels - 1) * 2
        ):
            raise ValueError(
                "residual_fractal image_size must support quadrant splitting and pooling"
            )
        if (
            self.architecture == "residual_fractal"
            and self.image_size // 4 ** (self.levels - 1) < 4
        ):
            raise ValueError("too many residual fractal levels for image_size")
        return self


class ResolvedNeuralConfig(NeuralConfig):
    image_size: Annotated[int, Field(ge=32, le=32768)]


class DataSource(Schema):
    paths: tuple[str, ...]
    label: Literal[0, 1] | None = None
    path_column: str = "path"
    image_column: str = "image"
    label_column: str = "label"
    group_column: str = "group"
    label_map: dict[str, Literal[0, 1]] = {"0": 0, "1": 1, "real": 0, "synthetic": 1}

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if not self.paths or any(not p.strip() for p in self.paths):
            raise ValueError("source paths must not be empty")
        columns = (self.path_column, self.image_column, self.label_column, self.group_column)
        if any(not c.strip() for c in columns) or len(set(columns)) != len(columns):
            raise ValueError("source column names must be nonempty and distinct")
        if not self.label_map or any(k != k.strip().lower() for k in self.label_map):
            raise ValueError("label_map keys must be nonempty, trimmed lowercase strings")
        if "" in self.label_map:
            raise ValueError("label_map keys must not be empty")
        return self


type SourceSpec = str | DataSource


class DataSplits(Schema):
    train: tuple[SourceSpec, ...] = ()
    validation: tuple[SourceSpec, ...] = ()
    test: tuple[SourceSpec, ...] = ()

    @model_validator(mode="after")
    def validate_splits(self) -> Self:
        if not (self.train or self.validation or self.test):
            raise ValueError("data must contain at least one nonempty split")
        for sources in (self.train, self.validation, self.test):
            validate_sources(sources)
        return self


class SearchCandidate(Schema):
    name: str = Field(min_length=1, max_length=100)
    features: FeatureConfig


class SearchConfig(Schema):
    candidates: tuple[SearchCandidate, ...]
    regularization_c: tuple[Annotated[float, Field(gt=0)], ...] = (0.1, 1.0, 10.0)

    @model_validator(mode="after")
    def bounded(self) -> Self:
        if not self.candidates or not self.regularization_c:
            raise ValueError("search requires candidates and regularization values")
        if len({c.name for c in self.candidates}) != len(self.candidates):
            raise ValueError("search candidate names must be unique")
        if len(self.candidates) * len(self.regularization_c) > 100:
            raise ValueError("limit one search to at most 100 trials")
        return self


def validate_sources(sources: tuple[SourceSpec, ...]) -> None:
    if any(isinstance(s, str) and not s.strip() for s in sources):
        raise ValueError("source paths must not be empty")


class TaskConfig(Schema):
    task: Literal["extract", "train", "evaluate", "predict", "search"]
    manifest: str | None = None
    input_dir: str | None = None
    inputs: tuple[SourceSpec, ...] | None = None
    data: DataSplits | None = None
    split: Literal["train", "validation", "test"] | None = None
    recursive: bool = True
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS
    batch_size: Annotated[int, Field(ge=1, le=4096)] = 64
    on_error: Literal["raise", "skip"] = "raise"
    output_dir: str = "outputs"
    model: str | None = None
    features: FeatureConfig = FeatureConfig()
    training: TrainingConfig = TrainingConfig()
    tracking: TrackingConfig = TrackingConfig()
    neural: NeuralConfig | None = None
    search: SearchConfig | None = None
    workers: Annotated[int, Field(ge=1, le=1024)] | Literal["auto"] = 1
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def require_model(self) -> Self:
        if (self.search is not None) != (self.task == "search"):
            raise ValueError("search configuration is required only for the search task")
        if self.task == "search" and (
            self.data is None or not self.data.validation or self.neural is not None
        ):
            raise ValueError(
                "feature search requires explicit train/validation data and no neural configuration"
            )
        if self.neural is not None:
            if self.task == "extract":
                raise ValueError("neural configurations support train, predict, and evaluate")
            if self.task == "train" and (self.data is None or not self.data.validation):
                raise ValueError("neural training requires explicit train and validation splits")
        if self.task in {"predict", "evaluate"} and not self.model:
            raise ValueError("predict/evaluate require a model path")
        if sum(x is not None for x in (self.manifest, self.input_dir, self.inputs, self.data)) != 1:
            raise ValueError("provide exactly one of manifest, input_dir, inputs, or data")
        if self.manifest == "" or self.input_dir == "":
            raise ValueError("input paths must not be empty")
        if self.inputs is not None:
            if not self.inputs:
                raise ValueError("inputs must not be empty")
            validate_sources(self.inputs)
        if self.data is not None:
            if self.task in {"train", "search"}:
                if not self.data.train or self.split is not None:
                    raise ValueError("training requires data.train and no split selector")
            elif self.split is None or not getattr(self.data, self.split):
                raise ValueError(
                    "select a nonempty data split using split='train', 'validation', or 'test'"
                )
        elif self.split is not None:
            raise ValueError("split requires a data configuration")
        if self.task in {"train", "evaluate", "search"}:
            if self.input_dir is not None:
                raise ValueError("train/evaluate require a labeled manifest")
            if self.on_error != "raise":
                raise ValueError("train/evaluate require on_error='raise'")
        validate_extensions(self.extensions)
        return self


def validate_extensions(extensions: tuple[str, ...]) -> None:
    if not extensions or any(e not in IMAGE_EXTENSIONS for e in extensions):
        raise ValueError(f"extensions must be a nonempty subset of {IMAGE_EXTENSIONS}")
    if len(set(extensions)) != len(extensions):
        raise ValueError("extensions must be distinct")


def load_config(path: str | Path) -> TaskConfig:
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    match path.suffix.lower():
        case ".toml":
            data = tomllib.loads(content)
        case ".json":
            data = json.loads(content)
        case ".yaml" | ".yml":
            data = yaml.safe_load(content)
        case _:
            raise ValueError("configuration must be TOML, JSON, or YAML")
    return TaskConfig.model_validate_json(json.dumps(data, allow_nan=False))
