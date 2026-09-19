from typing import Annotated, Literal

from pydantic import Field

from fractal_detector.config import Schema


class DimensionEstimate(Schema):
    value: float | None
    r_squared: float | None
    slope_standard_error: float | None
    scales: tuple[int, ...]
    counts: tuple[float, ...]
    reason: str | None = None


class FeatureResult(Schema):
    schema_version: Literal[1] = 1
    image_id: str
    source: str | None
    original_shape: tuple[int, int]
    processed_shape: tuple[int, int]
    backend: Literal["native", "python"]
    features: dict[str, float]
    dimensions: dict[str, DimensionEstimate]
    diagnostics: dict[str, object]
    warnings: tuple[str, ...]
    elapsed_seconds: Annotated[float, Field(ge=0)]


class Contribution(Schema):
    feature: str
    value: float
    standardized_value: float
    log_odds: float


class Prediction(Schema):
    label: Literal["real", "synthetic"]
    synthetic_probability: Annotated[float, Field(ge=0, le=1)]
    threshold: float
    intercept: float
    contributions: tuple[Contribution, ...]
    explanation: str
    measurement: FeatureResult
