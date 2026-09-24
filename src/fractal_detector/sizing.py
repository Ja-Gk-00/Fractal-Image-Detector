import logging
import math
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image

from fractal_detector.config import (
    FeatureConfig,
    NeuralConfig,
    ResolvedFeatureConfig,
    ResolvedNeuralConfig,
)

if TYPE_CHECKING:
    from fractal_detector.dataset import Sample
    from fractal_detector.io import ImageInput

logger = logging.getLogger("fractal_detector")


def image_side(image: ImageInput, max_input_pixels: int) -> int:
    if isinstance(image, (str, Path, bytes)):
        with Image.open(BytesIO(image) if isinstance(image, bytes) else image) as opened:
            height, width = opened.height, opened.width
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError("animated/multi-frame images are unsupported")
            if opened.mode in {"I", "F"} or opened.mode.startswith("I;16"):
                raise ValueError(
                    "convert high-bit-depth files to uint8 or normalized arrays explicitly"
                )
    else:
        shape = np.shape(image)
        if len(shape) not in (2, 3) or (len(shape) == 3 and shape[2] not in (3, 4)):
            raise ValueError("expected (H,W), (H,W,3), or (H,W,4)")
        height, width = shape[:2]
    if min(height, width) < 16 or height * width > max_input_pixels:
        raise ValueError("images must be at least 16 pixels per side and within max_input_pixels")
    return max(height, width)


def largest_side(
    samples: Iterable[Sample],
    max_input_pixels: int,
    *,
    on_error: Literal["raise", "skip"] = "raise",
) -> int:
    largest = 0
    for sample in samples:
        try:
            largest = max(
                largest,
                image_side(
                    sample.image_bytes if sample.image_bytes is not None else sample.path,
                    max_input_pixels,
                ),
            )
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            if on_error == "raise":
                raise ValueError(f"cannot measure {sample.path}: {exc}") from exc
    return largest


def resolve_features(config: FeatureConfig, largest: int) -> ResolvedFeatureConfig:
    if isinstance(config.image_size, int):
        return ResolvedFeatureConfig.model_validate(config.model_dump())
    step = math.lcm(*config.scales, config.block_size)
    minimum = max(
        32,
        2 * max(config.scales),
        2 * config.block_size,
        config.svd_rank + 1,
        max(config.pca_ranks) + 1 if config.low_correlation else 0,
    )
    size = math.ceil(max(largest, minimum) / step) * step
    if size * size > config.max_input_pixels:
        raise ValueError(
            "automatic square image_size exceeds max_input_pixels; set features.image_size explicitly"
        )
    resolved = ResolvedFeatureConfig.model_validate(config.model_dump() | {"image_size": size})
    logger.info(
        "image size resolved",
        extra={
            "event": "image_size_resolved",
            "data": {
                "method": "features",
                "largest_source_side": largest,
                "image_size": size,
                "alignment": step,
            },
        },
    )
    return resolved


def resolve_dataset_features(
    config: FeatureConfig, samples: Iterable[Sample]
) -> ResolvedFeatureConfig:
    side = largest_side(samples, config.max_input_pixels) if config.image_size == "auto" else 0
    return resolve_features(config, side)


def resolve_neural(
    config: NeuralConfig, largest: int, max_input_pixels: int
) -> ResolvedNeuralConfig:
    if isinstance(config.image_size, int):
        return ResolvedNeuralConfig.model_validate(config.model_dump())
    if config.architecture == "srm_fractal":
        step, minimum = 2**config.levels, 2 ** (config.levels + 1)
    else:
        step, minimum = 2 * 4 ** (config.levels - 1), 4**config.levels
    step = math.lcm(step, 8)
    size = math.ceil(max(largest, minimum, 32) / step) * step
    if size * size > max_input_pixels:
        raise ValueError(
            "automatic neural image_size exceeds max_input_pixels; set neural.image_size explicitly"
        )
    resolved = ResolvedNeuralConfig.model_validate(config.model_dump() | {"image_size": size})
    logger.info(
        "image size resolved",
        extra={
            "event": "image_size_resolved",
            "data": {
                "method": "neural",
                "largest_source_side": largest,
                "image_size": size,
                "alignment": step,
            },
        },
    )
    return resolved


@contextmanager
def auto_sized_stream(
    samples: Iterable[Sample],
    config: FeatureConfig,
    *,
    on_error: Literal["raise", "skip"] = "raise",
) -> Iterator[tuple[ResolvedFeatureConfig, Iterable[Sample]]]:
    if isinstance(config.image_size, int):
        yield resolve_features(config, 0), samples
        return
    from fractal_detector.dataset import Sample

    with tempfile.TemporaryDirectory(prefix="fractal-size-") as directory:
        connection = sqlite3.connect(Path(directory) / "samples.db")
        try:
            connection.execute(
                "CREATE TABLE samples (path TEXT, label INTEGER, source_group TEXT, image BLOB)"
            )
            largest = 0
            for sample in samples:
                largest = max(
                    largest, largest_side((sample,), config.max_input_pixels, on_error=on_error)
                )
                connection.execute(
                    "INSERT INTO samples VALUES (?, ?, ?, ?)",
                    (sample.path, sample.label, sample.group, sample.image_bytes),
                )
            connection.commit()

            def replay() -> Iterator[Sample]:
                for path, label, group, image in connection.execute(
                    "SELECT path, label, source_group, image FROM samples ORDER BY rowid"
                ):
                    yield Sample(path=path, label=label, group=group, image_bytes=image)

            yield resolve_features(config, largest), replay()
        finally:
            connection.close()
