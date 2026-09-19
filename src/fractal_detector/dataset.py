import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import Field

from fractal_detector.config import IMAGE_EXTENSIONS, FeatureConfig, Schema, validate_extensions
from fractal_detector.features import extract
from fractal_detector.records import FeatureResult


class Sample(Schema):
    path: str
    label: Literal[0, 1] | None = None
    group: str | None = None
    image_bytes: bytes | None = Field(default=None, exclude=True, repr=False)


class ImageFailure(Schema):
    path: str
    error_type: str
    message: str


def iter_directory(
    path: str | Path,
    *,
    recursive: bool = True,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
    exclude_dirs: tuple[Path, ...] = (),
    include_manifests: bool = False,
) -> Iterator[Sample]:
    validate_extensions(extensions)
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"input directory does not exist: {root}")
    excluded = tuple(p.resolve() for p in exclude_dirs)

    def walk(directory: Path) -> Iterator[Sample]:
        if any(directory.is_relative_to(excluded_dir) for excluded_dir in excluded):
            return
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    continue
                child = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    if recursive and not child.is_junction():
                        yield from walk(child)
                elif entry.is_file(follow_symlinks=False) and (
                    child.suffix.lower() in extensions
                    or (include_manifests and child.suffix.lower() == ".parquet")
                ):
                    yield Sample(path=str(child))

    yield from walk(root)


def measure_batch(
    samples: Sequence[Sample],
    config: FeatureConfig,
    *,
    workers: int = 1,
) -> list[FeatureResult | ImageFailure]:
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 64:
        raise ValueError("workers must be in 1..64")

    def measure(sample: Sample) -> FeatureResult | ImageFailure:
        try:
            return extract_sample(sample, config)
        except (ValueError, OSError, Image.DecompressionBombError) as exc:
            return ImageFailure(path=sample.path, error_type=type(exc).__name__, message=str(exc))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(measure, samples, buffersize=2 * workers))


def read_manifest(path: str | Path, *, require_labels: bool = False) -> list[Sample]:
    from fractal_detector.sources import iter_manifest

    samples = []
    seen: set[str] = set()
    for sample in iter_manifest(path, require_labels=require_labels):
        key = os.path.normcase(sample.path)
        if key in seen:
            raise ValueError(f"duplicate image path: {sample.path}")
        seen.add(key)
        samples.append(sample)
    return samples


def extract_sample(sample: Sample, config: FeatureConfig) -> FeatureResult:
    image = sample.path if sample.image_bytes is None else sample.image_bytes
    return extract(image, config, source=sample.path)


def extract_dataset(
    samples: Sequence[Sample], config: FeatureConfig | None = None, *, workers: int = 1
) -> list[FeatureResult]:
    if isinstance(workers, bool) or not 1 <= workers <= 64:
        raise ValueError("workers must be in 1..64")
    config = config or FeatureConfig()

    def measure(sample: Sample) -> FeatureResult:
        try:
            return extract_sample(sample, config)
        except (ValueError, OSError) as exc:
            raise ValueError(f"cannot measure {sample.path}: {exc}") from exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(measure, samples, buffersize=2 * workers))
