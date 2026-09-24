import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter

from fractal_detector import FeatureConfig, Sample


@pytest.fixture
def config() -> FeatureConfig:
    return FeatureConfig(image_size=32, scales=(2, 4, 8), block_size=4, max_domains=8)


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[list[Sample], Path]:
    rng = np.random.default_rng(1337)
    samples = []
    manifest = tmp_path / "images.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "group"])
        for label in (0, 1):
            for i in range(12):
                pixels = rng.random((32, 32))
                if label == 0:
                    pixels = gaussian_filter(pixels, 1.0)
                path = tmp_path / f"class{label}_{i}.png"
                Image.fromarray(np.round(pixels * 255).astype(np.uint8)).save(path)
                group = f"source{i // 2}"
                writer.writerow([path.name, label, group])
                samples.append(Sample(path=str(path), label=label, group=group))
    return samples, manifest
