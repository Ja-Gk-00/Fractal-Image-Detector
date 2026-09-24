import argparse
import platform
import statistics
from time import perf_counter
from typing import Literal

import numpy as np

from fractal_detector import FeatureConfig, extract, native_available
from fractal_detector.io import atomic_json
from fractal_detector.kernels import affine_match, tile_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare equivalent C and NumPy kernels on reproducible inputs"
    )
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--output", default="outputs/benchmark.json")
    args = parser.parse_args()
    if args.repeat < 2:
        parser.error("--repeat must be at least 2")
    rng = np.random.default_rng(1337)
    image = rng.random((128, 128))
    ranges = rng.random((256, 64))
    domains = rng.random((512, 64))
    backends: list[Literal["python", "native"]] = (
        ["python", "native"] if native_available() else ["python"]
    )
    timings = {}
    for backend in backends:
        tasks = {
            "tile_stats": lambda b=backend: tile_stats(image, 4, b),
            "affine_match": lambda b=backend: affine_match(ranges, domains, 0.95, b),
            "extract": lambda b=backend: extract(image, FeatureConfig(backend=b)),
        }
        for name, task in tasks.items():
            task()
            measurements = []
            for _ in range(args.repeat):
                start = perf_counter()
                task()
                measurements.append(perf_counter() - start)
            timings[f"{backend}.{name}"] = {
                "median_seconds": statistics.median(measurements),
                "min_seconds": min(measurements),
                "repetitions": args.repeat,
            }
    if native_available():
        np.testing.assert_allclose(
            affine_match(ranges, domains, 0.95, "python"),
            affine_match(ranges, domains, 0.95, "native"),
            atol=1e-12,
        )
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": 1337,
        "timings": timings,
    }
    atomic_json(args.output, report)
    for name, values in timings.items():
        print(f"{name}: {1000 * values['median_seconds']:.3f} ms")


if __name__ == "__main__":
    main()
