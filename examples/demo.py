import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from fractal_detector import Detector, FeatureConfig, TaskConfig, run_task


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline software demo using procedural textures; not a detector accuracy benchmark"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    root = args.output.resolve()
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1337)
    with (root / "train.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "group"])
        for label in (0, 1):
            for i in range(16):
                pixels = rng.random((64, 64))
                if label == 0:
                    pixels = gaussian_filter(pixels, sigma=1)
                path = images / f"texture_{label}_{i}.png"
                Image.fromarray((pixels * 255).astype(np.uint8)).save(path)
                writer.writerow([path.relative_to(root).as_posix(), label, f"pair-{i // 2}"])
    summary = run_task(
        TaskConfig(
            task="train",
            manifest="train.csv",
            output_dir="training",
            features=FeatureConfig(image_size=64, scales=(2, 4, 8, 16), max_domains=16),
        ),
        base_dir=root,
    )
    detector = Detector.load(root / "training/model.json")
    prediction = detector.predict(images / "texture_1_0.png")
    print(
        "Demo completed on procedural textures. These labels do not represent real AI provenance."
    )
    print(f"Artifacts: {summary['output_dir']}")
    print(prediction.explanation)


if __name__ == "__main__":
    main()
