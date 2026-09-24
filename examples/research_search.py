import argparse
import json
from pathlib import Path

from fractal_detector import run_task
from fractal_detector.config import FeatureConfig, SearchCandidate, SearchConfig, TaskConfig


def candidates() -> tuple[SearchCandidate, ...]:
    variants: list[tuple[str, dict[str, object]]] = [
        ("legacy", {}),
        ("gaussian_multifractal", {"multifractal": True, "compression": False}),
        ("gaussian_sigma_0_6", {"multifractal": True, "compression": False, "gaussian_sigma": 0.6}),
        ("gaussian_sigma_1_5", {"multifractal": True, "compression": False, "gaussian_sigma": 1.5}),
        (
            "median_multifractal",
            {"multifractal": True, "compression": False, "residual": "median", "median_size": 3},
        ),
        (
            "pca_rank_8",
            {"multifractal": True, "compression": False, "residual": "svd", "svd_rank": 8},
        ),
        (
            "pca_rank_28",
            {"multifractal": True, "compression": False, "residual": "svd", "svd_rank": 28},
        ),
        (
            "low_correlation",
            {"low_correlation": True, "compression": False, "families": ["low_correlation"]},
        ),
        (
            "gaussian_256",
            {
                "image_size": 256,
                "scales": [2, 4, 8, 16, 32, 64],
                "multifractal": True,
                "compression": False,
            },
        ),
        (
            "combined_256",
            {
                "image_size": 256,
                "scales": [2, 4, 8, 16, 32, 64],
                "multifractal": True,
                "low_correlation": True,
                "compression": False,
                "lacunarity_mode": "sliding",
                "residual": "svd",
                "svd_rank": 28,
            },
        ),
    ]
    return tuple(
        SearchCandidate(
            name=name,
            features=FeatureConfig.model_validate_json(json.dumps({"image_size": 128, **values})),
        )
        for name, values in variants
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 30 prespecified trials using train/validation only"
    )
    parser.add_argument(
        "receipt", type=Path, help="preparation.json produced by the demonstration notebook"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/research-search"))
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    search = SearchConfig(candidates=candidates(), regularization_c=(0.1, 1.0, 10.0))
    config = TaskConfig.model_validate_json(
        json.dumps(
            {
                "task": "search",
                "data": {k: [receipt["manifests"][k]] for k in ("train", "validation")},
                "search": search.model_dump(mode="json"),
                "training": {"max_iter": 5000},
                "workers": "auto",
                "output_dir": str(args.output.resolve()),
            }
        )
    )
    summary = run_task(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
