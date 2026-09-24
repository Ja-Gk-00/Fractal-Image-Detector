import json
import tomllib
from pathlib import Path

from fractal_detector.config import TaskConfig

METHODS = (
    "gray_surface",
    "residual_surface",
    "edge_box",
    "renyi",
    "lacunarity",
    "statistics",
    "spectral",
    "compression",
    "gaussian_residual",
    "median_residual",
    "pca_residual",
    "multifractal",
    "low_correlation",
    "combined",
    "srm_fractal",
    "median_fractal",
    "gaussian_fractal",
    "dncnn_fractal",
)


def method_preset(method: str, *, deep: bool = False) -> dict[str, object]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    features: dict[str, object] = {
        "scales": [2, 4, 8, 16, 32, 64] if deep else [2, 4, 8, 16, 32],
        "compression": False,
    }
    settings: dict[str, object] = {
        "features": features,
        "training": {"regularization_c": 0.1 if deep else 1.0, "seed": 1337, "max_iter": 5000},
    }
    if method.endswith("_fractal"):
        settings["neural"] = {
            "architecture": "srm_fractal" if method == "srm_fractal" else "residual_fractal",
            "residual": "median" if method == "srm_fractal" else method.removesuffix("_fractal"),
            "channels": 32,
            "levels": 3,
            "backbone_blocks": 8 if deep else 2,
            "dncnn_depth": 17,
            "dncnn_channels": 64,
            "median_size": 7,
            "gaussian_sigma": 1.5,
            "epochs": 50 if deep else 30,
            "patience": 15 if deep else 8,
            "batch_size": 16,
            "learning_rate": 0.00005,
            "device": "auto",
            "mixed_precision": False,
        }
        return settings
    if method in {"gaussian_residual", "median_residual", "pca_residual"}:
        features["residual"] = (
            "svd" if method == "pca_residual" else method.removesuffix("_residual")
        )
        features["svd_rank"] = 28 if deep else 8
        features["median_size"] = 7 if deep else 3
    elif method != "combined":
        features["families"] = [method]
    if method in {"compression", "combined"}:
        features.update(compression=True, max_domains=256 if deep else 64, stride=4 if deep else 8)
    if method in {"multifractal", "low_correlation", "combined"}:
        features.update(
            multifractal=True,
            moment_orders=[float(q) for q in range(-4, 5)]
            if deep
            else [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0],
        )
    if method in {"low_correlation", "combined"}:
        features.update(
            low_correlation=True,
            pca_ranks=[24, 26, 28, 30, 32] if deep else [28],
            residual="svd",
            svd_rank=28,
        )
    if deep and method in {"lacunarity", "low_correlation", "combined"}:
        features.update(lacunarity_mode="sliding", lacunarity_stride=1)
    return settings


def write_method_configs(
    destination: str | Path,
    *,
    train: str = "../../data-splits/train.csv",
    validation: str = "../../data-splits/validation.parquet",
    test: str | None = None,
) -> list[Path]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for method in METHODS:
        for deep in (False, True):
            name = f"{method}-{'deep' if deep else 'standard'}"
            data = {"train": [train], "validation": [validation]}
            if test is not None:
                data["test"] = [test]
            values = {
                "task": "train",
                "data": data,
                "workers": "auto",
                "batch_size": 64,
                "output_dir": f"../../outputs/methods/{name}",
                "tracking": {
                    "enabled": False,
                    "uri": "http://127.0.0.1:5000",
                    "experiment": "fractal-methods",
                    "run_name": name,
                    "image_logging": "batch",
                },
                **method_preset(method, deep=deep),
            }
            TaskConfig.model_validate_json(json.dumps(values))
            path = destination / f"{name}.toml"
            path.write_text(_toml_config(values), encoding="utf-8")
            paths.append(path)
    return paths


def _toml_config(values: dict[str, object]) -> str:
    lines = []

    def table(items: dict[str, object], prefix: str = "") -> None:
        if prefix:
            lines.extend(("", f"[{prefix}]"))
        for key, value in items.items():
            if value is not None and not isinstance(value, dict):
                lines.append(f"{key} = {json.dumps(value, ensure_ascii=False, allow_nan=False)}")
        for key, value in items.items():
            if isinstance(value, dict):
                table(value, f"{prefix}.{key}" if prefix else key)

    table(values)
    text = "\n".join(lines) + "\n"
    TaskConfig.model_validate_json(json.dumps(tomllib.loads(text)))
    return text
