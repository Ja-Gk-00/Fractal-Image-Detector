import argparse
import csv
import json
from pathlib import Path

from fractal_detector import TaskConfig, run_task
from fractal_detector.sources import iter_sources


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small real-data training smoke test; not an accuracy benchmark"
    )
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/neural-pilot"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    data = {}
    for split, count in (("train", 8), ("validation", 4)):
        rows = list(iter_sources((receipt["manifests"][split],), require_labels=True))
        selected = [s for label in (0, 1) for s in [r for r in rows if r.label == label][:count]]
        path = output / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "label", "group"])
            writer.writeheader()
            writer.writerows({"path": s.path, "label": s.label, "group": s.group} for s in selected)
        data[split] = [str(path)]
    summaries = {}
    for method in ("srm", "median", "gaussian", "dncnn"):
        values = {
            "task": "train",
            "data": data,
            "workers": "auto",
            "output_dir": str(output / method),
            "features": {
                "image_size": 64,
                "scales": [2, 4, 8, 16],
                "compression": False,
                "multifractal": True,
            },
            "neural": {
                "architecture": "srm_fractal" if method == "srm" else "residual_fractal",
                "residual": "median" if method == "srm" else method,
                "image_size": 64,
                "channels": 8,
                "levels": 2,
                "backbone_blocks": 1,
                "epochs": 2,
                "batch_size": 4,
                "device": "cpu",
            },
            "tracking": {
                "enabled": True,
                "uri": f"sqlite:///{(output / 'mlflow.db').as_posix()}",
                "experiment": "neural-real-data-pilot",
                "run_name": method,
                "image_logging": "batch",
            },
        }
        config_path = output / f"{method}.json"
        config_path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
        summaries[method] = run_task(TaskConfig.model_validate_json(json.dumps(values)))
        print(f"Completed {method}", flush=True)
    (output / "pilot-summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
