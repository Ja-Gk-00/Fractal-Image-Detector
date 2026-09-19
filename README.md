# Fractal Image Detector

A Python API and CLI for extracting fractal image measurements, training a supervised
synthetic-image detector, and explaining its predictions. Python 3.14; optional C acceleration.

## Quick start

```console
uv sync --locked --all-extras --group docs
uv run fractal-detector doctor
uv run python examples/demo.py
```

The offline demo exercises the entire workflow using procedural textures. Train on
your own labeled real/synthetic images for actual detection; no pretrained model or
real-world accuracy claim is included.

```python
from fractal_detector import extract, read_manifest, train

measurement = extract("photo.jpg")
print(measurement.dimensions["gray"].value)

trained = train(read_manifest("train.csv", require_labels=True))
trained.detector.save("model.json")
prediction = trained.detector.predict("unknown.png")
print(prediction.synthetic_probability, prediction.explanation)
```

Manifest columns are `path,label,group`; labels are `real`/`0` and `synthetic`/`1`,
and groups are optional. Related images should share a group to prevent split leakage.

## Included

- Box-counting and differential fractal dimensions, Rényi dimensions, lacunarity,
  residual entropy, spectral repetition, and range/domain self-similarity.
- Gaussian or low-rank residuals; C kernels with a NumPy fallback.
- Supervised training, grouped/stratified holdouts, evaluation, and exact linear-score explanations.
- TOML/YAML/JSON tasks, structured logs, portable JSON models, optional MLflow tracking.
- Recursive directory prediction with bounded batches, incremental JSONL output, and error reports.
- CSV/Parquet inputs, embedded image bytes, multiple recursive input paths, and explicit train/validation/test splits.
- An MLflow tutorial, runnable demo configuration, and an MLflow Projects entry point.
- Pytest and property tests, branch coverage, Ruff, ty, C sanitizers, dependency audits,
  CodeQL, cross-platform CI, documentation checks, and wheel release automation.

## Guides

- [Executed COCO / AI-GenBench notebook](notebooks/coco_aigenbench_demo.ipynb) and [launch instructions](notebooks/README.md)
- [Usage and Python API](docs/usage.md)
- [Python methods: inputs, behavior, and return values](docs/api.md)
- [Create your own configuration](docs/configuration.md)
- [Parquet, multiple paths, and train/validation/test splits](docs/data.md)
- [MLflow: install, launch, run detection, and inspect results](docs/mlflow.md)
- [Process large image directories](docs/directories.md)
- [How each measurement, training, and evaluation method works](docs/methods.md)
- [Technology stack and why each tool is used](docs/tech-stack.md)
- [Literature review](docs/literature.md) and [development checks](docs/development.md)

For directory detection, edit the input, model, and output paths in
`configs/predict-directory.toml`, then run:

```console
uv run fractal-detector run configs/predict-directory.toml
```

The protocol and project organization draw on
[Fast-Fractal-Compression](https://github.com/Ja-Gk-00/Fast-Fractal-Compression).
