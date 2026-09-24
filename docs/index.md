# Fractal Image Detector

Measure image self-similarity, train a labeled real/synthetic classifier, and inspect
every prediction. The package runs on CPU, accepts files and NumPy arrays, and stores
portable JSON models. Its C extension accelerates range/domain matching and box statistics.

Start with [usage and the Python API](usage.md), then follow these guides:

- [Choose a method: measured ranking, TOML presets and required arguments](method-catalog.md).
- [Look up each Python method and its return value](api.md).
- [Create your own configuration](configuration.md).
- [Combine Parquet/files/directories and supply train/validation/test splits](data.md).
- [Launch MLflow and inspect your first detection run](mlflow.md).
- [Process a large image directory in bounded batches](directories.md).
- [Understand the fractal method](methods.md) and its [research basis](literature.md).
- [Use the learned models, multifractal spectrum, and 36 method presets](research-methods.md).
- [Read measured validation results](research-results.md) and [future improvements](future-work.md).
- [Understand the technology stack](tech-stack.md) and [run development checks](development.md).

No pretrained detector or validated accuracy claim is included. Train with images
whose provenance you know and evaluate on independent generators, sources, and subjects.
