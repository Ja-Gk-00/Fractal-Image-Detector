# Changelog

## 0.1.0

- Expanded notebook results with held-out real/synthetic and error galleries, per-class fractal/texture statistics, fit-quality and source-size plots, explanations for both classes, and CSV/MLflow exports.
- Executed COCO / AI-GenBench notebook with balanced sampling, related-image split grouping, every task/input mode, feature plots, explanations, and local MLflow tracking.
- Locked Jupyter dependencies, an execution/validation command, and CI checks for the saved notebook.
- Batched Parquet ingestion for paths/embedded images, multiple source paths, recursive shard discovery, and disk-backed path deduplication.
- Explicit train/validation/test datasets with leakage checks, independent reports, and MLflow split metrics.
- Recursive directory extraction/prediction with bounded batches, progress summaries, and explicit error handling.
- MLflow batch artifacts for large tasks and an MLflow Projects entry point using the launcher's run.
- Step-by-step MLflow, configuration, technology stack, and large-directory guides.
- Initial fractal measurement, supervised training, evaluation, and prediction API.
- C kernels for box statistics and affine domain search, with a NumPy fallback.
- Configuration tasks, JSON model artifacts, explanations, logging, and MLflow integration.
- Cross-platform quality checks, tests, documentation, and release build workflows.
