# Changelog

## Unreleased

- Default all experiment seeds to 1337, including sampling, presets and examples.
- Select omitted image sizes from the largest source dimension, rounded up to valid
  method geometry; explicit integer sizes still override this behavior. Training
  uses training images only and stores the resolved size for validation/prediction.
- Stream automatic extraction through a disk-backed sizing pass, including embedded
  Parquet images, with resolved settings in logs and artifacts.
- Replace 36 method JSON presets with TOML; add ranked TOML configurations and a
  guide covering every method, required arguments and run commands.
- Repeat the local feature comparison with seed 1337 and document its limits.

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

## Research extensions

- Preserve legacy APIs/model loading; add full multifractal spectra, median residuals, sliding lacunarity, PCA rank sweeps, feature-family selection and auto CPU sizing.
- Add optional trainable SRM and residual Fractal-CNNs, joint DnCNN training, safe tensor artifacts, early stopping, explanations and MLflow integration.
- Add 36 paired method configs, bounded validation-only parameter search with selection-data provenance, real-data experiments and future-work documentation.
- Test analytical spectra, compatibility, leakage guards, learned gradients, serialization, streaming and all presets.
