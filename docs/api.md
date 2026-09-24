# Python method reference

The tables describe the public methods: what they take, what they do, and what they
return. Import them from `fractal_detector`; call `Detector` and `TrackingRun` methods
on their respective objects. See [usage](usage.md) for runnable examples and
[how the methods work](methods.md) for the underlying calculations.

## Image measurement

| Method | How it works | Returns |
| --- | --- | --- |
| `extract(image, config=None, source=None)` | Decodes a path, encoded bytes or NumPy array, normalizes it, computes features and diagnostics. Optional `source` sets the provenance name in reports/logs. | `FeatureResult` for one image. |
| `extract_dataset(samples, config=None, workers=1)` | Applies `extract` to each `Sample.path` using image worker threads; preserves sample order and raises on image errors. | A complete list of `FeatureResult` objects. |
| `box_dimension(mask, scales, backend="auto")` | Counts occupied tiles in a binary 2D mask at each scale, then fits the logarithmic count/scale slope. | `DimensionEstimate`. |
| `differential_dimension(image, scales, backend="auto")` | Counts vertical brightness boxes over tiles of a square grayscale array in `[0,1]`, then fits the count/scale slope. | `DimensionEstimate`. |
| `estimate_dimension(scales, counts)` | Fits `log(count)` against `log(1/scale)` using positive counts. Fewer than three positive counts yields an undefined estimate. | `DimensionEstimate` with slope, fit quality and original counts. |
| `native_available()` | Checks whether the optional C extension was imported successfully. | `bool`. |

The two image-dimension functions operate directly on arrays; they do not decode,
resize, or normalize images. Tile sizes must divide the relevant image dimensions.
The complete extraction pipeline checks its scale geometry through `FeatureConfig`.

## Data and configuration

| Method | How it works | Returns |
| --- | --- | --- |
| `read_manifest(path, require_labels=False)` | Reads CSV/Parquet rows, resolves relative image paths, validates labels/files and rejects duplicate paths. | A list of `Sample` objects, including embedded bytes when present. |
| `iter_manifest(path, ...)` | Streams CSV rows or Parquet record batches with optional column/label mappings supplied by a `DataSource`. | An iterator of `Sample` objects; no duplicate check in this primitive. |
| `iter_sources(sources, ...)` | Combines paths or `DataSource` objects; recursively discovers images/Parquet, applies labels, and deduplicates paths with a temporary SQLite index. | An iterator of `Sample` objects. |
| `iter_directory(path, ...)` | Lazily scans matching image files, optionally recursing; skips symlinks/junctions and requested excluded directories. | An iterator of `Sample` objects in filesystem order. |
| `load_config(path)` | Reads TOML, YAML or JSON, then validates task fields and settings. Does not process images. | `TaskConfig`. |
| `run_config(path)` | Loads a task file and calls `run_task`, resolving task paths relative to the config file. | A summary dictionary; detailed results are written to disk. |
| `run_task(config, base_dir=".")` | Resolves paths, prepares logs and optional tracking, then dispatches extraction, training, prediction or evaluation. | A summary dictionary containing the task, sample count, output directory and run ID. |

For `iter_directory`, use `recursive`, `extensions`, and `exclude_dirs` to control
discovery. See [configuration](configuration.md) for task settings and
[directory processing](directories.md) for batching, progress and error handling.

## Training and evaluation

| Method | How it works | Returns |
| --- | --- | --- |
| `train(samples, feature_config=None, training_config=None, workers=1)` | Extracts measurements, rejects duplicates, creates a holdout, fits a scaler/classifier on training samples, and evaluates the holdout. | `TrainingResult`: detector, evaluation, split indices and measurements. |
| `train_splits(train_samples, validation_samples=None, test_samples=None, ...)` | Checks disjoint images/groups, fits only training samples, and evaluates each supplied held-out split without resplitting. | `SplitTrainingResult`: detector, `evaluations` by split, and `measurements` by split. |
| `fit_features(samples, results, feature_config, training_config=None)` | Validates aligned labeled samples and precomputed features, then fits the scaler and classifier on all supplied rows. It creates no holdout. | `Detector`. |
| `evaluate(detector, samples, workers=1)` | Extracts measurements with the detector's saved settings and delegates to `evaluate_features`. | `Evaluation`. |
| `evaluate_features(detector, samples, results)` | Checks labels, duplicate images and fitting-set overlap, predicts each feature row, then computes classification metrics. | `Evaluation`: metrics, confusion matrix, count and predictions. |

For precomputed-feature methods, samples and results must be in matching order.
Training/evaluation require both labels, `0` for real and `1` for synthetic.
These methods retain their inputs/results in memory. Metric definitions and split
behavior are described in [training and prediction](methods.md#training-and-prediction).

## Detector methods

| Method | How it works | Returns |
| --- | --- | --- |
| `Detector.load(path)` | Reads numeric JSON, validates the model version, feature schema and weight shapes, then constructs the detector. | `Detector`. |
| `detector.save(path)` | Atomically writes the model's settings, scaler, weights, threshold and training provenance to JSON. | `None`. |
| `detector.predict(image)` | Extracts features using saved settings and delegates to `predict_features`. | `Prediction`. |
| `detector.predict_features(result)` | Validates the feature configuration, standardizes values, computes the logistic probability, applies the threshold and ranks feature contributions. | `Prediction` including the original measurement. |
| `detector.predict_many(samples, workers=1)` | Extracts all sample features in order and predicts each one. | A complete list of predictions. |
| `detector.iter_predict(samples, ...)` | Takes any sample iterable, extracts one bounded batch at a time, then yields predictions in input order. | An iterator of predictions, or `ImageFailure` records in skip mode. |
| `detector.predict_directory(path, ...)` | Connects lazy directory scanning to `iter_predict`. | The same incremental prediction iterator. |
| `detector.predict_sources(sources, ...)` | Connects `iter_sources` to batched prediction for multiple image/manifest/directory inputs. | The same incremental prediction iterator, with duplicate paths processed once. |

All prediction iterator methods accept `batch_size=64`, `workers=1`, and `on_error="raise"`.
Use `on_error="skip"` to receive explicit failure records for unreadable images.
Directory prediction also accepts `recursive` and `extensions`. Consuming the iterator
does the work; wrapping it in `list()` retains all results in memory.

## Logging and MLflow

`configure_logging(path, level="INFO")` attaches a JSON file handler to the package
logger, creating parent directories when needed. It returns the handler so you can
remove and close it when finished. See [logging setup](usage.md#logging-and-mlflow).

Use `TrackingRun(config)` as a context manager. It creates a run or attaches to the
configured running `run_id`. On exit, a run it created is marked finished or failed;
an attached run stays open. `TrackingRun()` enables tracking by default, while an
explicit `TrackingConfig()` defaults to disabled.

All methods below return `None`. Disabled tracking performs no uploads.

| Method | What it records |
| --- | --- |
| `run.log_parameters(parameters)` | Flattens nested dictionaries into dotted parameter names and logs their values. |
| `run.log_metrics(metrics, step=0)` | Logs named numeric values with a timestamp and step. |
| `run.log_dict(data, artifact_file)` | Writes a dictionary as a JSON artifact. |
| `run.log_artifact(path, artifact_path=None)` | Uploads an existing file, optionally beneath an artifact subdirectory. |
| `run.log_features(results, start_step=0)` | Logs feature settings, each image's features/timing as metrics, and complete measurement artifacts. |
| `run.log_predictions(predictions, start_step=0)` | Logs the measurements, synthetic probabilities, and prediction artifacts with explanations. |
| `run.log_training(result)` | Logs settings, model, provenance and measurements. A `TrainingResult` gives holdout metrics; a `SplitTrainingResult` gives separate validation/test metrics and reports. |
| `run.log_evaluation(evaluation)` | Logs evaluation metrics, its summary and individual predictions. |

When logging successive batches manually, advance `start_step` to avoid reusing
image indices and artifact names. Task execution does this automatically. Task-level
`image_logging="batch"` uses JSONL shards and mean metrics through the generic logging
methods; `log_features` and `log_predictions` always log individual images.
Follow the [MLflow tutorial](mlflow.md) to launch the server and inspect these records.

## Research extensions

`FeatureConfig()` and `NeuralConfig()` now default to `image_size="auto"`;
`TrainingConfig()` defaults to seed 1337. Single-image extraction resolves its own
size; `extract_dataset` resolves a shared size. Training resolves only from fitting
data, while saved models keep concrete sizes for inference. See
[automatic sizing](method-catalog.md#automatic-image-sizing) for alignment and streaming behavior.

All earlier calls and default feature ordering remain supported. These additions are
described mathematically in [research methods](research-methods.md).

| API | Inputs and behavior | Return |
| --- | --- | --- |
| `multifractal_spectrum(mass, scales, orders, backend="auto", mass_floor=0)` | Nonnegative square mass array; fits partition scaling and computes alpha/f(alpha). Zero masses are excluded; diagnostics mark an entirely empty input. | Feature dictionary and full diagnostic dictionary |
| `sliding_lacunarity(mass, scales, stride=1)` | Uses summed-area tables over all valid windows, with no padding. | One lacunarity per scale |
| `method_preset(method, deep=False)` | One of the 18 names in `presets.METHODS`. | Feature/training/neural settings dictionary |
| `write_method_configs(destination, train=..., validation=..., test=None)` | Writes and validates 36 paired TOML task files. | List of written paths |
| `search_features(train_samples, validation_samples, SearchConfig(...), training_config=None, workers="auto")` | Fits only on training data; selects by validation AUC, then log loss, then smaller feature count. | `SearchResult`: detector, trial records, best index, report |
| `SearchCandidate(name=..., features=FeatureConfig(...))` | Names a complete feature configuration for search. | Validated candidate |
| `SearchConfig(candidates=(...), regularization_c=(0.1, 1.0, 10.0))` | At most 100 candidate/regularization pairs. | Validated search configuration |

The following imports come from `fractal_detector.neural`, which requires the `neural`
extra. `NeuralConfig` itself is available from the package root without PyTorch.

| API | Behavior | Return |
| --- | --- | --- |
| `train_neural(train_samples, validation_samples, test_samples=None, feature_config=None, training_config=None, neural_config=None, workers="auto", epoch_callback=None)` | Checks decoded-image and group separation, trains the configured network and restores the best validation-loss checkpoint. Optional callback receives `(metrics, epoch)` after each epoch. | `NeuralTrainingResult`: detector, evaluation reports, measurements by split |
| `NeuralDetector.load(path, device="auto")` / `save(path)` | Validates metadata, checksum and tensor shapes; persists safetensors inside a JSON artifact. | Detector / `None` |
| `detector.predict(image)` | Network prediction plus independent handcrafted measurements and final-head contributions. | Existing `Prediction` type |
| `detector.predict_many(samples, workers="auto")` | Batched prediction, raising on unreadable input. | List of predictions |
| `detector.iter_predict(samples, workers="auto", batch_size=None, on_error="raise")` | Bounded batches from any iterable; `skip` yields ordered `ImageFailure` records. | Iterator of predictions/errors |
| `evaluate_neural(detector, samples, workers="auto", allow_validation=False)` | Computes the existing classification metrics; rejects fitting and model-selection overlap by default. `allow_validation=True` permits the stored validation data for an explicitly labeled validation report only. | `Evaluation` |

Use `detector.iter_predict(iter_sources(("images", "images.parquet")))` for recursive,
multiple-source neural prediction. Neural detectors do not offer `predict_features`:
the image tensor, not a handcrafted feature record, is the network input. Existing
`Detector.load` remains the schema-1 linear-model loader; use `NeuralDetector.load`
for schema-2 neural models. The CLI selects the neural path when `[neural]` is present.
