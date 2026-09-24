# Usage and API

For a first tracked run, follow the [short MLflow tutorial](mlflow.md).
For file-based tasks, start with [creating a configuration](configuration.md).
For a concise description of each callable method, see the [Python method reference](api.md).
For Parquet, multiple input paths and explicit dataset splits, see [data sources](data.md).

## Install and run

Use Python 3.14. The repository pins 3.14.7 and includes a `uv.lock` with exact dependency
versions. From the repository root:

```console
uv sync --locked --all-extras --group docs
uv run fractal-detector doctor
uv run python examples/demo.py
```

The demo generates procedural textures, exercises training and prediction, and writes
artifacts under `outputs/demo`. It is a software smoke test, not an AI detection benchmark.

Without uv, create and activate a Python 3.14 virtual environment, then:

```console
python -m pip install ".[mlflow]"
fractal-detector doctor
```

Source builds try to compile the C extension. Install Visual Studio C++ Build Tools on
Windows, Xcode command-line tools on macOS, or a C compiler and Python headers on Linux.
If compilation is unavailable, installation uses the NumPy implementation. Select
`backend="native"` to require acceleration and get an explicit error if unavailable.
Set `FRACTAL_REQUIRE_NATIVE=1` during installation to make a failed native build fatal.
Set `FRACTAL_BUILD_NATIVE=0` to install without compiling.

## Measure one image

```python
from fractal_detector import FeatureConfig, extract

config = FeatureConfig(backend="auto")
measurement = extract("photo.jpg", config)
print(measurement.dimensions["gray"].value)
print(measurement.dimensions["gray"].r_squared)
print(measurement.dimensions["gray"].scales)
print(measurement.dimensions["gray"].counts)
print(measurement.features)
print(measurement.warnings)
```

`extract` also accepts encoded image bytes and arrays with shape `(H,W)`, `(H,W,3)`, or `(H,W,4)`:
`uint8` in `[0,255]` or finite floating point in `[0,1]`. Inputs are copied as needed;
caller arrays are never modified. EXIF orientation is applied to files, transparency is
composited over white, and luminance is center-cropped and resized. Images must be at
least 16 pixels per side. Animated and high-bit-depth files require explicit conversion.
Raw camera files and video are not supported.

`FeatureResult` contains dimensions, all 21 default features, per-scale diagnostics,
compression matches, preprocessing warnings, the actual backend, input identity, and
elapsed time. Turn off compression for 16 features. Serialize with
`measurement.model_dump_json()` or `measurement.model_dump(mode="json")`.
Image identity is SHA-256 over the original dimensions and decoded grayscale float64
pixels, before cropping and resizing. It detects exact decoded copies, not near-duplicates.

## Prepare a manifest

```csv
path,label,group
photos/camera_a_001.jpg,real,camera-a-subject-001
photos/camera_b_002.jpg,0,camera-b-subject-002
generated/model_a_001.png,synthetic,generator-a-prompt-001
generated/model_b_002.png,1,generator-b-prompt-002
```

Paths resolve relative to the CSV. Labels are `real`/`0` and `synthetic`/`1`. The `group`
column is optional. If present for any training sample, provide it for all samples.
Use it to keep related subjects, source images, prompt families, or other dependent
observations together. Training needs both classes and at least two distinct images
per class, with enough images/groups for both classes in the holdout.

For extraction and prediction, only `path` is required. Empty manifests, duplicate paths,
malformed labels, unknown columns, missing files, and unreadable images fail explicitly.
Processing never silently skips samples. For large collections, use a directory task
or `Detector.predict_directory`: both process bounded batches and let you explicitly
record and skip unreadable images. See [large image directories](directories.md).
`extract_dataset` and `predict_many` retain their complete results in memory.

## Train, save, predict, evaluate

The example below creates an automatic holdout. For existing train/validation/test
datasets, use [`train_splits` or a `[data]` task](data.md#supply-train-validation-and-test-explicitly).

```python
from fractal_detector import (
    Detector,
    FeatureConfig,
    TrainingConfig,
    evaluate,
    read_manifest,
    train,
)

samples = read_manifest("train.csv", require_labels=True)
trained = train(
    samples,
    FeatureConfig(residual="gaussian"),
    TrainingConfig(seed=1337, test_size=0.25),
    workers=4,
)
print(trained.evaluation.metrics)
trained.detector.save("model.json")

detector = Detector.load("model.json")
prediction = detector.predict("unknown.png")
print(prediction.label, prediction.synthetic_probability)
print(prediction.explanation)
print(prediction.contributions)

test = read_manifest("independent_test.csv", require_labels=True)
report = evaluate(detector, test, workers=4)
print(report.metrics, report.confusion_matrix)
```

Training performs a reproducible stratified split when no groups exist; otherwise it
uses a group-disjoint split with both classes on each side. Standardization and logistic
regression are fitted only on the training portion. The returned detector is that exact
model, not a refit using the holdout. Split indices refer to manifest order. Exact image
duplicates are rejected before fitting. Independent evaluation rejects fitting images
and fitting groups, and requires both classes for meaningful ROC-AUC.

Metrics include accuracy, balanced accuracy, ROC-AUC, average precision, log loss,
and Brier score. Confusion matrix rows are actual labels `[real, synthetic]`, columns
are predicted labels in the same order. `threshold` defaults to 0.5. Choose thresholds
on a separate validation set; repeated tuning against the holdout biases its estimate.
The probabilities have not received a separate calibration step.

For your own cross-validation loop, call `extract_dataset(samples, config)`, then
`fit_features(training_samples, training_measurements, config, training_config)` and
`evaluate_features(detector, validation_samples, validation_measurements)`. Reuse feature
results only with identical extraction settings; the API checks the feature schema and
configuration. `fit_features` performs no split. `Detector.predict_features(result)`
accepts a compatible measurement. JSON model loading validates versions, names, shapes,
and finite weights and never executes pickle data.

## Configuration tasks

```console
uv run fractal-detector run configs/train.toml
uv run fractal-detector run configs/extract.yaml
uv run fractal-detector run configs/predict.json
uv run fractal-detector run configs/evaluate.toml
```

Edit the input and output paths first. TOML, JSON, and YAML are equivalent; YAML uses safe loading.
Unknown fields and invalid types are errors. Task file paths resolve relative to the
configuration file. For `run_task(TaskConfig(...), base_dir=...)`, they resolve relative
to `base_dir`. MLflow URIs are passed unchanged and are not resolved as file paths.

| Setting | Default | Meaning |
| --- | --- | --- |
| `task` | required | `extract`, `train`, `predict`, `evaluate` |
| `manifest` / `input_dir` / `inputs` / `data` | exactly one required | CSV/Parquet, one directory, multiple source paths, or named dataset splits; see [data](data.md) |
| `split` | none | Select `train`, `validation`, or `test` for a non-training task using `data` |
| `output_dir` | `outputs` | Output directory; use a different directory for each task |
| `model` | none | JSON model path required for prediction/evaluation |
| `workers` | 1 | Image extraction threads, 1–1024, or `"auto"` for the available CPU count |
| `batch_size` | 64 | Maximum images retained per extraction/prediction batch, 1–4096 |
| `recursive` | true | Include input subdirectories |
| `extensions` | common image formats | Allowed suffixes, matched without case sensitivity |
| `on_error` | `raise` | Stop on an unreadable image, or explicitly `skip` for extraction/prediction |
| `features` | protocol defaults | See the [measurement protocol](methods.md) |
| `training` | see below | Split and classifier settings |
| `tracking` | disabled | MLflow settings |
| `log_level` | `INFO` | JSON log level |

Training settings: `seed=1337`, `test_size=0.25`, `regularization_c=1.0`, `max_iter=2000`,
`threshold=0.5`, `class_weight=None`. `class_weight="balanced"` adjusts fitting weights
for class imbalance; this also changes the interpretation of uncalibrated probabilities.
Nonconverged fitting fails rather than saving a partially converged model.

Prediction and evaluation use the preprocessing stored in the model. Omit `features`
in those configurations; explicitly supplied conflicting settings fail.

Every task writes `config.json`, `environment.json`, and `events.jsonl`.
CSV inputs also write `manifest.json`. Extraction/prediction record attempted paths in
`manifest.jsonl`, failures in `errors.jsonl`, and progress in `summary.json` after each batch.
Extraction writes `features.jsonl`. Training adds `model.json`, `features.jsonl`,
`split.json`, and `report.json`. Prediction writes `predictions.jsonl`; evaluation
writes `report.json` including individual predictions. Existing result files are replaced
and event logs append, so use unique output directories to separate runs.

Explicit-split training writes separate feature files and reports for each supplied
split; see the [output descriptions](data.md#supply-train-validation-and-test-explicitly).

## Logging and MLflow

CLI tasks log structured JSON automatically, including fractal dimensions and counts.
Library calls use the `fractal_detector` logger and do not alter root logging settings:

```python
import logging
from fractal_detector import configure_logging, extract

handler = configure_logging("measurements.jsonl", "INFO")
try:
    measurement = extract("photo.jpg")
finally:
    logging.getLogger("fractal_detector").removeHandler(handler)
    handler.close()
```

Log records and artifacts contain source paths and per-image measurements. Original
image files are not uploaded. The documented statistics are retained; full pixel arrays,
FFT arrays, and SVD factors are not retained in reports.

Enable MLflow in any task configuration:

```toml
[tracking]
enabled = true
uri = "http://127.0.0.1:5000"
experiment = "fractal-detection"
run_name = "gaussian-baseline"
```

Start a local server with the optional extra installed:

```console
uv run --extra mlflow mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Or use an absolute SQLite URI directly, such as `sqlite:///D:/experiments/mlflow.db`
on Windows or `sqlite:////home/me/experiments/mlflow.db` on Linux. Omitting `uri`
uses MLflow's configured default, including `MLFLOW_TRACKING_URI`. Standard MLflow
authentication environment variables apply to remote servers.

Use the same integration directly from Python:

```python
from fractal_detector import TrackingConfig, TrackingRun, extract, read_manifest, train

with TrackingRun(TrackingConfig(enabled=True, uri="http://127.0.0.1:5000")) as run:
    measurement = extract("photo.jpg")
    run.log_features([measurement])

with TrackingRun(TrackingConfig(enabled=True, experiment="training")) as run:
    trained = train(read_manifest("train.csv", require_labels=True))
    run.log_training(trained)
```

`log_predictions` and `log_evaluation` cover the other APIs. Feature and training settings
are searchable MLflow parameters; `log_parameters` also accepts your own nested mappings.
Metrics use image indices
as steps; complete measurements, explanations, configuration, and the JSON model are
artifacts. The CLI logs runtime versions and manifest provenance too. Download `model.json`
from a run and pass it to `Detector.load`; no MLflow runtime is needed for that model file.
Supply `run_id` to attach to your own running experiment. Attached runs are left running;
owned runs are marked `FINISHED` or `FAILED`. The integration does not change MLflow's
global tracking URI or active run. Tracking errors propagate instead of silently losing data.

For large extraction/prediction tasks, set `[tracking] image_logging="batch"` to upload
one JSONL artifact per batch and batch-mean metrics. Every per-image dimension and
explanation remains in those artifacts. The default `full` mode records individual
artifacts and metric histories. See the [MLflow tutorial](mlflow.md) for the UI workflow
and the optional `mlflow run` launch command.

## Interpreting predictions

Each feature contribution is `weight × (value − training_mean) / training_scale`.
The intercept plus all contributions gives the synthetic log-odds; applying the logistic
function gives the reported probability. Positive contributions favor synthetic, negative
ones favor real. This exactly explains the linear model's score; it is not a causal
explanation of how an image was made.

Dimensions describe a finite image at selected scales. They are affected by texture,
cropping, resizing, sharpening, denoising, and compression. Inspect low-fit and empty-residual
warnings. Match preprocessing between datasets, keep related samples together, and use
unseen generators and camera sources for a final test. Dataset provenance and shift matter
more than a high score on a small, convenient split.
