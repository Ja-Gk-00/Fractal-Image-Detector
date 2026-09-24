# Make your own configuration

Default random seed: **1337**. Omit `features.image_size` and `neural.image_size`
for automatic dataset sizing, or set an integer to override it. Training resolves
sizes from training images only; prediction uses the saved model's sizes. See the
[ranked method catalog](method-catalog.md) for the TOML presets, required fields,
and exact sizing rules.

A configuration describes **one task**. Save it as TOML, YAML, or JSON and run:

```console
uv run fractal-detector run configs/my-task.toml
```

The steps below use TOML. You can copy `configs/predict-directory.toml` to get started.

## 1. Choose a task and input

| Task | Input | Required model? | Result |
| --- | --- | --- | --- |
| `extract` | Images, directories, CSV/Parquet, or a selected data split | No | Fractal measurements and other features |
| `train` | Labeled inputs or explicit `data.train` with optional validation/test | No | Fitted model and available evaluation reports |
| `predict` | Images, directories, CSV/Parquet, or a selected data split | Yes | Labels, probabilities and explanations |
| `evaluate` | Labeled inputs or a selected data split | Yes | Classification metrics and predictions |

Specify exactly one of `input_dir`, `manifest`, `inputs`, and `data`. Directory names
do not supply labels; use labeled CSV/Parquet rows or source objects with `label`.
The [data guide](data.md) explains multiple paths, custom columns and explicit splits.

## 2. Point to your files

For a directory of unknown images, create `configs/my-task.toml`:

```toml
task = "predict"
input_dir = "D:/datasets/images"
model = "../outputs/train/model.json"
output_dir = "../outputs/my-predictions"

recursive = true
batch_size = 64
workers = 4
on_error = "skip"
```

Replace the paths with your own. `model` must be a model you already trained, for example
with `configs/train.toml`. To inspect images before training, change `task` to `extract`
and remove `model`.

Relative paths are resolved from the **configuration file's directory**. For example,
`../outputs/train/model.json` inside `configs/my-task.toml` points to the repository's
`outputs/train/model.json`. CSV/Parquet image paths are relative to the manifest file.
Use forward slashes on Windows to avoid TOML backslash escapes.

All top-level settings must appear before `[data]`, `[features]`, `[training]`, or `[tracking]`
sections in TOML. Otherwise TOML treats them as fields of that section.

## 3. Choose processing settings

- `recursive=true` includes subdirectories. `false` scans only the specified folder.
- `extensions=[".jpg", ".jpeg", ".png"]` limits the formats. Matching is case-insensitive.
  The default also includes WebP, BMP and TIFF.
- `batch_size=64` keeps at most one batch of result objects in memory. Lower it when
  results are large. The allowed range is 1–4096.
- `workers=4` processes up to four images concurrently. Start with 2–4, then measure
  speed and memory use on your machine.
- `on_error="raise"` stops when an image cannot be read. `"skip"` records that error
  and continues for extraction/prediction. Training/evaluation always stop on errors.

For a large collection, see [directory processing](directories.md).

## 4. Configure measurements when training or extracting

```toml
[features]
image_size = 128
scales = [2, 4, 8, 16, 32]
residual = "gaussian"
gaussian_sigma = 1.0
compression = true
max_domains = 64
backend = "auto"

[training]
test_size = 0.25
seed = 1337
regularization_c = 1.0
threshold = 0.5
```

Each scale must divide `image_size` and be at most half of it. If you choose an image
size of 64, for example, `[2,4,8,16]` is valid. Use `residual="svd"` and `svd_rank=8`
to try a low-rank residual. The [method reference](methods.md) explains every feature setting.

For prediction or evaluation, omit `[features]`: the model supplies the exact settings
used for training. Conflicting overrides are rejected. `[training]` is used only by `train`.

For training, the beginning of the file would be:

```toml
task = "train"
manifest = "../data/train.csv"
output_dir = "../outputs/train"
workers = 4
```

See [usage](usage.md#prepare-a-manifest) for the CSV format and grouped splitting.
For existing splits, copy `configs/train-splits.toml`: only `data.train` is fitted,
and validation/test are reported separately without resplitting.

## 5. Optionally enable MLflow

```toml
[tracking]
enabled = true
uri = "http://127.0.0.1:5000"
experiment = "my-image-detector"
run_name = "first-directory-test"
image_logging = "batch"
```

Start the server before running this configuration. Follow the short
[MLflow tutorial](mlflow.md) for the commands and what to click in the UI.
Use `image_logging="full"` for per-image metrics and individual artifacts;
`"batch"` uploads complete extraction/prediction records in batch files and logs batch
means, which reduces tracking requests for large collections. Training/evaluation retain
their existing detailed reports and per-image logging.

## 6. Run and check the output

```console
uv run --extra mlflow fractal-detector run configs/my-task.toml
```

The command prints a final summary. For extraction/prediction, `summary.json` is updated
after each batch and reports processed, successful and failed counts. Open
`predictions.jsonl` for predictions, `features.jsonl` for extraction, and `errors.jsonl`
for unreadable files. Choose a new `output_dir` for each experiment.

Configuration errors, including misspelled fields, are reported before image processing.
They never silently switch to default values. To validate a file from Python without
running it:

```python
from fractal_detector import load_config

config = load_config("configs/my-task.toml")
print(config)
```

## Research configuration

See [the preset guide](research-methods.md) for all 36 standard/deep files. Set
`workers="auto"` to use the CPU count available to the process. Explicit worker counts
remain supported. Numeric libraries use one BLAS thread per image extraction worker.

New `[features]` settings:

| Setting | Default | Meaning |
| --- | --- | --- |
| `residual` | `"gaussian"` | Also accepts `"median"` and `"svd"` |
| `median_size` | 3 | Odd median window in 3–15 |
| `entropy_bins` | 64 | Histogram bins, 8–1024 |
| `families` | omitted | Select feature families; omit to retain all enabled families |
| `multifractal` | false | Append the full-spectrum feature vector |
| `moment_orders` | `[-4,-2,-1,0,1,2,4]` | At least three increasing distinct orders in −16…16 |
| `mass_floor` | 0 | Optional floor on occupied normalized box probabilities, at most 0.01 |
| `lacunarity_mode` | `"tiled"` | `"sliding"` uses overlapping valid windows |
| `lacunarity_stride` | 1 | Positive sliding-window stride |
| `low_correlation` | false | Append PCA-rank measurements |
| `pca_ranks` | `[24,28,32]` | Increasing distinct positive ranks below image size when enabled |

`families` accepts `dimensions` (all three estimators), `gray_surface`,
`residual_surface`, `edge_box`, `renyi`, `lacunarity`, `statistics`, `spectral`,
`compression`, `multifractal`, and `low_correlation`. Enable the corresponding boolean
before selecting compression, multifractal or low-correlation features. Dimensions
and diagnostics are still logged even when a family is excluded from the classifier.

Add `[neural]` to train an optional learned model. Neural training requires explicit
`data.train` and `data.validation`. Choose `architecture="srm_fractal"` or
`"residual_fractal"`; the latter accepts residual `median`, `gaussian`, or `dncnn`.
The presets expose geometry, channels, backbone/DnCNN depth, optimizer, augmentation,
early stopping and device settings. `TrainingConfig` seed, threshold and class weights
also apply; logistic C/max_iter do not control neural training.

For feature search use `task="search"`, explicit train/validation splits and
`[search] regularization_c=[0.1,1.0,10.0]`. Add candidates with TOML array tables:

```toml
[[search.candidates]]
name = "gaussian-256"
[search.candidates.features]
image_size = 256
scales = [2, 4, 8, 16, 32, 64]
compression = false
multifractal = true
```

The search ignores `data.test` entirely, including its file paths. Its selected model
contains validation provenance; use separate unseen inputs for subsequent evaluation.
The runnable `examples/research_search.py` compares ten feature configurations and
three C values using a notebook sample receipt.
