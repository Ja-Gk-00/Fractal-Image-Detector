# Data sources and explicit splits

Use `inputs` to combine image files, directories, CSV manifests and Parquet files.
Use `[data]` when you have already separated training, validation and test data.
The existing `manifest` and `input_dir` configurations still work. Choose exactly one
of these four input forms per task.

After updating the package, install its Parquet reader with the regular environment sync:

```console
uv sync --locked --extra mlflow
```

## Multiple paths and nested directories

```toml
task = "predict"
inputs = ["D:/photos", "D:/generated/nested", "../data/shard.parquet", "../data/list.csv"]
model = "../outputs/train/model.json"
output_dir = "../outputs/predictions"
recursive = true
batch_size = 64
workers = 4
on_error = "skip"
```

Edit the paths in `configs/predict-multiple.toml`, then run:

```console
uv run fractal-detector run configs/predict-multiple.toml
```

- Directory discovery includes images and `.parquet` files at every depth when
  `recursive=true`. CSV manifests must be listed explicitly.
- `extensions` filters discovered image files. Parquet discovery is independent of
  that filter; image paths inside manifests are used as written.
- Symlinks and directory junctions encountered during discovery are skipped. Output
  subdirectories are excluded from discovery automatically.
- Repeated image paths are processed once, including overlaps between roots and
  manifests. Conflicting labels/groups for the same path raise an error. A temporary
  SQLite index stores these identities on disk; it is removed when iteration closes.
- Different paths containing identical pixels remain separate prediction inputs.
  Training and evaluation reject duplicate decoded images.

Sources are read in configured order; directory entries follow filesystem order.
Task paths are relative to the configuration file. Image paths inside CSV/Parquet
files are relative to that manifest's directory. These are local paths, without glob
expansion or automatic remote downloads.

## Parquet schema

The default columns are:

| Column | Accepted content |
| --- | --- |
| `path` | Absolute image path or path relative to the Parquet file. |
| `image` | Encoded image bytes, or a struct with `bytes` and `path` fields. This supports common Hugging Face image records. |
| `label` | Optional `0`/`real` or `1`/`synthetic`; required for supervised tasks unless supplied on the source. |
| `group` | Optional string or integer identifying related samples. |

Provide `path` or `image`. Embedded bytes take precedence over paths; a null `bytes`
field falls back to the path. Bytes must contain an encoded image such as PNG or JPEG,
not a raw pixel buffer. Extra Parquet metadata columns are ignored. Records retain
their physical row order; embedded-image identifiers are `file.parquet#row=0`, etc.
The raw bytes are excluded from JSON reports and MLflow uploads.

For example, write a path-based manifest:

```python
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.table(
    {
        "path": ["photos/camera.png", "generated/image.png"],
        "label": [0, 1],
        "group": ["camera-subject-1", "generated-prompt-1"],
    }
)
pq.write_table(table, "data/images.parquet")
```

The paths above refer to images beneath `data/`. This two-row example illustrates
the format; automatic holdout training needs at least two images per class.

The reader selects only the configured columns and uses
[PyArrow record batches](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html#pyarrow.parquet.ParquetFile.iter_batches).
`batch_size` controls both reader rows and extraction batches. Arrow may also retain
decoded pages and Parquet metadata; a row limit is not a byte-level memory limit.
Training/evaluation materialize their datasets and features in memory.

## Custom columns and labels

Each path can also be a source object. A source object groups paths that share settings:

```toml
task = "train"
output_dir = "../outputs/custom-training"
inputs = [
  { paths = ["../data/a.parquet", "../data/b.parquet"], path_column = "filename", label_column = "category", group_column = "subject", label_map = { camera = 0, generated = 1 } }
]
```

The configurable columns are `path_column`, `image_column`, `label_column` and
`group_column`. Label text is trimmed and lowercased before lookup; use lowercase
`label_map` keys. Numeric class IDs can be mapped with quoted keys, such as
`label_map = { "4" = 0, "9" = 1 }`.

For directories of known provenance, explicitly assign labels:

```toml
inputs = [
  { paths = ["../data/camera-a", "../data/camera-b"], label = 0 },
  { paths = ["../data/generated"], label = 1 }
]
```

These labels apply to every record from those paths. They fill missing manifest labels;
conflicting existing labels fail. Folder names never determine labels automatically.
When used with `task="train"` and no `[data]`, the normal automatic holdout is retained.

## Supply train, validation and test explicitly

```toml
task = "train"
output_dir = "../outputs/explicit-training"
workers = 4

[data]
train = ["../data/train-a.parquet", "../data/train-b.parquet"]
validation = ["../data/validation.csv"]
test = ["../data/test.parquet"]
```

Every split accepts the same path strings and source objects as `inputs`, including
labeled nested directories. Copy `configs/train-splits.toml` for a complete example.

- The scaler and classifier fit **only on `data.train`**. No random split or refit is
  performed; `training.test_size` is ignored in this mode.
- Validation and test are optional. When supplied, each must contain both labels.
  They are evaluated separately using the same fitted model and configured threshold.
  There is no automatic threshold tuning, model selection, or early stopping.
- Supplying only `data.train` fits all its rows and produces no holdout score.
- Images and groups must be disjoint across all supplied splits, including between
  validation and test. If any sample has a group, all samples in all splits must have one.
- Use validation for your own configuration choices; reserve test data for final
  assessment. To keep test unseen during experiments, omit it until the final evaluation.

Outputs include `model.json`, `split-manifest.json`, `split.json`, per-split
`train-features.jsonl` / `validation-features.jsonl` / `test-features.jsonl`, and separate
`validation-report.json` / `test-report.json` when those splits are present.
`report.json` combines the evaluation reports. MLflow records `validation.*` and
`test.*` metrics, with `validation.json` and `test.json` artifacts.

## Select a split for prediction or evaluation

For `predict`, `extract` or `evaluate`, select the split to consume:

```toml
task = "evaluate"
split = "test"
model = "../outputs/explicit-training/model.json"
output_dir = "../outputs/test-evaluation"

[data]
test = ["../data/test.parquet"]
```

Only the selected split is read. Prediction/extraction do not require labels; evaluation
does. The standalone example is `configs/evaluate-split.toml`.

## Python API

```python
from fractal_detector import Detector, iter_sources, train_splits

result = train_splits(
    list(iter_sources(["data/train.parquet"], require_labels=True)),
    validation_samples=list(iter_sources(["data/validation.csv"], require_labels=True)),
    test_samples=list(iter_sources(["data/test.parquet"], require_labels=True)),
)
result.detector.save("model.json")
print(result.evaluations["validation"].metrics)

detector = Detector.load("model.json")
for prediction in detector.predict_sources(["incoming/photos", "incoming/shards"], batch_size=64):
    print(prediction)
```

Use `DataSource(paths=(...), label=0, ...)` for custom source settings in Python.
`iter_manifest` streams one CSV/Parquet file; `read_manifest` loads it into a list and
rejects duplicate paths. `iter_sources` streams and deduplicates multiple sources.
Malformed manifest schemas, labels, or missing referenced paths stop the task.
`on_error="skip"` applies to image decoding/measurement failures, which are recorded
in `errors.jsonl`; it does not hide dataset configuration errors.
