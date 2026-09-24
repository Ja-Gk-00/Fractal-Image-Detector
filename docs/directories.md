# Detect images in large directories

Use a directory directly when you have many images and no labels are needed. Both
`predict` and `extract` support this input. Training/evaluation accept labeled CSV/Parquet
or explicitly labeled source directories; see [data sources and splits](data.md).

## Command-line task

Create `configs/my-directory.toml`:

```toml
task = "predict"
input_dir = "D:/datasets/incoming-images"
model = "../outputs/train/model.json"
output_dir = "../outputs/incoming-predictions"
recursive = true
batch_size = 64
workers = 4
on_error = "skip"

[tracking]
enabled = false
image_logging = "batch"
```

Then run:

```console
uv run fractal-detector run configs/my-directory.toml
```

Copy `configs/predict-directory.toml` for an example that includes the full extension
filter and optional MLflow server settings. To extract measurements without a model,
use `task="extract"` and remove `model`.
For several roots or Parquet files, use `inputs=["folder-a", "folder-b", "data.parquet"]`
as shown in `configs/predict-multiple.toml`. Task discovery descends into nested folders
and reads discovered Parquet shards as well as image files.

## What makes this suitable for large collections?

- Extraction with omitted or `"auto"` image size first scans all image headers to
  choose one shared analysis size. Sample records and embedded Parquet bytes are
  spooled to temporary SQLite storage, then replayed in batches. Allow disk space
  and a first-pass delay before results appear. Setting `[features] image_size=256`
  skips this pass. Prediction uses the saved model's size and needs no sizing scan.
- Directory entries are discovered lazily with `os.scandir`; even a flat folder is not
  converted to a list of all filenames. Traversal order follows the filesystem.
- At most `batch_size` measurements/predictions are retained at a time. Up to `workers`
  images are being decoded/computed concurrently. No image-count-sized feature matrix
  is created for directory extraction or prediction.
- Results and provenance are written as JSONL and flushed after each batch. Counts in
  `summary.json` are updated after each batch, so you can inspect progress while it runs.
- Optional MLflow batch logging uploads one complete JSONL artifact per batch and records
  aggregate metrics instead of uploading many individual image artifacts.

Memory also depends on each decoded image's original resolution, the saved model, and
feature settings. The batch size does not limit bytes per input image. Start with four
workers and a batch size of 64; lower either if memory use is high. Model fitting and
ROC-AUC evaluation are still in-memory operations. Parquet and the `inputs` form stream
manifest batches. The legacy `manifest="file.csv"` form loads CSV rows into memory;
use `inputs=["file.csv"]` to stream them. A temporary SQLite index keeps duplicate-path
checks off the Python heap; allow temporary disk space for large collections.

The scanner accepts JPEG, PNG, WebP, BMP and TIFF by default. The extension filter is
case-insensitive. Symbolic links and directory junctions are not traversed. An output
subdirectory inside the input tree is excluded automatically. The output directory must
not equal or contain the input directory. Maintain a stable input directory during a run:
the scanner is not a filesystem snapshot. Tasks and `predict_sources` deduplicate repeated
paths across inputs. Exact copies at different paths each receive a prediction.

## Output files and failures

| File | Contents |
| --- | --- |
| `predictions.jsonl` | One successful prediction per line, including dimensions and explanations |
| `features.jsonl` | Measurement records for an extraction task |
| `resolved-features.json` | Concrete feature settings chosen by automatic extraction sizing |
| `manifest.jsonl` | Input path, optional CSV label/group, and processing index for every attempted image |
| `errors.jsonl` | Processing index, path, error type and message for each failed image |
| `summary.json` | Status, attempted/successful/failed counts, elapsed time and MLflow run ID |
| `events.jsonl` | Detailed measurements, predictions, errors and batch-progress log events |
| `batches/` | Complete JSONL shards used when MLflow batch logging is enabled |

`on_error="raise"` is the default. The task stops at the first image error and marks the
summary failed. Already written records are retained. With `on_error="skip"`, image
decoding/validation errors are recorded and processing continues. A run with some failures
finishes as `completed_with_errors`; inspect the error count. If no image succeeds, the
task fails even in skip mode. Directory-access, model, tracking and output-write failures
stop the task rather than silently discarding data.

Reruns start from the beginning and replace result files; automatic resume is not implemented.
Use a fresh output directory for another run. Source images are never moved or changed.

## Python: consume predictions incrementally

```python
from fractal_detector import Detector, ImageFailure

detector = Detector.load("model.json")
for result in detector.predict_directory(
    "D:/datasets/incoming-images",
    recursive=True,
    workers=4,
    batch_size=64,
    on_error="skip",
):
    if isinstance(result, ImageFailure):
        print("Failed:", result.path, result.message)
        continue
    print(result.measurement.source, result.synthetic_probability)
```

This API returns an iterator. Do not wrap it in `list()` when bounded memory matters.
Skip mode yields `ImageFailure` records so the caller can persist them. For file output,
progress reporting and automatic tracking together, call `run_task(TaskConfig(...))`
with the same fields as the TOML configuration.

`iter_directory(path, recursive=True, extensions=(".png",))` yields `Sample` objects.
Pass your own iterable of samples to `detector.iter_predict(...)` when inputs come
from a database, queue, or another iterator.
For several directories/manifests, use `detector.predict_sources(["folder-a", "file.parquet"])`.
The single-directory `predict_directory` API discovers images only.

Read a large result file one line at a time too:

```python
import json

with open("outputs/incoming-predictions/predictions.jsonl", encoding="utf-8") as stream:
    for line in stream:
        result = json.loads(line)
        print(result["measurement"]["source"], result["label"])
```
