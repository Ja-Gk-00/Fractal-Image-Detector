# Choose and run a method

Start with `configs/research-selected.toml` for the strongest measured configuration
on the local seed-1337 sample. Use `configs/methods/` for the 36 standard/deep presets;
these now choose image size automatically. A larger analysis size is not itself proof
of better detection, and the measured ranking does not establish an ordering of
unbenchmarked neural models or isolated feature families.

## Measured ranking

The table below is generated from [the full results](research-results.json), with the
best validation ROC-AUC first. Each ranked file contains the exact feature settings
and best C for that candidate. Training uses 600 images and validation 200, balanced
between COCO and AI-GenBench. All random seeds are 1337. The comparison explicitly
fixes resolution at 128 or 256 pixels; the automatic-size presets are separate.

| Rank | Configuration | C | Features | Validation AUC | Accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | `configs/ranked/01-gaussian_256.toml` | 10 | 29 | 0.9084 | 0.820 |
| 2 | `configs/ranked/02-combined_256.toml` | 1 | 59 | 0.8928 | 0.825 |
| 3 | `configs/ranked/03-gaussian_multifractal.toml` | 1 | 29 | 0.8219 | 0.760 |
| 4 | `configs/ranked/04-legacy.toml` | 1 | 21 | 0.8135 | 0.720 |
| 5 | `configs/ranked/05-gaussian_sigma_0_6.toml` | 10 | 29 | 0.8087 | 0.705 |
| 6 | `configs/ranked/06-median_multifractal.toml` | 0.1 | 29 | 0.8085 | 0.745 |
| 7 | `configs/ranked/07-gaussian_sigma_1_5.toml` | 1 | 29 | 0.8078 | 0.730 |
| 8 | `configs/ranked/08-low_correlation.toml` | 0.1 | 30 | 0.7791 | 0.735 |
| 9 | `configs/ranked/09-pca_rank_8.toml` | 0.1 | 29 | 0.7751 | 0.690 |
| 10 | `configs/ranked/10-pca_rank_28.toml` | 0.1 | 29 | 0.7568 | 0.705 |


## Run any file

From the project directory:

```console
uv sync --locked --extra neural --extra mlflow
uv run fractal-detector run configs/research-selected.toml
uv run fractal-detector run configs/methods/compression-standard.toml
uv run --extra neural fractal-detector run configs/methods/dncnn_fractal-deep.toml
```

Before running, edit `[data]` in the selected file to point to your labeled manifests
or directories. Absolute paths are simplest. CSV rows need `path,label`; `group` is
recommended for related images. Labels are `real`/`0` and `synthetic`/`1`. Parquet also
supports embedded image bytes. See [data sources](data.md) for multiple paths and
labeled directory objects. Relative paths resolve from the TOML file's directory.

If uv is unavailable, install with `python -m pip install -e ".[neural,mlflow]"`,
then run `fractal-detector run <config.toml>` in that Python environment.

To write all paired presets with your paths already filled in:

```console
uv run python examples/method_configs.py --train D:/images/train.csv --validation D:/images/validation.parquet
```

This regenerates the 36 files in `configs/methods/`. Keep personal modifications in a
separate directory or pass `--output configs/my-methods`. Test data is optional and
should be added only after choosing a model; generator option `--test` adds it.

## Common configuration fields

```toml
task = "train"
output_dir = "../../outputs/my-experiment"
workers = "auto"

[data]
train = ["D:/images/train.csv"]
validation = ["D:/images/validation.parquet"]

[training]
seed = 1337
regularization_c = 1.0

[features]
compression = false
# Omit image_size for automatic sizing, or use image_size = 256.
```

| Field | Required? | Purpose |
| --- | --- | --- |
| `task` | Yes | `train`, `extract`, `predict`, `evaluate`, or `search` |
| Input source | Yes | Exactly one of `data`, `inputs`, `manifest`, `input_dir` |
| `data.train` | For explicit training | Labeled training sources; only these choose automatic training resolution |
| `data.validation` | For neural training/search | Validation sources; optional but recommended for ordinary linear training |
| `data.test` | No | Final evaluation sources, excluded from size selection and parameter search |
| `output_dir` | Recommended | Separate directory for each experiment |
| `workers` | No | Presets use `"auto"` to use available CPUs |
| `training.seed` | No | Defaults to 1337; explicit overrides remain possible |
| `training.regularization_c` | No | Linear classifier regularization; does not affect neural training |
| `features.image_size` | No | Automatic when omitted or `"auto"`; integer sets a fixed square side |
| `neural.image_size` | No | Same policy for the neural input, resolved independently of measurement size |
| `model` | For prediction/evaluation | Saved model file; preprocessing comes from that model |
| `split` | For prediction/evaluation/extraction with `data` | Which declared split to process |

All other settings have validated defaults. The method-specific fields below select
the intended algorithm; each preset already supplies them. The filename itself does
not select a method.

## All 18 method presets and their settings

For each prefix below, run
`fractal-detector run configs/methods/<prefix>-standard.toml` or replace `standard`
with `deep`. The order groups related methods; no accuracy ranking is claimed for
the methods that were not independently benchmarked.

| Prefix | Method-selecting fields in `[features]`, unless noted | Useful optional controls |
| --- | --- | --- |
| `combined` | `multifractal=true`, `low_correlation=true`, `compression=true`, `residual="svd"` | `pca_ranks`, `moment_orders`, `lacunarity_mode`, domain budget |
| `multifractal` | `families=["multifractal"]`, `multifractal=true`, `compression=false` | `moment_orders`, `mass_floor`, `scales`, residual filter |
| `pca_residual` | `residual="svd"`, `compression=false` | `svd_rank`, `scales`, `entropy_bins` |
| `gaussian_residual` | `residual="gaussian"`, `compression=false` | `gaussian_sigma`, `scales`, `entropy_bins` |
| `median_residual` | `residual="median"`, `compression=false` | Odd `median_size`, `scales`, `entropy_bins` |
| `low_correlation` | `families=["low_correlation"]`, `low_correlation=true`, `compression=false` | `pca_ranks`, `moment_orders`, `mass_floor`, `lacunarity_mode` |
| `gray_surface` | `families=["gray_surface"]`, `compression=false` | `scales`, `image_size` |
| `residual_surface` | `families=["residual_surface"]`, `compression=false` | `residual`, its filter/rank parameter, `scales` |
| `edge_box` | `families=["edge_box"]`, `compression=false` | `edge_quantile`, `scales` |
| `renyi` | `families=["renyi"]`, `compression=false` | `residual`, `scales`; orders remain 0, 1, 2 |
| `lacunarity` | `families=["lacunarity"]`, `compression=false` | `lacunarity_mode`, `lacunarity_stride`, `scales` |
| `statistics` | `families=["statistics"]`, `compression=false` | `residual`, `entropy_bins` |
| `spectral` | `families=["spectral"]`, `compression=false` | `residual`, filter/rank parameter, `image_size` |
| `compression` | `families=["compression"]`, `compression=true` | `block_size`, `stride`, `max_domains`, `s_clip`, `transform_ids` |
| `srm_fractal` | `[neural] architecture="srm_fractal"` | `channels`, `levels`, `backbone_blocks`, neural training controls |
| `median_fractal` | `[neural] architecture="residual_fractal"`, `residual="median"` | Odd `median_size`, `channels`, `levels` |
| `gaussian_fractal` | `[neural] architecture="residual_fractal"`, `residual="gaussian"` | `gaussian_sigma`, `median_size` (Gaussian kernel width), `channels`, `levels` |
| `dncnn_fractal` | `[neural] architecture="residual_fractal"`, `residual="dncnn"` | `dncnn_depth`, `dncnn_channels`, `denoiser_lr_ratio`, `channels`, `levels` |

Neural training additionally requires explicit train/validation splits and the
`neural` dependency extra. Common `[neural]` controls are `epochs`, `batch_size`,
`learning_rate`, `patience`, `weight_decay`, `label_smoothing`, `dropout`, `clip_norm`,
`restart_period`, `augment_probability`, `jpeg_min_quality`, `device`, and
`mixed_precision`. Set `device="cpu"`, `"cuda"`, or `"auto"`. Lower batch size or
explicitly cap image size when memory use is too high.

## Automatic image sizing

With `image_size` omitted, the package scans image headers and takes the **largest
height or width**. The existing square center-crop remains in place. It then rounds
the analysis side upward to meet the method's geometry. For example, a largest source
dimension of 641 becomes 672 with default box scales through 32. A 640-pixel maximum
remains 640. A minimum size may be needed to support all scales or PCA ranks.

- Handcrafted methods align to the least common multiple of the scales and block size.
- SRM networks align to their quadrant splits; residual networks also account for
  pooling between levels. Their input loader requires a multiple of eight.
- Training and parameter search inspect training images only. Validation and test
  use the stored training sizes. Prediction always uses the saved model's sizes.
- Standalone extraction uses the complete requested input set. A single-image API
  call uses that image; `extract_dataset` uses the whole sequence.
- Streaming extraction makes a first pass and stores sample records in a temporary
  SQLite file, then extracts in bounded batches. Embedded Parquet bytes are spooled
  to disk; they are not accumulated in RAM. The temporary file is cleaned up afterward.
- `features.max_input_pixels` remains a separate input safety limit, default 40 million.
  An automatically chosen square exceeding it fails with an explicit override hint.

Auto sizes and geometry alignment are recorded in `image_size_resolved` events.
Models and per-image diagnostics contain concrete numeric sizes; extraction also
writes `resolved-features.json`. MLflow neural runs log requested and resolved settings.
Set an integer `image_size` to skip automatic scanning and control computation cost.

```toml
[features]
image_size = 256
scales = [2, 4, 8, 16, 32, 64]

[neural]
image_size = 256
```

For neural prediction, include `[neural]` and set its `device`; the saved architecture
and sizes are authoritative. For linear prediction, omit `[neural]`. The
[MLflow guide](mlflow.md) explains tracking and Projects launches.
