# MLflow: a short tutorial

MLflow records your experiment and displays its parameters, metrics and output files.
The detector does the image processing; the MLflow server stores the results.

The first walkthrough works with the supplied demo, so you can try it before preparing
your own data. Its generated textures test the software; they do not measure real-world
AI-image detection accuracy.

## 1. Prepare the environment and demo data

Open a terminal in `Fractal-synthetic-image-detector`:

```console
uv sync --locked --extra mlflow
uv run --extra mlflow python examples/demo.py
```

This creates demo images and `outputs/demo/training/model.json`.
If you use the existing Windows virtual environment directly, the equivalent commands
below start with `.\.venv\Scripts\python.exe` or `.\.venv\Scripts\mlflow.exe`.

## 2. Start MLflow in terminal A

```console
uv run --extra mlflow mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Wait for the server to finish starting; the first launch can take a minute.
Keep this terminal running. Open **http://127.0.0.1:5000** in your browser.
The SQLite database and local artifacts persist after you stop the server with Ctrl+C.

## 3. Run detection in terminal B

Open another terminal in the same repository directory:

```console
uv run --extra mlflow fractal-detector run configs/demo-predict.toml
```

The supplied configuration selects the demo image directory and saved model. Its MLflow
section is:

```toml
[tracking]
enabled = true
uri = "http://127.0.0.1:5000"
experiment = "fractal-demo"
run_name = "directory-predictions"
image_logging = "full"
```

You have now run a detection task **with MLflow tracking**. Local results are also saved
under `outputs/mlflow-demo`.

## 4. Inspect the run

In the browser:

1. Open the **fractal-demo** experiment and select the newest run.
2. Check **Parameters** for preprocessing settings, batch size and worker count.
3. Check **Metrics** for `gray_dimension`, `residual_dimension`, `edge_dimension`, and
   `synthetic_probability`. Each step represents an image's index in the recorded input order.
4. Check **Artifacts → predictions** for per-image probabilities, explanations and full
   fractal measurements. **measurements** contains per-scale counts and fit quality.
5. Open **summary.json** for the processed and error counts. For a training task, open
   **model.json** and **holdout.json** for the fitted model and validation report.

Select multiple runs to compare experiments. Higher fractal dimension alone does not
mean an image is synthetic; inspect the learned contributions and independent evaluation.

## Use your own images

Copy `configs/predict-directory.toml`, set `input_dir`, `model` and a fresh `output_dir`,
then enable its `[tracking]` section. Run it with the same command as step 3, using your
new configuration path. For several directories or Parquet files, use
`configs/predict-multiple.toml`. For training, use labeled data and `task="train"`.
The [configuration guide](configuration.md) shows the complete process.

To supply train/validation/test explicitly, copy `configs/train-splits.toml`, edit its
paths and enable tracking. Run it with `uv run --extra mlflow fractal-detector run
configs/train-splits.toml`. MLflow shows separate `validation.*` and `test.*` metrics,
with `validation.json` and `test.json` reports. Only the train split fits the model.
See [data sources and splits](data.md) for the schemas and labeling options.

For large directories, select `image_logging="batch"`. MLflow then shows metrics such
as `batch.gray_dimension.mean` and stores complete per-image records under
**Artifacts → batches**. This keeps every dimension and explanation while avoiding
an individual artifact upload for every image. `progress.processed`, `progress.succeeded`
and `progress.failed` show cumulative counts. The local JSONL files remain available too.

## Optional: let MLflow launch the task

The walkthrough above launches the detector directly. The included `MLproject` also lets
you launch it using **MLflow Projects**. Keep the server from step 2 running.

In PowerShell, from the repository directory:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
uv run --extra mlflow mlflow run . --env-manager local --experiment-name fractal-demo -P config=configs/demo-predict.toml
```

In Bash:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uv run --extra mlflow mlflow run . --env-manager local --experiment-name fractal-demo -P config=configs/demo-predict.toml
```

`--env-manager local` uses the prepared uv environment, so no Conda setup is needed.
`-P config=...` chooses your task file. MLflow creates the run, starts the worker, and
marks the run finished or failed. The worker attaches all results to that same run.
The launcher's tracking URI, experiment and run ID take precedence over those in the
configuration. Tracking is enabled automatically for this launch mode.

Paths in the task file still resolve relative to that file. Run from the repository
directory when using a relative `config` path. To use a path containing spaces, quote
the complete parameter, for example `-P "config=D:/my experiments/task.toml"`.

See the official [MLflow Projects documentation](https://mlflow.org/docs/latest/ml/projects/)
for other launch environments.

## Quick fixes

| Problem | What to do |
| --- | --- |
| `uv` is not recognized | Install [uv](https://docs.astral.sh/uv/getting-started/installation/) or use the prepared `.venv` executables. |
| MLflow is missing | Run `uv sync --locked --extra mlflow`; keep `--extra mlflow` in the run commands. |
| Connection refused | Keep terminal A running and use the same port in the server command and configuration. |
| Demo model not found | Run step 1 before detection. |
| Your model's preprocessing conflicts | Remove `[features]` from the prediction configuration. |
| No matching images | Check `input_dir`, `recursive`, and `extensions`; relative paths start at the config file. |
| A task stops on an unreadable image | Use `on_error="skip"` for directory prediction/extraction and inspect `errors.jsonl`. |

For Python-only tracking, attaching to an existing run, or downloading models, see
[the API guide](usage.md#logging-and-mlflow).
