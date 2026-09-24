# COCO / AI-GenBench demonstration

Open [coco_aigenbench_demo.ipynb](coco_aigenbench_demo.ipynb) to read the executed
walkthrough, including plots and results. It covers the original handcrafted feature families, native and
NumPy execution, input formats, configuration tasks, training, evaluation, explanations,
batch prediction, structured logging, and MLflow.

The new multifractal/PCA extensions, learned models and paired presets are documented
in [research methods](../docs/research-methods.md), with separate reproducible
[research experiments](../docs/research-results.md). The executed notebook preserves
seed 1337, matching the current research comparison. The notebook explicitly uses
128-pixel analysis; method presets choose their image size automatically.

## See the data results

Start at **section 9** for the measured results:

- **9.1:** Real/synthetic summary tables and distributions of nine key measurements,
  including all three fractal dimensions, entropy, lacunarity, and compression error.
- **9.2:** Dimension fit quality (R² and slope error), original image resolution,
  and encoded file-size comparisons.
- **9.3:** Four held-out real and four synthetic examples, selected with a fixed seed,
  showing ground truth, predicted class, probability, dimensions, and texture statistics.
- **9.4:** False-positive/false-negative image examples, probability distributions,
  and a dimension scatter plot marking incorrect predictions.
- **10:** Exact feature contributions for one real and one synthetic example.

Each plot states whether it uses the complete balanced sample or only held-out test
images. Galleries use test images; test outcomes do not choose model settings.
Summary statistics use valid dimension estimates, with undefined values counted
separately. Class differences describe this dataset mix, not universal rules for
distinguishing real and generated images.

Plots are embedded in the notebook and exported as PNG files. Per-image statistics,
class summaries, example measurements, and contributions are also saved as CSV files
in the run directory. The **python-api-explanation** MLflow run contains these under
the **figures/** and **statistics/** artifact folders.

## Launch

From the project directory:

```console
uv sync --locked --extra mlflow --group notebooks
uv run --locked --extra mlflow --group notebooks jupyter lab notebooks/coco_aigenbench_demo.ipynb
```

In VS Code, open the notebook and select `.venv` as its Python kernel. If `uv` is not
on PATH but the environment is already installed, use this Windows command:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab notebooks/coco_aigenbench_demo.ipynb
```

Use **Restart Kernel and Run All**. For execution without a browser, with failures
stopping the run and successful outputs saved back into the notebook:

```console
uv run --locked --extra mlflow --group notebooks python notebooks/execute.py
```

The runner uses its own Python interpreter for the kernel. It validates every cell
and refuses error outputs or missing execution counts. CI checks the saved notebook
without requiring the private local datasets:

```console
uv run --locked --group notebooks python notebooks/execute.py --check
uv run --locked pytest notebooks/test_demo_data.py
```

## Data and sample size

The default layout is relative to the project:

```text
../data/
  coco/
    train2017/*.jpg
    val2017/*.jpg
  AI-GenBench-fake_part/
    data/*.parquet
```

Edit `DATA_DIR` in the first code cell, or set `FRACTAL_DEMO_DATA`. The notebook reads
local data only. Its default sample is **500 real + 500 fake**, with 300/100/100 images
of each class in train/validation/test. Change `PER_CLASS` to a multiple of 5, provided
enough samples are available in each pool.

`demo_data.py` reads shard metadata, joins known `paired_real_images` relationships,
assigns connected groups to seeded split pools, and samples across available generators.
This custom split differs from the official benchmark protocol. Exact decoded
duplicates are rejected during training; unknown links and near duplicates are not
automatically found. The detector's preprocessing and threshold remain fixed.

The first preparation reads image row groups from many large shards, which can take
several minutes. It copies only selected encoded images, without recompression, into
`outputs/notebook-demo/sample-*`. Preparation metadata stores original source paths,
shard row indices, labels, generators, groups, and split assignments. Unchanged samples
are cached; selected source paths, sizes, modification times, and selection records
contribute to the cache key. Original datasets are not modified.

Each execution creates a fresh `outputs/notebook-demo/run-*` directory with the model,
task files, predictions, evaluation reports, logs, figures, and `notebook-summary.json`.
The sample and supervised features are held in memory; increase the sample cautiously.
Large prediction/extraction examples use bounded batches and JSONL outputs.

## View MLflow results

The notebook logs directly to a local SQLite store, so no server is needed during
execution. Afterward, run this from the project directory:

```console
uv run --locked --extra mlflow mlflow server --backend-store-uri sqlite:///outputs/notebook-demo/mlflow.db --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000> and select **coco-aigenbench-notebook**. Training runs
contain separate validation/test metrics and complete measurements. Prediction runs
contain batch artifacts with probabilities, dimensions, and explanations. The notebook
also prints the exact MLflow Projects command for its generated training configuration.

## Interpretation

This sample mixes COCO photographs with fake images from multiple source collections.
Content, encoding, and resolution differences can influence the learned scores. The
demonstration shows a working workflow, not a general detector accuracy claim. Keep
test data separate when selecting settings, and use source/generator holdouts for
research conclusions.

JupyterLab provides the editor, IPykernel runs Python cells, nbformat validates the
document, nbclient executes it, pandas renders tables, and Matplotlib creates plots.
Their exact versions are locked in the `notebooks` dependency group. See the official
[JupyterLab guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
and [nbclient documentation](https://nbclient.readthedocs.io/en/latest/) for those tools.
