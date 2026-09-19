# Technology stack

This is a CPU-based Python package. Python connects the workflow, C accelerates two
expensive operations, and established numerical libraries handle the remaining mathematics.
It runs locally on CPU, with an optional MLflow server for experiment tracking.

## Runtime: what processes an image?

| Technology | Its job in this package | Why it is used |
| --- | --- | --- |
| Python 3.14 / CPython | Public API, CLI, configuration, file processing, reporting | Provides typed interfaces and a standard C extension interface. The development interpreter is pinned to 3.14.7. |
| C and the CPython buffer API | Tile minima/maxima/masses and exhaustive affine range/domain matching | Runs the expensive inner loops in compiled code. Inputs are checked and copied before releasing the GIL, allowing image workers to overlap native computation. |
| NumPy | Image arrays, vectorized calculations, FFT, SVD, and fallback kernels | Gives a common array representation and compiled numerical operations. The FFT measures frequency patterns; the SVD removes dominant low-rank image structure. |
| SciPy | Gaussian smoothing and the logistic function | Supplies tested numerical routines for residual extraction and converting classifier scores to probabilities. |
| scikit-learn | StandardScaler, logistic regression, data splits, evaluation metrics | Fits a compact supervised classifier whose score can be explained exactly by its feature contributions. |
| Pillow | Image decoding, EXIF orientation, transparency, cropping and resizing | Supports common raster formats with one consistent preprocessing path. |
| Pydantic | Configuration, feature, prediction and model schemas | Rejects unknown fields, invalid geometry, nonfinite values and incompatible saved models early. |
| PyYAML | Reading YAML task files | `safe_load` reads configuration data without executing Python constructors. |
| PyArrow | Batched Parquet reading and column selection | Uses Apache Arrow's compiled reader for path manifests and encoded-image columns without loading the whole table. |

The same pipeline works when the optional C extension is absent. `backend="auto"`
selects the available implementation; `backend="native"` requires C, and
`backend="python"` selects the NumPy fallback. Cython, Numba and mypyc are not build
dependencies: the expensive loops already run in C or numerical-library code.

## Files, concurrency and experiments

| Technology | How it is used |
| --- | --- |
| Python `argparse` | Implements `fractal-detector run ...` and `doctor`. |
| Python `tomllib`, `json`, `csv` | Read TOML/JSON tasks and CSV manifests; write portable models and reports. |
| JSONL | Stores one JSON record per line. Large prediction files can be written and read incrementally. |
| Python `os.scandir` | Discovers directory entries lazily, including large flat folders, without building a list of all image paths. |
| Python `concurrent.futures` | Uses a bounded thread pool within each image batch. `workers` controls concurrency; `batch_size` controls retained results. |
| Python `logging` | Writes structured events, measurements, errors, and progress to JSONL. |
| SHA-256 (`hashlib`) | Identifies exact decoded images for duplicate/leakage checks in training and evaluation. |
| MLflow, optional | Stores experiment parameters, metrics, artifacts and run status. Its browser UI lets you inspect and compare runs. |
| MLflow Projects (`MLproject`) | Launches a task with `mlflow run`; the worker attaches results to the launcher's run. |
| SQLite | Stores local MLflow metadata and maintains a temporary disk-backed index of seen image paths when combining sources. |
| HTTP | Connects the detector and browser to a running MLflow tracking server. Image processing still executes in the detector process. |

The JSON model contains numeric weights and preprocessing settings; loading it does
not execute a pickle. MLflow is optional for prediction, including models downloaded
from MLflow. Its own dependencies, such as SQLAlchemy and pandas, appear transitively
in the lockfile; application code does not require you to configure them separately.

## Development, packaging and quality

The optional `notebooks` dependency group adds JupyterLab (editor), IPykernel (Python
cell execution), nbformat (notebook schema validation), nbclient (automated execution),
pandas (tables and sample summaries), and Matplotlib (plots and exported PNG figures).
Run `uv sync --locked --extra mlflow --group notebooks` to install it. From the project
directory, open `notebooks/coco_aigenbench_demo.ipynb` or run
`uv run --locked --extra mlflow --group notebooks python notebooks/execute.py`.
The notebook's adjacent README describes the expected local dataset layout.

| Tool | Purpose |
| --- | --- |
| uv and `uv.lock` | Install/manage the Python environment and lock exact dependency versions and hashes. |
| `pyproject.toml` and setuptools | Declare package metadata, dependencies, command-line entry points and C extension builds. |
| `build` | Create a wheel and source archive through Python's standard build interface. |
| Twine | Validate distribution metadata before release. |
| cibuildwheel | Build and test platform-specific wheels in the release workflow. |
| pytest | Run unit, mathematical, API, filesystem and MLflow integration tests. |
| Hypothesis | Generate varied inputs for C/Python agreement tests. |
| pytest-cov and coverage.py | Measure executed lines and branches; CI enforces a coverage threshold. |
| Ruff | Check Python style, imports, common mistakes and selected security rules; format code. |
| ty | Check type consistency in the package, tests, examples and benchmarks. |
| clang-format | Keep C source formatting consistent. |
| AddressSanitizer / UndefinedBehaviorSanitizer | Check the C extension for memory and undefined-behavior defects in Linux CI. |
| pre-commit | Run selected quality checks before a Git commit. |
| pip-audit | Check installed dependencies against public vulnerability records. |
| MkDocs | Build the documentation site and validate its internal links/navigation. |
| GitHub Actions | Run quality checks and tests across Windows, Linux and macOS; prepare releases. |
| actionlint | Validate GitHub Actions workflow definitions. Go is used to run this tool in CI. |
| CodeQL | Perform static security analysis of Python and C in GitHub. |
| Dependabot | Propose updates to Python dependencies and pinned action references. |
| PyPI trusted publishing / OIDC | Let an explicitly enabled release job publish without storing a long-lived PyPI token. |

No MLflow server or compiler is required simply to import and use a compatible installed
wheel. See [development](development.md) for commands and [the MLflow tutorial](mlflow.md)
for the optional experiment service.
