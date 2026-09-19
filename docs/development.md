# Development

## Environment

Python 3.14.7 is the repository default. Direct dependency floors were checked against
PyPI releases on 2026-09-15; `uv.lock` records the resolved versions and hashes for
repeatable development. Package metadata uses lower bounds so compatible future releases
remain installable. Resolve updates deliberately and review the lockfile diff.
The notebook dependency group was resolved against PyPI on 2026-09-17.

```console
uv sync --locked --all-extras --group docs --group notebooks
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest --cov --cov-report=term-missing --cov-report=xml
uv run pytest notebooks/test_demo_data.py
uv run --group notebooks python notebooks/execute.py --check
uv run mkdocs build --strict
uv run python -m build
uv run twine check dist/*
uv run pip-audit --local --skip-editable
uv run python benchmarks/benchmark.py
```

Run the native formatting check with
`uv run clang-format --dry-run --Werror src/fractal_detector/_native.c`.
Install git hooks with `uv run pre-commit install` after initializing/cloning a Git
repository. Hooks use tools from the locked uv environment. The full MLflow tests use
temporary local SQLite stores and never require credentials or a network server.

## Layout

| Location | Responsibility |
| --- | --- |
| `src/fractal_detector/config.py`, `records.py` | Validated settings and result schemas |
| `io.py`, `dataset.py` | Image normalization, manifests, lazy directory traversal, threaded extraction |
| `sources.py` | CSV/Parquet batches, embedded images, multiple paths, disk-backed deduplication |
| `fractal.py`, `features.py` | Dimensions, residuals, mass/spectral features |
| `compression.py`, `kernels.py`, `_native.c` | Range/domain protocol and native boundary |
| `model.py` | Fitting, holdout evaluation, portable inference and explanations |
| `tracking.py`, `logging.py` | MLflow lifecycle and structured logging |
| `runner.py`, `streaming.py`, `cli.py` | Configuration execution, bounded batches, incremental reports, CLI |
| `project.py`, `MLproject` | MLflow Projects launch and attachment to its run |
| `tests/` | Mathematical oracles, native parity, input validation, model and MLflow integration |
| `configs/`, `examples/`, `benchmarks/` | Runnable configurations, offline demo, speed comparisons |
| `notebooks/` | Executed local-data walkthrough, balanced sampling helper, helper tests, notebook execution/validation |

Code documentation lives in this guide and the methods reference. Public signatures use
type annotations; short functions do not duplicate those signatures in boilerplate docstrings.

## Required checks

CI runs native tests on Windows, Linux, and macOS with Python 3.14, and separately installs
without native compilation. Ruff checks style, imports, common mistakes, and security rules;
ty checks package, tests, examples, benchmarks, and notebook Python helpers. Coverage includes branches and must
reach 90%. Known points, lines, planes, a Sierpinski carpet, analytical affine fits, and
randomized C/Python parity tests exercise the mathematical kernels. Python coverage does
not measure C lines; native buffer tests, property tests, and a Linux address/undefined-behavior
sanitizer job cover that boundary separately.

The workflow also checks its definitions with actionlint, C formatting, documentation links/build, wheel/sdist metadata,
installed-wheel imports outside the source tree, an isolated demo, and dependency
vulnerabilities. CodeQL scans Python and C. Actions are pinned by commit; Dependabot
updates actions and Python dependencies. A security scan needs GitHub code-scanning
availability for the repository. No local run can validate the hosted CI services themselves.

Keep package and feature protocol versions synchronized with persisted-model compatibility.
Do not weaken the coverage threshold or suppress new warnings to bypass regressions.
Add tests when a behavior or failure mode changes; keep synthetic fixtures explicitly
separate from real-world model validation data.

The saved notebook is schema-validated and checked for error outputs and unexecuted
cells in CI. Its sampler has small tests for label balance, group separation,
reproducibility, and byte-preserving image export. Re-executing the full notebook
requires local COCO / AI-GenBench data and is a separate command:
`uv run --locked --extra mlflow --group notebooks python notebooks/execute.py`.
The runner saves outputs only after every cell succeeds. See `notebooks/README.md`
for setup and the distinction between demonstration metrics and a research benchmark.

## Native development

After changing C sources, rebuild before running parity tests:

```console
uv run python setup.py build_ext --inplace
uv run pytest tests/test_kernels.py
```

For a mandatory fresh native install, set `FRACTAL_REQUIRE_NATIVE=1`, then
`uv sync --reinstall-package fractal-image-detector`. For a fallback install, use a fresh
virtual environment with `FRACTAL_BUILD_NATIVE=0`; an existing editable tree can retain
an already-built extension. CI isolates these installations in separate jobs.

Benchmarks warm up each operation and report median/minimum times, repetition count,
seed, platform, and Python version. They compare the same inputs and verify affine
outputs agree before reporting. Timing is informational, with no unstable wall-clock CI gate.

## Release preparation

The release workflow runs on a published GitHub release or manual dispatch. It builds
CPython 3.14 wheels with cibuildwheel on the three operating systems, runs installed-wheel
tests, and builds an sdist. Release artifacts can be downloaded without publishing to PyPI.

PyPI publication is a separate manually selected workflow-dispatch option, guarded by
the `pypi` GitHub environment and trusted publishing (OIDC). Before enabling it, configure
the environment, trusted publisher, package name ownership, and project licensing metadata.
Set the version in `pyproject.toml` and the package/model version declarations before a
release. The workflow configuration alone does not publish or deploy this workspace.
