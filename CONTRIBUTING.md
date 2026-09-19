# Contributing

Use the [development guide](docs/development.md) to install the locked environment and
run checks. Keep mathematical definitions in [the protocol](docs/methods.md) synchronized
with code. Changing feature meaning or order requires a model schema change.

For bug reports, include package/Python versions, the smallest relevant configuration,
the traceback, and a reproducible input when you can share it. State the backend and
whether the same input behaves differently with `backend="python"`.

Performance changes should include the benchmark parameters and parity results.
Detection-accuracy claims require independent labeled data and a documented split;
procedural test fixtures do not establish model quality.
