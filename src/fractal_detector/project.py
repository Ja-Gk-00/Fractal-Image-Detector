import importlib
import json
import os
import shlex
from pathlib import Path

from fractal_detector.config import load_config
from fractal_detector.runner import run_task


def run_project() -> dict[str, object]:
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        raise RuntimeError(
            "launch with: mlflow run . --env-manager local -P config=path/to/task.toml"
        )
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    tracking = importlib.import_module("mlflow.tracking")
    run = tracking.MlflowClient(tracking_uri=uri).get_run(run_id)
    parameter = run.data.params.get("config")
    if not parameter:
        raise ValueError("the MLflow Project run requires a config parameter")
    if os.name == "nt":
        if parameter.startswith('"') and parameter.endswith('"'):
            parameter = parameter[1:-1]
    elif parameter.startswith(("'", '"')):
        parts = shlex.split(parameter)
        if len(parts) != 1:
            raise ValueError("the MLflow Project config parameter must be a single path")
        parameter = parts[0]
    path = Path(parameter).resolve()
    config = load_config(path)
    attached = config.tracking.model_copy(update={"enabled": True, "uri": uri, "run_id": run_id})
    return run_task(config.model_copy(update={"tracking": attached}), base_dir=path.parent)


if __name__ == "__main__":
    print(json.dumps(run_project(), allow_nan=False))
