import argparse
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate or execute the local-data demonstration."
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate structure and stored outputs only"
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    path = project / "notebooks" / "coco_aigenbench_demo.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    if not args.check:
        manager = KernelManager(kernel_name="python3")
        spec = manager.kernel_spec
        if spec is None:
            raise RuntimeError("Install the notebooks dependency group to provide a Python kernel")
        spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=1800,
            allow_errors=False,
            resources={"metadata": {"path": str(project)}},
            on_cell_start=lambda cell, cell_index: print(f"Cell {cell_index + 1}", flush=True),
        )
        client.execute(cleanup_kc=True)
        nbformat.write(notebook, path)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code or any(cell.execution_count is None for cell in code):
        raise ValueError("The demonstration must contain executed code cells")
    if any(output.output_type == "error" for cell in code for output in cell.outputs):
        raise ValueError("The demonstration contains an error output")
    print(f"Validated {len(code)} executed cells: {path}")


if __name__ == "__main__":
    main()
