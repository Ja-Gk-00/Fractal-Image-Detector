import argparse
import json
import sys
from collections.abc import Sequence

from fractal_detector.kernels import native_available
from fractal_detector.runner import run_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fractal measurements and supervised synthetic-image detection"
    )
    parser.add_argument("--version", action="version", version="fractal-image-detector 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="execute a TOML, YAML, or JSON task configuration")
    run.add_argument("config")
    sub.add_parser("doctor", help="report the available computation backend")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        print(
            json.dumps({"python": sys.version.split()[0], "native_available": native_available()})
        )
        return 0
    try:
        summary = run_config(args.config)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"fractal-detector: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, allow_nan=False))
    return 0
