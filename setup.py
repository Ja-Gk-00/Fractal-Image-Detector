import os
import sys

from setuptools import Extension, setup

native = os.environ.get("FRACTAL_BUILD_NATIVE", "1") != "0"
required = os.environ.get("FRACTAL_REQUIRE_NATIVE", "0") == "1"
if required and not native:
    raise ValueError("FRACTAL_REQUIRE_NATIVE conflicts with FRACTAL_BUILD_NATIVE=0")

setup(
    ext_modules=[
        Extension(
            "fractal_detector._native",
            sources=["src/fractal_detector/_native.c"],
            extra_compile_args=["/O2", "/W4"]
            if sys.platform == "win32"
            else ["-O3", "-Wall", "-Wextra"],
            optional=not required,
        )
    ]
    if native
    else [],
)
