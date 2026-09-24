import hashlib
import json
import os
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from fractal_detector.config import FeatureConfig

type ImageInput = str | Path | bytes | NDArray[np.generic]
type FloatArray = NDArray[np.float64]


def atomic_json(path: str | Path, data: object) -> None:
    path = Path(path)
    payload = json.dumps(data, indent=2, allow_nan=False, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as f:
            temporary = Path(f.name)
            f.write(payload)
            f.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_image(
    image: ImageInput, config: FeatureConfig
) -> tuple[FloatArray, str, tuple[int, int], list[str]]:
    from fractal_detector.sizing import image_side, resolve_features

    config = resolve_features(
        config, image_side(image, config.max_input_pixels) if config.image_size == "auto" else 0
    )
    notes: list[str] = []
    if isinstance(image, (str, Path, bytes)):
        with Image.open(BytesIO(image) if isinstance(image, bytes) else image) as opened:
            if opened.width * opened.height > config.max_input_pixels:
                raise ValueError("image exceeds max_input_pixels")
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError("animated/multi-frame images are unsupported")
            oriented = ImageOps.exif_transpose(opened)
            if oriented.mode in {"I", "F"} or oriented.mode.startswith("I;16"):
                raise ValueError(
                    "convert high-bit-depth files to uint8 or normalized arrays explicitly"
                )
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                oriented = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
                notes.append("alpha composited over white")
            values = np.asarray(oriented.convert("RGB"), dtype=np.float64) / 255.0
    else:
        raw = np.asarray(image)
        if raw.ndim not in (2, 3) or (raw.ndim == 3 and raw.shape[2] not in (3, 4)):
            raise ValueError("expected (H,W), (H,W,3), or (H,W,4)")
        if raw.size == 0 or raw.shape[0] * raw.shape[1] > config.max_input_pixels:
            raise ValueError("empty image or max_input_pixels exceeded")
        if raw.dtype == np.uint8:
            values = raw.astype(np.float64) / 255.0
        elif np.issubdtype(raw.dtype, np.floating):
            values = raw.astype(np.float64)
        else:
            raise ValueError("arrays must be uint8 [0,255] or floating point [0,1]")
        if not np.all(np.isfinite(values)) or float(values.min()) < 0 or float(values.max()) > 1:
            raise ValueError("image values must be finite and in [0,1]")
        if values.ndim == 3 and values.shape[2] == 4:
            values = values[:, :, :3] * values[:, :, 3:] + (1 - values[:, :, 3:])
            notes.append("alpha composited over white")
    shape = (int(values.shape[0]), int(values.shape[1]))
    if min(shape) < 16:
        raise ValueError("images must be at least 16 pixels on each side")
    gray = values if values.ndim == 2 else values @ np.array([0.299, 0.587, 0.114])
    gray = np.ascontiguousarray(gray, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(shape).encode("ascii"))
    digest.update(gray.astype("<f8").tobytes())
    side = min(shape)
    top, left = (shape[0] - side) // 2, (shape[1] - side) // 2
    if shape[0] != shape[1]:
        notes.append("center crop removed borders to preserve aspect ratio")
    cropped = gray[top : top + side, left : left + side]
    if side != config.image_size:
        notes.append(
            f"resized from {side} to {config.image_size}; resizing can alter forensic traces"
        )
        resized = Image.fromarray(cropped.astype(np.float32)).resize(
            (config.image_size, config.image_size), Image.Resampling.BICUBIC
        )
        cropped = np.asarray(resized, dtype=np.float64).clip(0, 1)
    return np.ascontiguousarray(cropped), digest.hexdigest(), shape, notes


def source_name(image: ImageInput) -> str | None:
    return os.fspath(image) if isinstance(image, (str, Path)) else None
