import csv
import os
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from pathlib import Path

import pyarrow.parquet as pq

from fractal_detector.config import IMAGE_EXTENSIONS, DataSource, SourceSpec, validate_extensions
from fractal_detector.dataset import Sample, iter_directory


def _rows(path: Path, source: DataSource, batch_size: int) -> Iterator[dict[str, object]]:
    columns = (source.path_column, source.image_column, source.label_column, source.group_column)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if not fields or source.path_column not in fields:
                raise ValueError(
                    "CSV manifest requires a path column; label and group are optional"
                )
            if len(set(fields)) != len(fields) or set(fields) - set(columns):
                raise ValueError("manifest has unknown or duplicate columns")
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError("malformed manifest row")
                yield dict(row)
    elif path.suffix.lower() == ".parquet":
        with pq.ParquetFile(path) as parquet:
            fields = parquet.schema_arrow.names
            if len(set(fields)) != len(fields):
                raise ValueError("Parquet manifest has duplicate columns")
            if not ({source.path_column, source.image_column} & set(fields)):
                raise ValueError("Parquet requires a path or image column")
            selected = [c for c in columns if c in fields]
            for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
                yield from batch.to_pylist()
    else:
        raise ValueError("manifest must be CSV or Parquet")


def _sample(row: Mapping[str, object], path: Path, index: int, source: DataSource) -> Sample:
    location = row.get(source.path_column)
    image = row.get(source.image_column)
    if isinstance(image, dict):
        location = location or image.get("path")
        image = image.get("bytes")
    if image is not None and not isinstance(image, bytes):
        raise ValueError(f"image at row {index} must contain encoded bytes or a bytes/path struct")
    if image is None:
        if not isinstance(location, str) or not location.strip():
            raise ValueError(f"missing image path or bytes at row {index}")
        file = (path.parent / location).resolve()
        if not file.is_file():
            raise ValueError(f"image not found at row {index}: {file}")
        identity = str(file)
    else:
        identity = f"{path}#row={index}"
    value = row.get(source.label_column)
    text = "" if value is None else str(value).strip().lower()
    if isinstance(value, bool) or (text and text not in source.label_map):
        raise ValueError(
            f"label at row {index} must match label_map (default: real/0 or synthetic/1)"
        )
    label = source.label_map.get(text)
    if source.label is not None:
        if label is not None and label != source.label:
            raise ValueError(f"source label conflicts with row {index}")
        label = source.label
    group = row.get(source.group_column)
    if group is not None and (isinstance(group, bool) or not isinstance(group, (str, int))):
        raise ValueError(f"group at row {index} must be a string or integer")
    return Sample(
        path=identity,
        label=label,
        group=(str(group).strip() or None) if group is not None else None,
        image_bytes=image,
    )


def iter_manifest(
    path: str | Path,
    *,
    source: DataSource | None = None,
    require_labels: bool = False,
    batch_size: int = 64,
) -> Iterator[Sample]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    path = Path(path).resolve()
    source = source or DataSource(paths=(str(path),))
    count = 0
    for index, row in enumerate(_rows(path, source, batch_size)):
        sample = _sample(row, path, index, source)
        if require_labels and sample.label is None:
            raise ValueError(f"missing label at row {index} in {path}")
        count += 1
        yield sample
    if count == 0:
        raise ValueError(f"manifest is empty: {path}")


def iter_sources(
    sources: Sequence[SourceSpec | Path],
    *,
    base_dir: str | Path = ".",
    recursive: bool = True,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
    exclude_dirs: tuple[Path, ...] = (),
    require_labels: bool = False,
    batch_size: int = 64,
) -> Iterator[Sample]:
    if isinstance(sources, (str, bytes)) or not sources:
        raise ValueError("sources must be a nonempty sequence of paths or DataSource objects")
    validate_extensions(extensions)
    base = Path(base_dir).resolve()
    with closing(sqlite3.connect("")) as seen:
        seen.execute("CREATE TABLE seen (path TEXT PRIMARY KEY, label INTEGER, group_name TEXT)")
        for spec in sources:
            source = spec if isinstance(spec, DataSource) else DataSource(paths=(str(spec),))
            for raw_path in source.paths:
                path = (base / raw_path).resolve()
                files = (
                    (
                        Path(s.path)
                        for s in iter_directory(
                            path,
                            recursive=recursive,
                            extensions=extensions,
                            exclude_dirs=exclude_dirs,
                            include_manifests=True,
                        )
                    )
                    if path.is_dir()
                    else iter((path,))
                )
                for file in files:
                    if file.suffix.lower() in {".csv", ".parquet"}:
                        samples = iter_manifest(
                            file,
                            source=source,
                            require_labels=require_labels,
                            batch_size=batch_size,
                        )
                    elif file.suffix.lower() in extensions and file.is_file():
                        samples = iter((Sample(path=str(file), label=source.label),))
                    else:
                        raise ValueError(
                            f"input must be an existing image, directory, CSV, or Parquet: {file}"
                        )
                    for sample in samples:
                        if require_labels and sample.label is None:
                            raise ValueError(
                                f"missing label for {sample.path}; supply a labeled manifest or source label"
                            )
                        key = os.path.normcase(sample.path)
                        previous = seen.execute(
                            "SELECT label, group_name FROM seen WHERE path = ?", (key,)
                        ).fetchone()
                        if previous is not None:
                            if previous != (sample.label, sample.group):
                                raise ValueError(
                                    f"conflicting labels/groups for duplicate input: {sample.path}"
                                )
                            continue
                        seen.execute(
                            "INSERT INTO seen VALUES (?, ?, ?)", (key, sample.label, sample.group)
                        )
                        yield sample
