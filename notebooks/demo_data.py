import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


@dataclass(frozen=True)
class Record:
    source: str
    row: int | None
    label: int
    generator: str
    file_id: str
    pairs: tuple[str, ...] = ()
    group: str = ""
    split: str = ""
    path: str = ""


def canonical_pair(value: str) -> str:
    value = value.replace("\\", "/")
    prefix, _, name = value.rpartition("/")
    if prefix in {"COCO2017_train", "COCO2017_val"}:
        name = f"{int(Path(name).stem):012d}"
    return f"real:{prefix}/{name}"


def inventory(data_root: Path) -> tuple[list[Record], dict[str, int]]:
    records: list[Record] = []
    shards = sorted((data_root / "AI-GenBench-fake_part").rglob("*.parquet"))
    columns = ["label", "generator", "file_id", "paired_real_images"]
    for shard in shards:
        offset = 0
        with pq.ParquetFile(shard) as parquet:
            for batch in parquet.iter_batches(columns=columns, batch_size=4096):
                for row in batch.to_pylist():
                    if row["label"] != 1:
                        raise ValueError(f"Expected fake label 1 in {shard}, row {offset}")
                    records.append(
                        Record(
                            str(shard.resolve()),
                            offset,
                            1,
                            row["generator"],
                            row["file_id"],
                            tuple(row["paired_real_images"] or ()),
                        )
                    )
                    offset += 1
    fake_count = len(records)
    for folder, prefix in (("train2017", "COCO2017_train"), ("val2017", "COCO2017_val")):
        for path in sorted((data_root / "coco" / folder).rglob("*.jpg")):
            records.append(Record(str(path.resolve()), None, 0, "COCO", f"{prefix}/{path.stem}"))
    counts = {
        "real": len(records) - fake_count,
        "synthetic": fake_count,
        "parquet_shards": len(shards),
    }
    if not counts["real"] or not counts["synthetic"]:
        raise ValueError(
            f"Expected coco/train2017 or val2017 and AI-GenBench-fake_part under {data_root}"
        )
    return records, counts


def select_balanced(records: list[Record], per_class: int, seed: int) -> list[Record]:
    if per_class < 10 or per_class % 5:
        raise ValueError("per_class must be a multiple of 5, at least 10 (60/20/20 splits)")
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def identity(record: Record) -> str:
        return f"fake:{record.file_id}" if record.label else canonical_pair(record.file_id)

    for record in records:
        own = identity(record)
        for pair in record.pairs:
            a, b = find(own), find(canonical_pair(pair))
            parent[max(a, b)] = min(a, b)

    pools: dict[tuple[str, int], dict[str, list[Record]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        group = find(identity(record))
        bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) % 100
        split = "train" if bucket < 60 else "validation" if bucket < 80 else "test"
        pools[split, record.label][record.generator].append(
            replace(record, group=group, split=split)
        )

    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not cryptography
    selected: list[Record] = []
    targets = {"train": per_class * 3 // 5, "validation": per_class // 5, "test": per_class // 5}
    for split, target in targets.items():
        for label in (0, 1):
            by_generator = pools[split, label]
            generators = sorted(by_generator)
            rng.shuffle(generators)
            for name in generators:
                rng.shuffle(by_generator[name])
            chosen: list[Record] = []
            while len(chosen) < target:
                available = [name for name in generators if by_generator[name]]
                if not available:
                    raise ValueError(f"Not enough {split} label {label} samples for {target} rows")
                for name in available:
                    chosen.append(by_generator[name].pop())
                    if len(chosen) == target:
                        break
            selected.extend(chosen)
    return selected


def materialize(records: list[Record], destination: Path) -> list[Record]:
    destination.mkdir(parents=True, exist_ok=True)
    prepared: dict[tuple[str, int | None], Record] = {}
    fake_by_shard: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        if record.label:
            fake_by_shard[record.source].append(record)
        else:
            target = destination / "images" / record.split / "real" / Path(record.source).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.source, target)
            prepared[record.source, record.row] = replace(record, path=str(target.resolve()))

    formats = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tiff"}
    for number, (source, chosen) in enumerate(sorted(fake_by_shard.items()), start=1):
        wanted = {record.row: record for record in chosen}
        offset = 0
        with pq.ParquetFile(source) as parquet:
            for group_index in range(parquet.num_row_groups):
                count = parquet.metadata.row_group(group_index).num_rows
                indices = sorted(
                    i for i in wanted if i is not None and offset <= i < offset + count
                )
                if indices:
                    table = parquet.read_row_group(group_index, columns=["image"])
                    for index in indices:
                        record = wanted[index]
                        image = table["image"][index - offset].as_py()
                        payload = image["bytes"] if isinstance(image, dict) else image
                        if not isinstance(payload, bytes):
                            raise ValueError(f"Expected embedded image bytes: {source}#{index}")
                        with Image.open(BytesIO(payload)) as opened:
                            suffix = formats.get(opened.format or "")
                            if suffix is None:
                                raise ValueError(f"Unsupported sampled format: {opened.format}")
                        name = f"{Path(source).stem}-row-{index}{suffix}"
                        target = destination / "images" / record.split / "synthetic" / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(payload)
                        prepared[source, index] = replace(record, path=str(target.resolve()))
                    del table
                offset += count
        if number % 12 == 0 or number == len(fake_by_shard):
            print(f"Copied selected bytes from {number}/{len(fake_by_shard)} shards", flush=True)
    return [prepared[record.source, record.row] for record in records]


def prepare_sample(
    data_root: Path, output_root: Path, *, per_class: int = 500, seed: int = 1337
) -> dict:
    data_root = data_root.resolve()
    records, counts = inventory(data_root)
    selected = select_balanced(records, per_class, seed)
    sources = sorted({record.source for record in selected})
    signature = hashlib.sha256(
        json.dumps(
            [[p, Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in sources]
            + [[r.source, r.row, r.group, r.split] for r in selected]
        ).encode()
    ).hexdigest()[:16]
    destination = output_root.resolve() / f"sample-{seed}-{per_class}-{signature}"
    receipt = destination / "preparation.json"
    if receipt.exists():
        cached = json.loads(receipt.read_text(encoding="utf-8"))
        if all(Path(record["path"]).is_file() for record in cached["records"]):
            print(f"Reusing {len(selected)} prepared images in {destination.name}")
            return cached
    prepared = materialize(selected, destination)
    manifests = {}
    for split in ("train", "validation", "test"):
        rows = [
            {"path": r.path, "label": r.label, "group": r.group}
            for r in prepared
            if r.split == split
        ]
        csv_path = destination / f"{split}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "label", "group"])
            writer.writeheader()
            writer.writerows(rows)
        parquet_path = destination / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), parquet_path)
        manifests[split] = str(csv_path if split == "train" else parquet_path)
    summary = {
        "data_root": str(data_root),
        "directory": str(destination),
        "inventory": counts,
        "seed": seed,
        "per_class": per_class,
        "manifests": manifests,
        "records": [asdict(r) for r in prepared],
    }
    receipt.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
