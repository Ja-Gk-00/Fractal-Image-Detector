from collections import Counter
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from demo_data import Record, canonical_pair, materialize, select_balanced
from PIL import Image


def test_balanced_selection_keeps_related_records_together() -> None:
    records = []
    for index in range(500):
        paired = f"COCO2017_train/{index}"
        records.append(Record(f"{index}.jpg", None, 0, "COCO", paired))
        records.append(
            Record("fake.parquet", index, 1, f"generator-{index % 4}", f"fake-{index}", (paired,))
        )
    first = select_balanced(records, 200, 1337)
    assert first == select_balanced(records, 200, 1337)
    assert first != select_balanced(records, 200, 43)
    assert Counter((r.split, r.label) for r in first) == {
        ("train", 0): 120,
        ("train", 1): 120,
        ("validation", 0): 40,
        ("validation", 1): 40,
        ("test", 0): 40,
        ("test", 1): 40,
    }
    by_group: dict[str, set[str]] = {}
    for record in first:
        by_group.setdefault(record.group, set()).add(record.split)
    assert all(len(splits) == 1 for splits in by_group.values())
    assert any(sum(r.group == group for r in first) > 1 for group in by_group)
    for split in ("train", "validation", "test"):
        counts = Counter(r.generator for r in first if r.split == split and r.label == 1)
        assert len(counts) == 4
        assert max(counts.values()) - min(counts.values()) <= 1
    assert canonical_pair("COCO2017_train/7") == canonical_pair("COCO2017_train/000000000007.jpg")


@pytest.mark.parametrize("size", [0, 9, 11])
def test_invalid_balanced_size(size: int) -> None:
    with pytest.raises(ValueError, match="multiple of 5"):
        select_balanced([], size, 1337)


def test_insufficient_pool_is_explicit() -> None:
    with pytest.raises(ValueError, match="Not enough"):
        select_balanced([], 10, 1337)


def test_materialize_preserves_original_bytes_and_parquet_row(tmp_path: Path) -> None:
    payloads = []
    for color in ("red", "blue"):
        stream = BytesIO()
        Image.new("RGB", (32, 32), color=color).save(stream, format="PNG")
        payloads.append(stream.getvalue())
    real = tmp_path / "real.png"
    real.write_bytes(payloads[0])
    shard = tmp_path / "fake.parquet"
    pq.write_table(
        pa.table({"image": [{"bytes": p, "path": None} for p in payloads]}), shard, row_group_size=1
    )
    records = [
        Record(str(real), None, 0, "COCO", "real", split="test"),
        Record(str(shard), 1, 1, "generator", "fake", split="test"),
    ]
    result = materialize(records, tmp_path / "sample")
    assert [Path(record.path).read_bytes() for record in result] == payloads
    assert [(r.source, r.row) for r in result] == [(str(real), None), (str(shard), 1)]
    assert real.read_bytes() == payloads[0]
