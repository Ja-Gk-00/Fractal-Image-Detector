import os
from typing import Literal

type Workers = int | Literal["auto"]


def resolve_workers(workers: Workers) -> int:
    if workers == "auto":
        return max(1, os.process_cpu_count() or 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 1024:
        raise ValueError("workers must be 'auto' or an integer in 1..1024")
    return workers
