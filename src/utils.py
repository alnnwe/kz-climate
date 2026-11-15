"""Small helpers shared by all modules: timing, memory measurement, JSON I/O."""
import json
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


def df_memory_mb(df: pd.DataFrame) -> float:
    """Deep memory usage of a DataFrame in megabytes."""
    return float(df.memory_usage(deep=True).sum()) / 1024 ** 2


@contextmanager
def measure(label: str, results: dict | None = None):
    """Context manager that records wall time and peak Python memory.

    Usage:
        stats = {}
        with measure("csv_read", stats):
            ...
        stats["csv_read"] -> {"seconds": .., "peak_mb": ..}
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if results is not None:
            results[label] = {"seconds": round(elapsed, 3), "peak_mb": round(peak / 1024 ** 2, 1)}
        print(f"  [{label}] {elapsed:.2f} s, peak {peak / 1024 ** 2:.1f} MB")


def save_json(obj, path: Path) -> None:
    """Write an object to a pretty-printed UTF-8 JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path, default=None):
    """Read JSON; return `default` if the file is missing or corrupted."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as exc:
        print(f"  ! {path} is corrupted ({exc}); using default")
        return default


def dir_size_mb(path: Path) -> float:
    """Total size of all files under a directory (or one file), MB."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1024 ** 2
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024 ** 2
