"""Comparison of data-processing approaches (requirement 9): time, peak memory, equal results.

Benchmark A - storage format, task "annual mean temperature per city" over
              ~10 M hourly rows:
              1) pandas, full CSV load (naive)
              2) pandas, CSV with chunksize + incremental aggregation
              3) pandas, Parquet (all columns)
              4) pandas, Parquet with column pruning (only 3 columns)
Benchmark B - processing engine, task "decade means + 10-year moving average"
              over the daily table: pandas vs SQL in SQLite.

Every approach runs in a fresh child process, so the measured peak memory
(max resident set size of that process; the baseline = interpreter + pandas
already imported, ~150 MB) is not polluted by the other approaches.
"""
import multiprocessing as mp
import sqlite3
import threading
import time

import numpy as np
import pandas as pd

import config
from src.utils import dir_size_mb, save_json


# ------------------------------------------------------------------ memory helper
class PeakMemory:
    """Track the peak resident memory (RSS) of the current process while a block runs.

    A background thread samples psutil every 10 ms. (ru_maxrss is not used:
    on Linux it is inherited from the parent across exec, which would make
    every child report the parent's peak.)
    """

    def __init__(self, interval: float = 0.01):
        import psutil
        self.proc = psutil.Process()
        self.interval = interval
        self.baseline_mb = self.peak_mb = self.proc.memory_info().rss / 1024 ** 2
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self.proc.memory_info().rss / 1024 ** 2)
            time.sleep(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, self.proc.memory_info().rss / 1024 ** 2)


# ------------------------------------------------------------------ benchmark A tasks
def _annual_from_hourly(df: pd.DataFrame) -> pd.DataFrame:
    year = pd.to_datetime(df["time"]).dt.year
    out = df.groupby([df["city"], year], observed=True)["temperature_2m"].mean()
    out = out.rename_axis(["city", "year"]).reset_index(name="t")
    out["city"] = out["city"].astype(str)
    return out


def a_csv_full():
    files = sorted(config.RAW_DIR.glob("*/*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return _annual_from_hourly(df)


def a_csv_chunked():
    sums, counts = {}, {}
    for f in sorted(config.RAW_DIR.glob("*/*.csv")):
        for chunk in pd.read_csv(f, usecols=["city", "time", "temperature_2m"], chunksize=500_000,
                                 dtype={"temperature_2m": "float32"}):
            key = [chunk["city"], chunk["time"].str.slice(0, 4).astype(int)]
            g = chunk.groupby(key)["temperature_2m"].agg(["sum", "count"])
            for k, row in g.iterrows():
                sums[k] = sums.get(k, 0.0) + row["sum"]
                counts[k] = counts.get(k, 0) + row["count"]
    rows = [{"city": c, "year": y, "t": sums[(c, y)] / counts[(c, y)]} for (c, y) in sums]
    return pd.DataFrame(rows)


def a_parquet_full():
    df = pd.read_parquet(config.HOURLY_PARQUET_DIR)
    return _annual_from_hourly(df)


def a_parquet_pruned():
    df = pd.read_parquet(config.HOURLY_PARQUET_DIR, columns=["city", "time", "temperature_2m"])
    return _annual_from_hourly(df)


# ------------------------------------------------------------------ benchmark B tasks
def b_pandas():
    d = pd.read_parquet(config.DAILY_PARQUET, columns=["city", "date", "tmean"])
    d = d[d["date"] < "2026-01-01"]
    d["year"] = d["date"].dt.year
    d["decade"] = d["year"] // 10 * 10
    dec = d.groupby(["city", "decade"], observed=True)["tmean"].mean().round(4).reset_index()
    ann = d.groupby(["city", "year"], observed=True)["tmean"].mean().reset_index()
    ann["ma10"] = ann.groupby("city", observed=True)["tmean"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    dec["city"] = dec["city"].astype(str)
    return dec.sort_values(["city", "decade"]).reset_index(drop=True), ann["ma10"].round(4).to_numpy()


def b_sql():
    conn = sqlite3.connect(config.DB_PATH)
    dec = pd.read_sql_query("""
        SELECT c.city_name AS city, (CAST(substr(d.date,1,4) AS INTEGER)/10)*10 AS decade,
               ROUND(AVG(d.tmean), 4) AS tmean
        FROM daily_weather d JOIN cities c USING (city_id)
        WHERE d.date < '2026-01-01'
        GROUP BY city, decade ORDER BY city, decade""", conn)
    ma = pd.read_sql_query("""
        WITH a AS (SELECT city_id, CAST(substr(date,1,4) AS INTEGER) AS year, AVG(tmean) AS t
                   FROM daily_weather WHERE date < '2026-01-01' GROUP BY city_id, year)
        SELECT ROUND(AVG(t) OVER (PARTITION BY city_id ORDER BY year ROWS BETWEEN 9 PRECEDING AND CURRENT ROW), 4) AS ma10
        FROM a ORDER BY city_id, year""", conn)
    conn.close()
    return dec, ma["ma10"].to_numpy()


TASKS = {
    "A1 pandas CSV full load": a_csv_full,
    "A2 pandas CSV chunksize": a_csv_chunked,
    "A3 pandas Parquet (all columns)": a_parquet_full,
    "A4 pandas Parquet (3 columns)": a_parquet_pruned,
    "B1 pandas (daily Parquet)": b_pandas,
    "B2 SQL (SQLite)": b_sql,
}


def _child(name: str, queue: mp.Queue) -> None:
    """Run one task in a child process and send back time, memory and result."""
    try:
        with PeakMemory() as mem:
            t0 = time.perf_counter()
            result = TASKS[name]()
            elapsed = time.perf_counter() - t0
        queue.put({"name": name, "seconds": elapsed, "baseline_mb": mem.baseline_mb, "peak_mb": mem.peak_mb,
                   "result": result})
    except Exception as exc:                 # report the error instead of leaving the parent waiting
        queue.put({"name": name, "error": f"{type(exc).__name__}: {exc}"})


def run_task(name: str, timeout_s: int = 1800) -> dict:
    """Run TASKS[name] in a fresh 'spawn' process (clean memory measurement)."""
    import queue as queue_mod
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(name, q))
    p.start()
    t0 = time.time()
    while True:
        try:
            out = q.get(timeout=2)
            break
        except queue_mod.Empty:
            if not p.is_alive():
                raise RuntimeError(f"benchmark process for {name!r} died (exit code {p.exitcode})")
            if time.time() - t0 > timeout_s:
                p.terminate()
                raise TimeoutError(f"benchmark {name!r} exceeded {timeout_s}s")
    p.join()
    if "error" in out:
        raise RuntimeError(f"benchmark {name!r} failed: {out['error']}")
    return out


def run_benchmarks(include_full_csv: bool = True) -> pd.DataFrame:
    """Run all benchmarks, check that results agree, save a table."""
    rows, results = [], {}
    has_raw = any(config.RAW_DIR.glob("*/*.csv"))
    if not has_raw:
        print("  no raw CSV in data/raw: CSV approaches (A1, A2) are skipped")
    for name in TASKS:
        if name.startswith("A1") and not include_full_csv:
            continue
        if name[:2] in ("A1", "A2") and not has_raw:
            continue
        print(f"  running {name} ...")
        out = run_task(name)
        results[name] = out["result"]
        rows.append({"approach": name, "seconds": round(out["seconds"], 2),
                     "peak_rss_mb": round(out["peak_mb"], 1),
                     "extra_over_baseline_mb": round(out["peak_mb"] - out["baseline_mb"], 1)})
        print(f"    {out['seconds']:.2f} s, peak RSS {out['peak_mb']:.0f} MB")
    bench = pd.DataFrame(rows)

    # ---- equality checks
    def annual_key(df):
        return df.assign(city=df["city"].astype(str)).sort_values(["city", "year"]).reset_index(drop=True)["t"].to_numpy()

    ref = annual_key(results["A4 pandas Parquet (3 columns)"])
    diffs = {}
    for name in [n for n in results if n.startswith("A")]:
        diffs[name] = float(np.nanmax(np.abs(annual_key(results[name]) - ref)))
    dec_p, ma_p = results["B1 pandas (daily Parquet)"]
    dec_s, ma_s = results["B2 SQL (SQLite)"]
    diffs["B1 pandas (daily Parquet)"] = 0.0
    diffs["B2 SQL (SQLite)"] = float(max(np.nanmax(np.abs(dec_p["tmean"].to_numpy() - dec_s["tmean"].to_numpy())),
                                         np.nanmax(np.abs(ma_p - ma_s))))
    bench["max_abs_diff_vs_reference"] = bench["approach"].map(diffs).round(5)
    bench["storage_mb"] = bench["approach"].map({
        "A1 pandas CSV full load": dir_size_mb(config.RAW_DIR),
        "A2 pandas CSV chunksize": dir_size_mb(config.RAW_DIR),
        "A3 pandas Parquet (all columns)": dir_size_mb(config.HOURLY_PARQUET_DIR),
        "A4 pandas Parquet (3 columns)": dir_size_mb(config.HOURLY_PARQUET_DIR),
        "B1 pandas (daily Parquet)": dir_size_mb(config.DAILY_PARQUET),
        "B2 SQL (SQLite)": dir_size_mb(config.DB_PATH),
    }).round(1)
    bench.to_csv(config.TABLE_DIR / "benchmarks.csv", index=False)
    save_json(bench.to_dict(orient="records"), config.TABLE_DIR / "benchmarks.json")
    print(bench.to_string(index=False))
    return bench
