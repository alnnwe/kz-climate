"""Raw CSV -> cleaned, type-optimised Parquet + data quality audit + daily aggregates.

Steps
1. `measure_dtype_optimisation` - memory of one city's data with default
   pandas dtypes vs optimised dtypes (category, float32, datetime64).
2. `build_hourly_parquet` - reads raw CSV in chunks (chunksize), cleans every
   chunk and writes a Parquet dataset partitioned by city.
3. `audit_quality` - missing values, duplicates, time-index continuity,
   physical range violations and IQR outliers, per city.
4. `build_daily` - hourly -> daily (local time UTC+5) aggregation, city by city.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from src.utils import df_memory_mb, dir_size_mb, measure, save_json

# physically plausible limits used for cleaning (values outside -> NaN)
VALID_RANGES = {
    "temperature_2m": (-65, 55),
    "relative_humidity_2m": (0, 100),
    "surface_pressure": (750, 1100),
    "precipitation": (0, 200),
}

OPTIMISED_DTYPES = {c: "float32" for c in config.HOURLY_VARIABLES}


def raw_files(city: str | None = None) -> list:
    """Sorted list of raw CSV chunk files (all cities or one)."""
    pattern = f"{city}/*.csv" if city else "*/*.csv"
    return sorted(config.RAW_DIR.glob(pattern))


def read_raw_default(city: str) -> pd.DataFrame:
    """Read one city's raw CSVs with pandas defaults (object/float64)."""
    return pd.concat([pd.read_csv(f) for f in raw_files(city)], ignore_index=True)


def optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast: float64 -> float32, city -> category, time string -> datetime64."""
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], format="%Y-%m-%dT%H:%M")
    out["city"] = out["city"].astype(pd.CategoricalDtype(list(config.CITIES)))
    for col in config.HOURLY_VARIABLES:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def measure_dtype_optimisation(city: str = "Astana") -> dict:
    """Memory before/after dtype optimisation for one city (MB)."""
    df_default = read_raw_default(city)
    before = df_memory_mb(df_default)
    df_opt = optimise_dtypes(df_default)
    after = df_memory_mb(df_opt)
    n_cities = len({p.parent.name for p in raw_files()})
    result = {
        "city": city,
        "rows": len(df_default),
        "memory_default_mb": round(before, 1),
        "memory_optimised_mb": round(after, 1),
        "reduction_pct": round(100 * (1 - after / before), 1),
        "dtypes_default": {k: str(v) for k, v in df_default.dtypes.items()},
        "dtypes_optimised": {k: str(v) for k, v in df_opt.dtypes.items()},
        "estimated_full_default_mb": round(before * n_cities, 0),
        "estimated_full_optimised_mb": round(after * n_cities, 0),
    }
    print(f"  {city}: {before:.1f} MB -> {after:.1f} MB (-{result['reduction_pct']}%)")
    return result


def clean_chunk(chunk: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Optimise dtypes and replace physically impossible values by NaN."""
    chunk = optimise_dtypes(chunk)
    violations = {}
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in chunk:
            continue
        bad = (chunk[col] < lo) | (chunk[col] > hi)
        n_bad = int(bad.sum())
        if n_bad:
            chunk.loc[bad, col] = np.nan
        violations[col] = n_bad
    return chunk, violations


def build_hourly_parquet(chunksize: int = 250_000) -> dict:
    """Stream raw CSV -> cleaned Parquet partitioned by city.

    Only one chunk (chunksize rows) is in memory at a time. Duplicated
    timestamps are dropped within a city; short temperature gaps (<= 3 h) are
    linearly interpolated, longer gaps stay NaN.
    """
    stats = {"chunks": 0, "rows_in": 0, "rows_out": 0, "range_violations": {}, "duplicates_removed": 0}
    schema = None
    for city in config.CITIES:
        files = raw_files(city)
        if not files:
            print(f"  ! no raw data for {city}, skipped")
            continue
        city_dir = config.HOURLY_PARQUET_DIR / f"city={city}"
        city_dir.mkdir(parents=True, exist_ok=True)
        writer = None
        last_time = None
        tail = None                                        # last rows of the previous chunk (for interpolation)
        for f in files:
            for chunk in pd.read_csv(f, chunksize=chunksize):
                stats["chunks"] += 1
                stats["rows_in"] += len(chunk)
                chunk, viol = clean_chunk(chunk)
                for k, v in viol.items():
                    stats["range_violations"][k] = stats["range_violations"].get(k, 0) + v
                n0 = len(chunk)
                chunk = chunk.drop_duplicates(subset="time")
                if last_time is not None:                   # duplicates across chunk borders
                    chunk = chunk[chunk["time"] > last_time]
                stats["duplicates_removed"] += n0 - len(chunk)
                if chunk.empty:
                    continue
                # interpolate short temperature gaps (limit 3 hours)
                temp = chunk["temperature_2m"]
                if temp.isna().any():
                    joined = pd.concat([tail, temp]) if tail is not None else temp
                    chunk["temperature_2m"] = joined.interpolate(limit=3, limit_area="inside").iloc[-len(temp):].values
                tail = chunk["temperature_2m"].iloc[-3:]
                last_time = chunk["time"].iloc[-1]
                table = pa.Table.from_pandas(chunk.drop(columns="city"), preserve_index=False)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(city_dir / "part-0.parquet", schema, compression="zstd")
                writer.write_table(table.cast(schema))
                stats["rows_out"] += len(chunk)
        if writer is not None:
            writer.close()
        print(f"  {city}: written")
    stats["parquet_size_mb"] = round(dir_size_mb(config.HOURLY_PARQUET_DIR), 1)
    stats["raw_csv_size_mb"] = round(dir_size_mb(config.RAW_DIR), 1)
    return stats


def load_hourly(cities: list[str] | None = None, columns: list[str] | None = None,
                start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Read the partitioned hourly Parquet with optional city/column/time filters.

    Filters are pushed down to PyArrow, so only the needed row groups are read.
    """
    filters = []
    if cities:
        filters.append(("city", "in", list(cities)))
    if start:
        filters.append(("time", ">=", pd.Timestamp(start)))
    if end:
        filters.append(("time", "<=", pd.Timestamp(end)))
    df = pd.read_parquet(config.HOURLY_PARQUET_DIR, columns=columns, filters=filters or None)
    if "city" in df.columns:
        df["city"] = df["city"].astype(pd.CategoricalDtype(list(config.CITIES)))
    return df


def iqr_outliers(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Boolean mask of values outside [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (series < q1 - k * iqr) | (series > q3 + k * iqr)


def audit_quality() -> pd.DataFrame:
    """Per-city data quality report (saved to reports/tables/quality_audit.csv)."""
    rows = []
    for city in config.CITIES:
        df = load_hourly([city])
        if df.empty:
            continue
        df = df.sort_values("time")
        full_index = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
        step = df["time"].diff().dropna()
        # IQR outliers of temperature computed within each calendar month
        # (otherwise every cold winter hour would look like an outlier)
        month = df["time"].dt.month
        out_mask = df.groupby(month)["temperature_2m"].transform(lambda s: iqr_outliers(s))
        row = {
            "city": city,
            "rows": len(df),
            "first": df["time"].min(),
            "last": df["time"].max(),
            "expected_hours": len(full_index),
            "missing_timestamps": len(full_index) - df["time"].nunique(),
            "duplicate_timestamps": int(df["time"].duplicated().sum()),
            "irregular_steps": int((step != pd.Timedelta(hours=1)).sum()),
            "temp_iqr_outliers": int(out_mask.sum()),
            "temp_iqr_outliers_pct": round(100 * out_mask.mean(), 3),
            "temp_min": float(df["temperature_2m"].min()),
            "temp_max": float(df["temperature_2m"].max()),
        }
        for col in config.HOURLY_VARIABLES:
            row[f"na_{col}"] = int(df[col].isna().sum())
        rows.append(row)
    audit = pd.DataFrame(rows)
    audit.to_csv(config.TABLE_DIR / "quality_audit.csv", index=False)
    return audit


def dew_point_magnus(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    """Dew point (°C) from temperature and relative humidity, Magnus formula."""
    a, b = 17.62, 243.12
    gamma = np.log(rh.clip(lower=1) / 100.0) + a * temp_c / (b + temp_c)
    return (b * gamma / (a - gamma)).astype("float32")


def aggregate_city_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly -> daily for one city, using local days (UTC+5)."""
    df = df.copy()
    df["date"] = (df["time"] + pd.Timedelta(hours=config.LOCAL_UTC_OFFSET_HOURS)).dt.floor("D")
    df["dew_point"] = dew_point_magnus(df["temperature_2m"], df["relative_humidity_2m"])
    df["wet_hour"] = (df["precipitation"] >= 0.1).astype("float32")
    g = df.groupby("date")
    daily = pd.DataFrame({
        "tmean": g["temperature_2m"].mean(),
        "tmax": g["temperature_2m"].max(),
        "tmin": g["temperature_2m"].min(),
        "dew_point": g["dew_point"].mean(),
        "humidity": g["relative_humidity_2m"].mean(),
        "humidity_min": g["relative_humidity_2m"].min(),
        "precipitation": g["precipitation"].sum(min_count=1),
        "precip_hours": g["wet_hour"].sum(),
        "pressure": g["surface_pressure"].mean(),
        "pressure_range": g["surface_pressure"].max() - g["surface_pressure"].min(),
        "n_hours": g["temperature_2m"].count(),
    })
    # values at the last local hour of the day (23:00 local) - strong forecast features
    last = df[df["time"].dt.hour == (23 - config.LOCAL_UTC_OFFSET_HOURS) % 24].set_index("date")
    daily["t_23h"] = last["temperature_2m"].reindex(daily.index)
    daily["pressure_23h"] = last["surface_pressure"].reindex(daily.index)
    daily = daily[daily["n_hours"] >= 20]          # drop incomplete days (first/last local day)
    return daily.astype("float32").reset_index()


def build_daily() -> pd.DataFrame:
    """Daily table for all cities, saved to data/processed/daily.parquet."""
    parts = []
    for city in config.CITIES:
        hourly = load_hourly([city])
        if hourly.empty:
            continue
        d = aggregate_city_daily(hourly)
        d.insert(0, "city", city)
        parts.append(d)
    daily = pd.concat(parts, ignore_index=True)
    daily["city"] = daily["city"].astype(pd.CategoricalDtype(list(config.CITIES)))
    daily["n_hours"] = daily["n_hours"].astype("int8")
    daily.to_parquet(config.DAILY_PARQUET, index=False)
    print(f"  daily table: {len(daily):,} rows, {df_memory_mb(daily):.1f} MB in memory")
    return daily


def load_daily(cities: list[str] | None = None) -> pd.DataFrame:
    """Read the daily Parquet (optionally a subset of cities)."""
    try:
        df = pd.read_parquet(config.DAILY_PARQUET)
    except FileNotFoundError:
        raise FileNotFoundError("daily.parquet not found - run the preprocessing step first (menu item 2)")
    if cities:
        df = df[df["city"].isin(cities)]
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)


def run_preprocessing() -> dict:
    """Full preprocessing step used by the menu and the pipeline."""
    report = {}
    print("1) dtype optimisation (memory before/after)")
    report["dtype_optimisation"] = measure_dtype_optimisation()
    print("2) raw CSV -> Parquet (chunked)")
    timing = {}
    with measure("csv_to_parquet", timing):
        report["parquet_build"] = build_hourly_parquet()
    report["parquet_build"].update(timing["csv_to_parquet"])
    print("3) data quality audit")
    audit = audit_quality()
    report["audit_summary"] = {
        "cities": int(len(audit)),
        "total_rows": int(audit["rows"].sum()),
        "missing_timestamps": int(audit["missing_timestamps"].sum()),
        "duplicate_timestamps": int(audit["duplicate_timestamps"].sum()),
        "na_temperature": int(audit["na_temperature_2m"].sum()),
    }
    print(audit[["city", "rows", "missing_timestamps", "duplicate_timestamps", "temp_iqr_outliers_pct"]].to_string(index=False))
    print("4) daily aggregation")
    daily = build_daily()
    report["daily_rows"] = len(daily)
    save_json(report, config.TABLE_DIR / "preprocessing_report.json")
    return report
