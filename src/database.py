"""SQLite storage: normalised schema, loading, and query helpers.

Schema (see docs/er_diagram.md):

    regions 1--* cities 1--* daily_weather
                         1--* hourly_weather      (recent years only, optional)
                         1--* extreme_events *--1 event_types
                         1--* forecasts      *--1 models
                         1--* model_metrics  *--1 models
"""
import sqlite3
from contextlib import closing

import pandas as pd

import config
from src.utils import load_json

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS regions (
    region_id    INTEGER PRIMARY KEY,
    region_name  TEXT NOT NULL UNIQUE,
    macro_zone   TEXT NOT NULL CHECK (macro_zone IN ('North','South','East','West','Central'))
);

CREATE TABLE IF NOT EXISTS cities (
    city_id      INTEGER PRIMARY KEY,
    city_name    TEXT NOT NULL UNIQUE,
    region_id    INTEGER NOT NULL REFERENCES regions(region_id),
    latitude     REAL NOT NULL,
    longitude    REAL NOT NULL,
    grid_lat     REAL,           -- centre of the ERA5 cell returned by the API
    grid_lon     REAL,
    elevation_m  REAL
);

CREATE TABLE IF NOT EXISTS daily_weather (
    city_id        INTEGER NOT NULL REFERENCES cities(city_id),
    date           TEXT    NOT NULL,          -- ISO yyyy-mm-dd, local day (UTC+5)
    tmean REAL, tmax REAL, tmin REAL,
    dew_point REAL, humidity REAL, humidity_min REAL,
    precipitation REAL, precip_hours REAL,
    pressure REAL, pressure_range REAL,
    t_23h REAL, pressure_23h REAL,
    n_hours INTEGER CHECK (n_hours BETWEEN 0 AND 24),
    PRIMARY KEY (city_id, date)
);

CREATE TABLE IF NOT EXISTS hourly_weather (
    city_id      INTEGER NOT NULL REFERENCES cities(city_id),
    time_utc     TEXT    NOT NULL,
    temperature  REAL, humidity REAL, pressure REAL, precipitation REAL,
    PRIMARY KEY (city_id, time_utc)
);

CREATE TABLE IF NOT EXISTS event_types (
    event_type_id INTEGER PRIMARY KEY,
    code          TEXT NOT NULL UNIQUE,      -- heat_wave / cold_wave
    description   TEXT,
    rule          TEXT
);

CREATE TABLE IF NOT EXISTS extreme_events (
    event_id       INTEGER PRIMARY KEY,
    city_id        INTEGER NOT NULL REFERENCES cities(city_id),
    event_type_id  INTEGER NOT NULL REFERENCES event_types(event_type_id),
    start_date     TEXT NOT NULL,
    end_date       TEXT NOT NULL,
    duration_days  INTEGER NOT NULL,
    peak_value     REAL,                     -- max Tmax (heat) or min Tmin (cold)
    mean_exceedance REAL                     -- mean distance from the threshold, °C
);

CREATE TABLE IF NOT EXISTS models (
    model_id     INTEGER PRIMARY KEY,
    model_name   TEXT NOT NULL UNIQUE,
    family       TEXT,                       -- baseline / statistical / ml
    description  TEXT
);

CREATE TABLE IF NOT EXISTS forecasts (
    city_id   INTEGER NOT NULL REFERENCES cities(city_id),
    model_id  INTEGER NOT NULL REFERENCES models(model_id),
    date      TEXT NOT NULL,                 -- target day
    y_true    REAL,
    y_pred    REAL,
    PRIMARY KEY (city_id, model_id, date)
);

CREATE TABLE IF NOT EXISTS model_metrics (
    model_id      INTEGER NOT NULL REFERENCES models(model_id),
    city_id       INTEGER REFERENCES cities(city_id),   -- NULL = all cities
    split         TEXT NOT NULL,
    mae REAL, rmse REAL, smape REAL, mase REAL,
    train_seconds REAL, predict_seconds REAL
);

CREATE INDEX IF NOT EXISTS idx_daily_date   ON daily_weather(date);
CREATE INDEX IF NOT EXISTS idx_events_city  ON extreme_events(city_id, event_type_id);
CREATE INDEX IF NOT EXISTS idx_fc_model     ON forecasts(model_id, date);
"""


def get_connection() -> sqlite3.Connection:
    """Open the project database with foreign keys enabled."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection, drop: bool = False) -> None:
    """Create all tables (optionally dropping existing ones first)."""
    if drop:
        for t in ("model_metrics", "forecasts", "models", "extreme_events", "event_types",
                  "hourly_weather", "daily_weather", "cities", "regions"):
            conn.execute(f"DROP TABLE IF EXISTS {t}")
    conn.executescript(SCHEMA_SQL)


def load_dimensions(conn: sqlite3.Connection) -> dict:
    """Fill regions and cities. Returns {city_name: city_id}."""
    state = load_json(config.STATE_FILE, default={}) or {}
    grid = state.get("cities", {})
    regions = sorted({(m["region"], m["macro_zone"]) for m in config.CITIES.values()})
    conn.executemany("INSERT OR REPLACE INTO regions(region_id, region_name, macro_zone) VALUES (?,?,?)",
                     [(i + 1, r, z) for i, (r, z) in enumerate(regions)])
    region_ids = {r: i + 1 for i, (r, _) in enumerate(regions)}
    city_ids = {}
    for i, (name, m) in enumerate(config.CITIES.items(), start=1):
        g = grid.get(name, {})
        conn.execute(
            "INSERT OR REPLACE INTO cities VALUES (?,?,?,?,?,?,?,?)",
            (i, name, region_ids[m["region"]], m["lat"], m["lon"],
             g.get("grid_lat"), g.get("grid_lon"), g.get("elevation")),
        )
        city_ids[name] = i
    conn.commit()
    return city_ids


def city_id_map(conn: sqlite3.Connection) -> dict:
    """{city_name: city_id} read from the database."""
    return dict(conn.execute("SELECT city_name, city_id FROM cities").fetchall())


def load_daily_table(conn: sqlite3.Connection, daily: pd.DataFrame, chunksize: int = 50_000) -> int:
    """Insert the daily DataFrame into daily_weather (in chunks)."""
    ids = city_id_map(conn)
    df = daily.copy()
    df["city_id"] = df["city"].astype(str).map(ids)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    cols = ["city_id", "date", "tmean", "tmax", "tmin", "dew_point", "humidity", "humidity_min",
            "precipitation", "precip_hours", "pressure", "pressure_range", "t_23h", "pressure_23h", "n_hours"]
    conn.execute("DELETE FROM daily_weather")
    df[cols].to_sql("daily_weather", conn, if_exists="append", index=False, chunksize=chunksize)
    conn.commit()
    return len(df)


def load_hourly_table(conn: sqlite3.Connection, hourly: pd.DataFrame) -> int:
    """Insert (recent) hourly observations into hourly_weather."""
    ids = city_id_map(conn)
    df = pd.DataFrame({
        "city_id": hourly["city"].astype(str).map(ids),
        "time_utc": hourly["time"].dt.strftime("%Y-%m-%d %H:%M"),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["surface_pressure"],
        "precipitation": hourly["precipitation"],
    })
    conn.execute("DELETE FROM hourly_weather")
    df.to_sql("hourly_weather", conn, if_exists="append", index=False, chunksize=50_000)
    conn.commit()
    return len(df)


def load_events(conn: sqlite3.Connection, events: pd.DataFrame) -> int:
    """Insert heat/cold wave events (output of analysis.detect_waves)."""
    ids = city_id_map(conn)
    types = [
        (1, "heat_wave", "Heat wave",
         f">= {config.WAVE_MIN_DAYS} consecutive days with Tmax above the calendar-day "
         f"{int(config.HEAT_PCTL * 100)}th percentile of {config.BASELINE_PERIOD[0]}-{config.BASELINE_PERIOD[1]}"),
        (2, "cold_wave", "Cold wave",
         f">= {config.WAVE_MIN_DAYS} consecutive days with Tmin below the calendar-day "
         f"{int(config.COLD_PCTL * 100)}th percentile of {config.BASELINE_PERIOD[0]}-{config.BASELINE_PERIOD[1]}"),
    ]
    conn.executemany("INSERT OR REPLACE INTO event_types VALUES (?,?,?,?)", types)
    type_ids = {code: i for i, code, _, _ in types}
    conn.execute("DELETE FROM extreme_events")
    rows = [
        (ids[r.city], type_ids[r.event_type], r.start_date.strftime("%Y-%m-%d"), r.end_date.strftime("%Y-%m-%d"),
         int(r.duration_days), float(r.peak_value), float(r.mean_exceedance))
        for r in events.itertuples()
    ]
    conn.executemany(
        "INSERT INTO extreme_events(city_id, event_type_id, start_date, end_date, duration_days, peak_value, mean_exceedance) "
        "VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)


def save_forecasts(conn: sqlite3.Connection, preds: pd.DataFrame, metrics: pd.DataFrame,
                   model_info: dict) -> None:
    """Store test-period predictions and metrics of every model."""
    ids = city_id_map(conn)
    conn.execute("DELETE FROM forecasts")
    conn.execute("DELETE FROM model_metrics")
    conn.execute("DELETE FROM models")
    model_ids = {}
    for i, (name, (family, desc)) in enumerate(model_info.items(), start=1):
        conn.execute("INSERT INTO models VALUES (?,?,?,?)", (i, name, family, desc))
        model_ids[name] = i
    p = preds.copy()
    p["city_id"] = p["city"].astype(str).map(ids)
    p["model_id"] = p["model"].map(model_ids)
    p["date"] = pd.to_datetime(p["date"]).dt.strftime("%Y-%m-%d")
    p[["city_id", "model_id", "date", "y_true", "y_pred"]].to_sql(
        "forecasts", conn, if_exists="append", index=False, chunksize=50_000)
    m = metrics.copy()
    m["model_id"] = m["model"].map(model_ids)
    m["city_id"] = m["city"].map(lambda c: ids.get(c) if c != "ALL" else None)
    m[["model_id", "city_id", "split", "mae", "rmse", "smape", "mase", "train_seconds", "predict_seconds"]].to_sql(
        "model_metrics", conn, if_exists="append", index=False)
    conn.commit()


def run_query(sql: str, params: tuple | dict = ()) -> pd.DataFrame:
    """Execute a SQL query and return the result as a pandas DataFrame."""
    with closing(get_connection()) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def table_counts() -> dict:
    """Row count of every table (used in the menu and README)."""
    with closing(get_connection()) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
