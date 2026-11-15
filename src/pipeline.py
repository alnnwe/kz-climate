"""Pipeline steps used by both the console menu (main.py) and `python main.py --all`."""
from contextlib import closing

import pandas as pd

import config
from src import analysis, benchmarks, database, downloader, forecasting, preprocessing, sql_queries, visualization
from src.utils import load_json, save_json

SUMMARY_FILE = config.TABLE_DIR / "run_summary.json"


def _update_summary(key: str, value) -> None:
    """Keep a JSON summary of the last run (program state on disk)."""
    summary = load_json(SUMMARY_FILE, default={}) or {}
    summary[key] = value
    summary["updated_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    save_json(summary, SUMMARY_FILE)


def step_download(cities: list[str] | None = None) -> None:
    state = downloader.download_all(cities)
    n_files = sum(len(c.get("chunks", {})) for c in state.get("cities", {}).values())
    _update_summary("download", {"raw_files": n_files, "period": state.get("period")})


def step_preprocess() -> None:
    if not preprocessing.raw_files():
        if any(config.HOURLY_PARQUET_DIR.glob("city=*/*.parquet")) and config.DAILY_PARQUET.exists():
            print("  no raw CSV in data/raw - using the existing Parquet files in data/processed "
                  "(run menu item 1 to download the raw data and rebuild them)")
            return
        raise FileNotFoundError("no raw data in data/raw and no Parquet in data/processed - run the download (menu item 1)")
    report = preprocessing.run_preprocessing()
    _update_summary("preprocessing", {k: v for k, v in report.items() if k != "dtype_optimisation"})


def step_database(hourly_from: str = "2021-01-01") -> None:
    """Create schema and load cities, daily data, recent hourly data and (if present) events/forecasts."""
    daily = preprocessing.load_daily()
    with closing(database.get_connection()) as conn:
        database.create_schema(conn, drop=True)
        database.load_dimensions(conn)
        n_daily = database.load_daily_table(conn, daily)
        hourly = preprocessing.load_hourly(start=hourly_from)
        n_hourly = database.load_hourly_table(conn, hourly)
        events_path = config.TABLE_DIR / "extreme_events.csv"
        n_events = 0
        if events_path.exists():
            ev = pd.read_csv(events_path, parse_dates=["start_date", "end_date"])
            n_events = database.load_events(conn, ev)
        pred_path = config.PROCESSED_DIR / "forecast_predictions.parquet"
        metrics_path = config.TABLE_DIR / "forecast_metrics.csv"
        if pred_path.exists() and metrics_path.exists():
            database.save_forecasts(conn, pd.read_parquet(pred_path), pd.read_csv(metrics_path),
                                    forecasting.MODEL_INFO)
    counts = database.table_counts()
    print("  table rows:", counts)
    _update_summary("database", counts)


def step_sql() -> dict:
    res = sql_queries.run_all_queries()
    _update_summary("sql_queries", len(res))
    return res


def step_analysis() -> dict:
    daily = preprocessing.load_daily()
    res = analysis.run_analysis(daily)
    figs = visualization.make_analysis_figures(daily, res)
    # extreme events go to SQLite as well
    with closing(database.get_connection()) as conn:
        database.create_schema(conn)
        if conn.execute("SELECT COUNT(*) FROM cities").fetchone()[0] == 0:
            database.load_dimensions(conn)
        database.load_events(conn, res["events"])
    t = res["trends"]
    _update_summary("analysis", {
        "figures": len(figs),
        "mean_trend_1976_2025": round(float(t[t["period"] == "1976-2025"]["ols_per_decade"].mean()), 3),
        "events": int(len(res["events"])),
        "stl_strength": res["stl_strength"],
    })
    print(f"  {len(figs)} figures saved to {config.FIG_DIR}")
    return res


def step_forecast(tune: bool = True) -> dict:
    daily = preprocessing.load_daily()
    res = forecasting.run_forecasting(daily, tune=tune)
    visualization.make_forecast_figures(res)
    with closing(database.get_connection()) as conn:
        database.create_schema(conn)
        if conn.execute("SELECT COUNT(*) FROM cities").fetchone()[0] == 0:
            database.load_dimensions(conn)
        database.save_forecasts(conn, res["predictions"], res["metrics"], forecasting.MODEL_INFO)
    best = res["metrics"][res["metrics"]["city"] == "ALL"].sort_values("mae").iloc[0]
    _update_summary("forecast", {"best_model": best["model"], "best_mae": float(best["mae"])})
    return res


def step_benchmarks(include_full_csv: bool = True) -> pd.DataFrame:
    bench = benchmarks.run_benchmarks(include_full_csv)
    _update_summary("benchmarks", bench.to_dict(orient="records"))
    return bench


def run_all(skip_download: bool = False, tune: bool = True) -> None:
    """Full reproducible pipeline, in dependency order."""
    if not skip_download:
        print("\n### 1. Download"); step_download()
    print("\n### 2. Preprocessing"); step_preprocess()
    print("\n### 3. Analysis + figures"); step_analysis()
    print("\n### 4. Forecasting"); step_forecast(tune)
    print("\n### 5. Database"); step_database()
    print("\n### 6. SQL queries"); step_sql()
    print("\n### 7. Benchmarks"); step_benchmarks()
    print("\nDone. Results: reports/tables, reports/figures, data/kz_climate.db")
