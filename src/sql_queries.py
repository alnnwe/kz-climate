"""Analytical SQL queries (SQLite). Each entry: title, SQL, and which SQL features it shows.

All queries are executed by `run_all_queries`, which loads each result into a
pandas DataFrame and saves it to reports/tables/sql_<key>.csv.
"""
import time

import pandas as pd

import config
from src.database import run_query

QUERIES = {
    "q01_decade_means": {
        "title": "Mean annual temperature by city and decade",
        "features": "JOIN, GROUP BY, computed column",
        "sql": """
            SELECT c.city_name,
                   r.macro_zone,
                   (CAST(substr(d.date, 1, 4) AS INTEGER) / 10) * 10 AS decade,
                   ROUND(AVG(d.tmean), 2)  AS mean_temp,
                   ROUND(AVG(d.tmax), 2)   AS mean_tmax,
                   ROUND(AVG(d.tmin), 2)   AS mean_tmin,
                   COUNT(*)                AS n_days
            FROM daily_weather d
            JOIN cities  c ON c.city_id = d.city_id
            JOIN regions r ON r.region_id = c.region_id
            WHERE d.date < '2026-01-01'
            GROUP BY c.city_name, r.macro_zone, decade
            ORDER BY c.city_name, decade;
        """,
    },
    "q02_warming_rank": {
        "title": "Warming 1991-2020 vs 1961-1990 per city, ranked",
        "features": "CTE, conditional aggregation, RANK() window",
        "sql": """
            WITH period_means AS (
                SELECT city_id,
                       AVG(CASE WHEN date BETWEEN '1961-01-01' AND '1990-12-31' THEN tmean END) AS base_1961_1990,
                       AVG(CASE WHEN date BETWEEN '1991-01-01' AND '2020-12-31' THEN tmean END) AS norm_1991_2020,
                       AVG(CASE WHEN date BETWEEN '2016-01-01' AND '2025-12-31' THEN tmean END) AS last_decade
                FROM daily_weather
                GROUP BY city_id
            )
            SELECT c.city_name,
                   ROUND(p.base_1961_1990, 2) AS base_1961_1990,
                   ROUND(p.norm_1991_2020, 2) AS norm_1991_2020,
                   ROUND(p.last_decade, 2)    AS mean_2016_2025,
                   ROUND(p.norm_1991_2020 - p.base_1961_1990, 2) AS delta_norms,
                   ROUND(p.last_decade - p.base_1961_1990, 2)    AS delta_last_decade,
                   RANK() OVER (ORDER BY p.last_decade - p.base_1961_1990 DESC) AS warming_rank
            FROM period_means p
            JOIN cities c ON c.city_id = p.city_id
            ORDER BY warming_rank;
        """,
    },
    "q03_rolling_annual": {
        "title": "Annual mean temperature with 10-year moving average",
        "features": "CTE, AVG() OVER (ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)",
        "sql": """
            WITH annual AS (
                SELECT city_id, CAST(substr(date, 1, 4) AS INTEGER) AS year, AVG(tmean) AS t
                FROM daily_weather
                WHERE date < '2026-01-01'
                GROUP BY city_id, year
            )
            SELECT c.city_name, a.year, ROUND(a.t, 2) AS annual_mean,
                   ROUND(AVG(a.t) OVER (PARTITION BY a.city_id ORDER BY a.year
                                        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW), 2) AS ma10
            FROM annual a JOIN cities c USING (city_id)
            ORDER BY c.city_name, a.year;
        """,
    },
    "q04_record_days": {
        "title": "Hottest and coldest day ever recorded in each city",
        "features": "ROW_NUMBER() window in subquery, UNION ALL",
        "sql": """
            SELECT city_name, 'hottest' AS record, date, tmax AS value FROM (
                SELECT c.city_name, d.date, ROUND(d.tmax, 1) AS tmax,
                       ROW_NUMBER() OVER (PARTITION BY d.city_id ORDER BY d.tmax DESC) AS rn
                FROM daily_weather d JOIN cities c ON c.city_id = d.city_id
            ) WHERE rn = 1
            UNION ALL
            SELECT city_name, 'coldest', date, tmin FROM (
                SELECT c.city_name, d.date, ROUND(d.tmin, 1) AS tmin,
                       ROW_NUMBER() OVER (PARTITION BY d.city_id ORDER BY d.tmin ASC) AS rn
                FROM daily_weather d JOIN cities c ON c.city_id = d.city_id
            ) WHERE rn = 1
            ORDER BY city_name, record;
        """,
    },
    "q05_hot_days_having": {
        "title": "Cities with more than 100 days of Tmax >= 35 °C in 2011-2025",
        "features": "GROUP BY + HAVING",
        "sql": """
            SELECT c.city_name,
                   SUM(d.tmax >= 35) AS hot_days_35,
                   SUM(d.tmax >= 40) AS hot_days_40,
                   ROUND(MAX(d.tmax), 1) AS max_tmax
            FROM daily_weather d JOIN cities c ON c.city_id = d.city_id
            WHERE d.date BETWEEN '2011-01-01' AND '2025-12-31'
            GROUP BY c.city_name
            HAVING SUM(d.tmax >= 35) > 100
            ORDER BY hot_days_35 DESC;
        """,
    },
    "q06_yoy_lag": {
        "title": "Year-over-year change of annual mean (Astana, Almaty)",
        "features": "CTE, LAG() window",
        "sql": """
            WITH annual AS (
                SELECT c.city_name, CAST(substr(d.date, 1, 4) AS INTEGER) AS year, AVG(d.tmean) AS t
                FROM daily_weather d JOIN cities c ON c.city_id = d.city_id
                WHERE c.city_name IN ('Astana', 'Almaty') AND d.date < '2026-01-01'
                GROUP BY c.city_name, year
            )
            SELECT city_name, year, ROUND(t, 2) AS annual_mean,
                   ROUND(t - LAG(t) OVER (PARTITION BY city_name ORDER BY year), 2) AS yoy_change
            FROM annual
            ORDER BY city_name, year;
        """,
    },
    "q07_waves_by_decade_zone": {
        "title": "Heat and cold waves per decade and macro-zone",
        "features": "4-table JOIN, GROUP BY",
        "sql": """
            SELECT r.macro_zone,
                   (CAST(substr(e.start_date, 1, 4) AS INTEGER) / 10) * 10 AS decade,
                   SUM(t.code = 'heat_wave') AS heat_waves,
                   SUM(t.code = 'cold_wave') AS cold_waves,
                   ROUND(AVG(CASE WHEN t.code = 'heat_wave' THEN e.duration_days END), 1) AS avg_heat_duration,
                   COUNT(DISTINCT c.city_id) AS n_cities
            FROM extreme_events e
            JOIN event_types t ON t.event_type_id = e.event_type_id
            JOIN cities c      ON c.city_id = e.city_id
            JOIN regions r     ON r.region_id = c.region_id
            WHERE e.start_date < '2026-01-01'
            GROUP BY r.macro_zone, decade
            ORDER BY r.macro_zone, decade;
        """,
    },
    "q08_anomalous_cold_subquery": {
        "title": "January days at least 15 °C colder than the city's January norm",
        "features": "subquery in FROM (derived table) + JOIN; the correlated-subquery version took ~80 s, this one <1 s",
        "sql": """
            SELECT c.city_name,
                   COUNT(*) AS extreme_cold_days,
                   ROUND(n.jan_norm, 1) AS january_norm,
                   ROUND(MIN(d.tmean), 1) AS coldest_day_mean,
                   MAX(d.date) AS last_case
            FROM daily_weather d
            JOIN cities c ON c.city_id = d.city_id
            JOIN (SELECT city_id, AVG(tmean) AS jan_norm
                  FROM daily_weather
                  WHERE substr(date, 6, 2) = '01'
                  GROUP BY city_id) n ON n.city_id = d.city_id
            WHERE substr(d.date, 6, 2) = '01'
              AND d.tmean < n.jan_norm - 15
            GROUP BY c.city_name, n.jan_norm
            ORDER BY extreme_cold_days DESC;
        """,
    },
    "q09_frost_free_season": {
        "title": "Frost-free season length by decade (last spring / first autumn frost)",
        "features": "multiple CTEs, JOIN of CTEs, julianday arithmetic",
        "sql": """
            WITH frost AS (
                SELECT city_id, date, CAST(substr(date, 1, 4) AS INTEGER) AS year
                FROM daily_weather WHERE tmin < 0
            ),
            last_spring AS (
                SELECT city_id, year, MAX(date) AS last_frost
                FROM frost WHERE substr(date, 6, 2) < '07' GROUP BY city_id, year
            ),
            first_autumn AS (
                SELECT city_id, year, MIN(date) AS first_frost
                FROM frost WHERE substr(date, 6, 2) >= '07' GROUP BY city_id, year
            ),
            season AS (
                SELECT s.city_id, s.year,
                       julianday(a.first_frost) - julianday(s.last_frost) - 1 AS frost_free_days
                FROM last_spring s JOIN first_autumn a USING (city_id, year)
                WHERE s.year < 2026
            )
            SELECT c.city_name, (year / 10) * 10 AS decade,
                   ROUND(AVG(frost_free_days), 1) AS avg_frost_free_days
            FROM season JOIN cities c USING (city_id)
            GROUP BY c.city_name, decade
            ORDER BY c.city_name, decade;
        """,
    },
    "q10_longest_heat_wave": {
        "title": "Longest heat wave per city",
        "features": "JOIN + window ROW_NUMBER over events",
        "sql": """
            SELECT city_name, start_date, end_date, duration_days, peak_value FROM (
                SELECT c.city_name, e.start_date, e.end_date, e.duration_days, ROUND(e.peak_value, 1) AS peak_value,
                       ROW_NUMBER() OVER (PARTITION BY e.city_id
                                          ORDER BY e.duration_days DESC, e.peak_value DESC) AS rn
                FROM extreme_events e
                JOIN event_types t ON t.event_type_id = e.event_type_id AND t.code = 'heat_wave'
                JOIN cities c ON c.city_id = e.city_id
            ) WHERE rn = 1
            ORDER BY duration_days DESC;
        """,
    },
    "q11_model_leaderboard": {
        "title": "Forecast model leaderboard on the test period (all cities)",
        "features": "JOIN, aggregation, RANK() window, subquery for improvement over persistence",
        "sql": """
            WITH per_model AS (
                SELECT m.model_name, m.family,
                       AVG(ABS(f.y_pred - f.y_true)) AS mae,
                       SQRT(AVG((f.y_pred - f.y_true) * (f.y_pred - f.y_true))) AS rmse,
                       COUNT(*) AS n
                FROM forecasts f JOIN models m ON m.model_id = f.model_id
                GROUP BY m.model_name, m.family
            )
            SELECT model_name, family, ROUND(mae, 3) AS mae, ROUND(rmse, 3) AS rmse, n,
                   ROUND(100 * (1 - mae / (SELECT mae FROM per_model WHERE model_name = 'Persistence')), 1)
                       AS improvement_vs_persistence_pct,
                   RANK() OVER (ORDER BY mae) AS rank
            FROM per_model
            ORDER BY rank;
        """,
    },
    "q12_diurnal_cycle": {
        "title": "Mean diurnal temperature cycle (local hour) in January and July, recent hourly data",
        "features": "hourly table, GROUP BY on derived columns, CASE",
        "sql": """
            SELECT c.city_name,
                   (CAST(substr(h.time_utc, 12, 2) AS INTEGER) + 5) % 24 AS local_hour,
                   ROUND(AVG(CASE WHEN substr(h.time_utc, 6, 2) = '01' THEN h.temperature END), 2) AS jan_temp,
                   ROUND(AVG(CASE WHEN substr(h.time_utc, 6, 2) = '07' THEN h.temperature END), 2) AS jul_temp
            FROM hourly_weather h JOIN cities c ON c.city_id = h.city_id
            WHERE c.city_name IN ('Astana', 'Almaty')
            GROUP BY c.city_name, local_hour
            ORDER BY c.city_name, local_hour;
        """,
    },
}


def run_all_queries(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Run every query, save each result as CSV, return {key: DataFrame}."""
    results = {}
    timings = []
    for key, q in QUERIES.items():
        t0 = time.perf_counter()
        try:
            df = run_query(q["sql"])
        except Exception as exc:                      # e.g. table not loaded yet
            print(f"  ! {key} failed: {exc}")
            continue
        elapsed = time.perf_counter() - t0
        results[key] = df
        timings.append({"query": key, "title": q["title"], "features": q["features"],
                        "rows": len(df), "seconds": round(elapsed, 3)})
        df.to_csv(config.TABLE_DIR / f"sql_{key}.csv", index=False)
        if verbose:
            print(f"\n== {key}: {q['title']}  ({len(df)} rows, {elapsed:.2f}s)")
            print(df.head(8).to_string(index=False))
    pd.DataFrame(timings).to_csv(config.TABLE_DIR / "sql_query_summary.csv", index=False)
    return results
