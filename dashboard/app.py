"""Streamlit dashboard: climate and weather of Kazakhstan cities.

Run from the project root:   streamlit run dashboard/app.py
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.sql_queries import QUERIES  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#b9b8b2"
DIVERGING = ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3a3a3", "#e34948", "#9e1f1f"]

st.set_page_config(page_title="KZ Climate", layout="wide")


# ------------------------------------------------------------------ data loading (cached)
@st.cache_data
def load_daily() -> pd.DataFrame:
    d = pd.read_parquet(config.DAILY_PARQUET)
    d["date"] = pd.to_datetime(d["date"])
    d["city"] = d["city"].astype(str)
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    return d


@st.cache_data
def load_table(name: str, **kw) -> pd.DataFrame:
    path = config.TABLE_DIR / name
    return pd.read_csv(path, **kw) if path.exists() else pd.DataFrame()


@st.cache_data
def load_predictions() -> pd.DataFrame:
    path = config.PROCESSED_DIR / "forecast_predictions.parquet"
    if not path.exists():
        return pd.DataFrame()
    p = pd.read_parquet(path)
    p["date"] = pd.to_datetime(p["date"])
    return p


@st.cache_data
def run_sql(sql: str) -> pd.DataFrame:
    with sqlite3.connect(config.DB_PATH) as conn:
        return pd.read_sql_query(sql, conn)


def colour_map(cities: list[str]) -> dict:
    """Fixed colour per city (by its position in the config), grey beyond 8."""
    order = list(config.CITIES)
    return {c: SERIES[order.index(c)] if order.index(c) < len(SERIES) else GREY for c in cities}


try:
    daily = load_daily()
except FileNotFoundError:
    st.error("data/processed/daily.parquet not found. Run `python main.py --all --skip-download` first.")
    st.stop()

# ------------------------------------------------------------------ sidebar filters
st.sidebar.header("Filters")
zones = sorted({m["macro_zone"] for m in config.CITIES.values()})
sel_zones = st.sidebar.multiselect("Macro-region", zones, default=zones)
zone_cities = [c for c, m in config.CITIES.items() if m["macro_zone"] in sel_zones]
sel_cities = st.sidebar.multiselect("Cities", zone_cities,
                                    default=[c for c in ("Astana", "Almaty") if c in zone_cities] or zone_cities[:2])
y_min, y_max = int(daily["year"].min()), int(daily["year"].max())
years = st.sidebar.slider("Years", y_min, y_max, (1960, y_max))
variable = st.sidebar.selectbox("Variable", ["tmean", "tmax", "tmin", "precipitation", "humidity", "pressure"],
                                format_func=lambda v: {"tmean": "Mean temperature, °C", "tmax": "Max temperature, °C",
                                                       "tmin": "Min temperature, °C", "precipitation": "Precipitation, mm",
                                                       "humidity": "Relative humidity, %", "pressure": "Surface pressure, hPa"}[v])
if not sel_cities:
    st.warning("Select at least one city in the sidebar.")
    st.stop()
if len(sel_cities) > 8:
    st.sidebar.info("More than 8 cities: extra cities are drawn in grey.")

d = daily[daily["city"].isin(sel_cities) & daily["year"].between(*years)]
cmap = colour_map(sel_cities)

st.title("Climate and Weather of Kazakhstan Cities")
st.caption(f"ERA5 reanalysis via Open-Meteo, hourly data {y_min}-{daily['date'].max():%Y-%m-%d}, "
           f"{daily['city'].nunique()} cities, {len(daily):,} city-days")

tab_over, tab_trend, tab_ext, tab_fc, tab_sql, tab_quality = st.tabs(
    ["Overview", "Trends", "Extremes", "Forecast", "SQL", "Data quality"])

# ------------------------------------------------------------------ overview
with tab_over:
    trends = load_table("warming_trends.csv")
    if not trends.empty:
        t = trends[(trends["period"] == "1976-2025") & trends["city"].isin(sel_cities)]
        cols = st.columns(min(4, len(t)) or 1)
        for i, row in enumerate(t.sort_values("ols_per_decade", ascending=False).head(4).itertuples()):
            cols[i].metric(row.city, f"+{row.ols_per_decade:.2f} °C / decade", "trend 1976-2025", delta_color="off")
    full = d[d["year"] <= 2025]
    annual = full.groupby(["city", "year"])[variable].agg("sum" if variable == "precipitation" else "mean").reset_index()
    annual["ma10"] = annual.groupby("city")[variable].transform(lambda s: s.rolling(10, min_periods=5).mean())
    fig = go.Figure()
    for city in sel_cities:
        g = annual[annual["city"] == city]
        fig.add_trace(go.Scatter(x=g["year"], y=g[variable].round(2), mode="lines", name=f"{city} annual",
                                 line=dict(color=cmap[city], width=1), opacity=0.45, showlegend=False))
        fig.add_trace(go.Scatter(x=g["year"], y=g["ma10"].round(2), mode="lines", name=city,
                                 line=dict(color=cmap[city], width=3)))
    fig.update_layout(title=f"Annual {variable} with 10-year moving average", hovermode="x unified",
                      template="plotly_white", height=430, yaxis_title=variable)
    st.plotly_chart(fig, width="stretch")

    monthly = d.groupby(["city", "month"])[variable].mean().reset_index()
    fig = px.line(monthly, x="month", y=variable, color="city", markers=True, color_discrete_map=cmap,
                  title=f"Seasonal cycle ({years[0]}-{years[1]})", template="plotly_white")
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)),
                     ticktext=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ trends
with tab_trend:
    ann = load_table("annual_by_city.csv")
    if not ann.empty:
        a = ann[ann["city"].isin(sel_cities) & ann["year"].between(*years)]
        heat = a.pivot(index="city", columns="year", values="anomaly")
        lim = float(np.nanmax(np.abs(heat.values)))
        fig = px.imshow(heat, color_continuous_scale=DIVERGING, zmin=-lim, zmax=lim, aspect="auto",
                        labels=dict(color="°C"), title="Warming stripes: annual anomaly vs 1961-1990")
        st.plotly_chart(fig, width="stretch")
        dec = a.groupby(["city", "decade"])["anomaly"].mean().reset_index()
        fig = px.bar(dec, x="decade", y="anomaly", color="city", barmode="group", color_discrete_map=cmap,
                     template="plotly_white", title="Mean anomaly by decade (2020s = 2020-2025)")
        st.plotly_chart(fig, width="stretch")
    st.subheader("Seasonal warming rate 1976-2025, °C per decade")
    st.dataframe(load_table("seasonal_trends_1976_2025.csv").query("city in @sel_cities"), hide_index=True)
    st.subheader("Trend statistics")
    st.dataframe(load_table("warming_trends.csv").query("city in @sel_cities"), hide_index=True)

# ------------------------------------------------------------------ extremes
with tab_ext:
    ev = load_table("extreme_events.csv", parse_dates=["start_date", "end_date"])
    if ev.empty:
        st.info("Run the analysis step to detect heat and cold waves.")
    else:
        e = ev[ev["city"].isin(sel_cities) & ev["start_date"].dt.year.between(*years)]
        c1, c2, c3 = st.columns(3)
        c1.metric("Heat waves", int((e["event_type"] == "heat_wave").sum()))
        c2.metric("Cold waves", int((e["event_type"] == "cold_wave").sum()))
        c3.metric("Longest wave, days", int(e["duration_days"].max()) if len(e) else 0)
        cnt = e.groupby(["decade", "event_type"]).size().reset_index(name="events")
        fig = px.bar(cnt, x="decade", y="events", color="event_type", barmode="group", template="plotly_white",
                     color_discrete_map={"heat_wave": SERIES[1], "cold_wave": SERIES[0]},
                     title="Heat and cold waves per decade (selected cities)")
        st.plotly_chart(fig, width="stretch")
        idx = load_table("climate_indices_annual.csv")
        index_name = st.selectbox("Climate index", ["hot_days_35", "summer_days", "tropical_nights", "frost_days",
                                                   "ice_days", "severe_cold_days", "heavy_precip_days"])
        ii = idx[idx["city"].isin(sel_cities) & idx["year"].between(*years)]
        fig = px.line(ii, x="year", y=index_name, color="city", color_discrete_map=cmap, template="plotly_white",
                      title=f"{index_name} per year")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(e.sort_values("duration_days", ascending=False).head(50), hide_index=True)

# ------------------------------------------------------------------ forecast
with tab_fc:
    preds = load_predictions()
    metrics = load_table("forecast_metrics.csv")
    if preds.empty:
        st.info("Run the forecasting step first.")
    else:
        city = st.selectbox("City", sel_cities)
        models = st.multiselect("Models", sorted(preds["model"].unique()),
                                default=["Persistence", "HistGB + spatial"])
        p = preds[(preds["city"] == city) & preds["model"].isin(models)]
        dmin, dmax = p["date"].min().date(), p["date"].max().date()
        rng = st.slider("Period", dmin, dmax, (max(dmin, dmax - pd.Timedelta(days=180)), dmax))
        p = p[(p["date"].dt.date >= rng[0]) & (p["date"].dt.date <= rng[1])]
        fig = go.Figure()
        truth = p.drop_duplicates("date")
        fig.add_trace(go.Scatter(x=truth["date"], y=truth["y_true"], name="Actual", line=dict(color="#0b0b0b", width=2)))
        model_order = sorted(preds["model"].unique())
        for m in models:
            colour = GREY if m == "Persistence" else SERIES[model_order.index(m) % len(SERIES)]
            g = p[p["model"] == m]
            fig.add_trace(go.Scatter(x=g["date"], y=g["y_pred"].round(2), name=m,
                                     line=dict(color=colour, width=1.5)))
        fig.update_layout(title=f"{city}: next-day mean temperature", template="plotly_white",
                          hovermode="x unified", height=450, yaxis_title="°C")
        st.plotly_chart(fig, width="stretch")
        m = metrics[metrics["city"].isin([city, "ALL"])][["model", "city", "mae", "rmse", "smape", "mase",
                                                           "train_seconds", "model_size_mb"]]
        st.dataframe(m.sort_values(["city", "mae"]), hide_index=True)
        nd = load_table("next_day_forecast.csv")
        if not nd.empty:
            st.subheader("Forecast for the day after the last available data")
            st.dataframe(nd, hide_index=True)

# ------------------------------------------------------------------ SQL
with tab_sql:
    if not config.DB_PATH.exists():
        st.info("Database not found - run the database step.")
    else:
        key = st.selectbox("Query", list(QUERIES), format_func=lambda k: f"{k}: {QUERIES[k]['title']}")
        st.caption("SQL features: " + QUERIES[key]["features"])
        st.code(QUERIES[key]["sql"].strip(), language="sql")
        try:
            st.dataframe(run_sql(QUERIES[key]["sql"]), hide_index=True)
        except Exception as exc:
            st.error(f"Query failed: {exc}")

# ------------------------------------------------------------------ data quality
with tab_quality:
    st.subheader("Hourly data audit")
    st.dataframe(load_table("quality_audit.csv"), hide_index=True)
    st.subheader("Benchmarks: storage formats and engines")
    st.dataframe(load_table("benchmarks.csv"), hide_index=True)
    st.subheader("Daily IQR outliers (by city and month)")
    st.dataframe(load_table("daily_iqr_outliers.csv"), hide_index=True)
