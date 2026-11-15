"""Figures: static PNG (matplotlib) for the report and interactive HTML (Plotly).

Colour rules: categorical colours follow a fixed order (never cycled),
anomalies use a diverging blue<->red scale with a grey midpoint, and when
13 cities are on one chart only two are coloured and the rest are grey.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import config

# ---------------------------------------------------------------- style
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#b9b8b2"
INK, INK2 = "#0b0b0b", "#52514e"
DIVERGING = LinearSegmentedColormap.from_list(
    "kz_div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3a3a3", "#e34948", "#9e1f1f"])
HIGHLIGHT = {"Astana": SERIES[0], "Almaty": SERIES[1]}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.titleweight": "bold",
    "axes.titlesize": 11, "legend.frameon": False,
})


def _save(fig, name: str) -> str:
    path = config.FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _save_html(fig: go.Figure, name: str) -> str:
    path = config.FIG_DIR / f"{name}.html"
    fig.update_layout(template="plotly_white", font=dict(family="Arial", size=12))
    fig.write_html(path, include_plotlyjs="cdn")
    return str(path)


# ---------------------------------------------------------------- figures
def warming_stripes(annual: pd.DataFrame) -> str:
    """Heat-map of annual anomalies (rows = cities from north to south)."""
    order = sorted(config.CITIES, key=lambda c: -config.CITIES[c]["lat"])
    tab = annual.pivot_table(index="city", columns="year", values="anomaly", observed=True).reindex(order)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    lim = np.nanmax(np.abs(tab.values))
    im = ax.imshow(tab.values, aspect="auto", cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), interpolation="none")
    ax.set_yticks(range(len(order)), order)
    years = tab.columns.to_numpy()
    ticks = [i for i, y in enumerate(years) if y % 10 == 0]
    ax.set_xticks(ticks, years[ticks])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Anomaly vs 1961-1990, °C")
    ax.set_title("Annual mean temperature anomaly, 1940-2025 (cities ordered north to south)")
    return _save(fig, "01_warming_stripes")


def decadal_anomaly_static(annual: pd.DataFrame) -> str:
    """Decadal mean anomaly: all cities grey, Astana and Almaty highlighted."""
    dec = annual[annual["year"] <= 2025].groupby(["city", "decade"], observed=True)["anomaly"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for city, g in dec.groupby("city", observed=True):
        if city in HIGHLIGHT:
            continue
        ax.plot(g["decade"], g["anomaly"], color=GREY, lw=1.2, alpha=0.9)
    for city, color in HIGHLIGHT.items():
        g = dec[dec["city"] == city]
        ax.plot(g["decade"], g["anomaly"], color=color, lw=2.2, marker="o", ms=5, label=city)
        ax.annotate(city, (g["decade"].iloc[-1], g["anomaly"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", color=INK, va="center", fontsize=9)
    ax.plot([], [], color=GREY, lw=1.2, label="other 11 cities")
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.set_xlabel("Decade (2020s = 2020-2025)")
    ax.set_ylabel("Anomaly vs 1961-1990, °C")
    ax.set_title("Warming by decade")
    ax.legend(loc="upper left")
    return _save(fig, "02_decadal_anomaly")


def decadal_anomaly_interactive(annual: pd.DataFrame) -> str:
    """Interactive Plotly version: hover each city, click legend to show/hide."""
    a = annual[annual["year"] <= 2025].copy()
    a["ma10"] = a.groupby("city", observed=True)["anomaly"].transform(lambda s: s.rolling(10, min_periods=5).mean())
    fig = go.Figure()
    for city in config.CITIES:
        g = a[a["city"] == city]
        hl = city in HIGHLIGHT
        fig.add_trace(go.Scatter(
            x=g["year"], y=g["ma10"].round(2), name=city, mode="lines",
            line=dict(color=HIGHLIGHT.get(city, GREY), width=3 if hl else 1.5),
            visible=True if hl else "legendonly" if city not in ("Shymkent", "Petropavl", "Aktau") else True,
            hovertemplate=f"<b>{city}</b><br>%{{x}}: %{{y:+.2f}} °C<extra></extra>"))
    fig.add_hline(y=0, line_color="#8a8984", line_width=1)
    fig.update_layout(title="10-year moving average of annual anomaly vs 1961-1990 (click legend to add cities)",
                      xaxis_title="Year", yaxis_title="Anomaly, °C", hovermode="x unified", height=520)
    return _save_html(fig, "02_decadal_anomaly_interactive")


def trend_map(trends: pd.DataFrame) -> str:
    """Interactive map: warming rate 1976-2025 per city."""
    t = trends[trends["period"] == "1976-2025"].copy()
    t["label"] = t["city"] + ": +" + t["ols_per_decade"].round(2).astype(str) + " °C/decade"
    fig = px.scatter_geo(t, lat="lat", lon="lon", color="ols_per_decade", size=t["ols_per_decade"].clip(lower=0.05),
                         hover_name="city", text="city",
                         hover_data={"ols_per_decade": ":.2f", "ols_p": ":.4f", "lat": False, "lon": False},
                         color_continuous_scale=["#fbe3d7", "#f3a3a3", "#e34948", "#9e1f1f"],
                         labels={"ols_per_decade": "°C / decade"}, scope="asia")
    fig.update_traces(textposition="top center")
    fig.update_geos(lataxis_range=[39, 57], lonaxis_range=[45, 88], showcountries=True, countrycolor="#8a8984")
    fig.update_layout(title="Warming rate of annual mean temperature, 1976-2025", height=560)
    return _save_html(fig, "03_trend_map")


def trend_bars(trends: pd.DataFrame) -> str:
    """Horizontal bars: warming rate per city for two periods."""
    t = trends.pivot(index="city", columns="period", values="ols_per_decade").sort_values("1976-2025")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    y = np.arange(len(t))
    ax.barh(y + 0.2, t["1976-2025"], height=0.38, color=SERIES[1], label="1976-2025")
    ax.barh(y - 0.2, t["1940-2025"], height=0.38, color=SERIES[0], label="1940-2025")
    ax.set_yticks(y, t.index)
    for i, v in enumerate(t["1976-2025"]):
        ax.text(v + 0.01, i + 0.2, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("Linear trend of annual mean temperature, °C per decade")
    ax.set_title("Warming rate by city")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    return _save(fig, "04_trend_bars")


def extremes_by_decade(wave_counts: pd.DataFrame, indices: pd.DataFrame) -> str:
    """Heat/cold waves per decade + hot days and frost days per decade (two panels)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    w = wave_counts.pivot(index="decade", columns="event_type", values="events_per_10y")
    x = np.arange(len(w))
    axes[0].bar(x - 0.2, w["heat_wave"], width=0.38, color=SERIES[1], label="Heat waves")
    axes[0].bar(x + 0.2, w["cold_wave"], width=0.38, color=SERIES[0], label="Cold waves")
    axes[0].set_xticks(x, [f"{d}s" for d in w.index])
    axes[0].set_ylabel("Events per decade, 13 cities total")
    axes[0].set_title("Heat and cold waves (≥3 days beyond p90 / p10)")
    axes[0].legend()
    axes[0].grid(axis="x", visible=False)

    dec = indices[indices["year"] <= 2025].groupby("decade")[["hot_days_35", "frost_days"]].mean()
    ax2 = axes[1]
    ax2.plot(dec.index, dec["frost_days"], color=SERIES[0], marker="o", lw=2, label="Frost days (Tmin<0)")
    ax2.set_ylabel("Days per year, mean of 13 cities")
    ax2.set_title("Frost days per year")
    ax2.legend(loc="upper right")
    ax2.set_xticks(dec.index, [f"{d}s" for d in dec.index])
    return _save(fig, "05_extremes_by_decade")


def hot_days_small_multiples(indices: pd.DataFrame) -> str:
    """Days with Tmax >= 35 °C per year, 10-year mean, one panel per southern/western city."""
    cities = ["Shymkent", "Kyzylorda", "Atyrau", "Aktau", "Almaty", "Aktobe"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.5), sharex=True)
    for ax, city in zip(axes.ravel(), cities):
        g = indices[(indices["city"] == city) & (indices["year"] <= 2025)]
        ax.bar(g["year"], g["hot_days_35"], color="#f3c1ae", width=0.9)
        ax.plot(g["year"], g["hot_days_35"].rolling(10, min_periods=5).mean(), color=SERIES[1], lw=2)
        ax.set_title(city, fontsize=10)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Days / year")
    fig.suptitle("Hot days (Tmax ≥ 35 °C) per year with 10-year moving average", fontweight="bold")
    return _save(fig, "06_hot_days_small_multiples")


def stl_plot(stl: pd.DataFrame, city: str = "Astana") -> str:
    """STL decomposition of monthly mean temperature."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 7.5), sharex=True)
    for ax, col, color in zip(axes, ["observed", "trend", "seasonal", "resid"],
                              [INK2, SERIES[1], SERIES[0], GREY]):
        ax.plot(stl.index, stl[col], color=color, lw=0.9 if col != "trend" else 2)
        ax.set_ylabel(col + ", °C")
    axes[0].set_title(f"STL decomposition of monthly mean temperature, {city} (period = 12 months, robust)")
    return _save(fig, "07_stl_astana")


def acf_pacf_plot(daily: pd.DataFrame, city: str = "Astana") -> str:
    """ACF and PACF of daily temperature anomalies."""
    from src.analysis import daily_anomalies
    d = daily_anomalies(daily[daily["city"] == city])
    s = d.set_index("date")["anomaly"].astype(float).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    plot_acf(s, lags=40, ax=axes[0], color=SERIES[0], vlines_kwargs={"colors": SERIES[0]})
    plot_pacf(s, lags=40, ax=axes[1], color=SERIES[0], vlines_kwargs={"colors": SERIES[0]}, method="ywm")
    axes[0].set_title(f"ACF of daily anomaly, {city}")
    axes[1].set_title(f"PACF of daily anomaly, {city}")
    for ax in axes:
        ax.set_xlabel("Lag, days")
    return _save(fig, "08_acf_pacf")


def correlation_heatmaps(weather_corr: pd.DataFrame, city_corr: pd.DataFrame) -> str:
    """Two heat-maps: weather variables (Spearman) and anomaly correlation between cities."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, corr, title in ((axes[0], weather_corr, "Daily weather variables (Spearman, anomalies)"),
                            (axes[1], city_corr, "Daily temperature anomalies between cities (Pearson)")):
        im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
        ax.grid(False)
        for i in range(len(corr)):
            for j in range(len(corr)):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.6 else INK)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    path = config.FIG_DIR / "09_correlations.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def corr_vs_distance(pairs: pd.DataFrame) -> str:
    """Scatter: inter-city anomaly correlation vs distance."""
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.scatter(pairs["distance_km"], pairs["corr"], s=36, color=SERIES[0], edgecolor="white", linewidth=1)
    z = np.polyfit(pairs["distance_km"], pairs["corr"], 1)
    xs = np.linspace(pairs["distance_km"].min(), pairs["distance_km"].max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=SERIES[1], lw=2, label=f"linear fit: {z[0] * 1000:.2f} per 1000 km")
    ax.set_xlabel("Distance between cities, km")
    ax.set_ylabel("Correlation of daily anomalies")
    ax.set_title("Weather anomalies are shared over ~1000 km")
    ax.legend()
    return _save(fig, "10_corr_vs_distance")


def monthly_boxplot(daily: pd.DataFrame, city: str = "Astana") -> str:
    """Box-plots of daily mean temperature by month (IQR outliers visible)."""
    d = daily[daily["city"] == city]
    data = [d.loc[d["date"].dt.month == m, "tmean"].dropna() for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.boxplot(data, tick_labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
               patch_artist=True, boxprops=dict(facecolor="#cde2fb", color=SERIES[0]),
               medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=SERIES[0]),
               capprops=dict(color=SERIES[0]), flierprops=dict(marker="o", ms=2.5, mfc=SERIES[1], mec="none", alpha=0.6))
    ax.set_ylabel("Daily mean temperature, °C")
    ax.set_title(f"Daily mean temperature by month, {city} 1940-2026 (dots = IQR outliers)")
    return _save(fig, "11_monthly_boxplot")


def forecast_example(predictions: pd.DataFrame, city: str = "Astana", start="2025-01-01", end="2025-03-31") -> str:
    """Actual vs forecasts for one city over a winter period."""
    p = predictions[(predictions["city"] == city) & (predictions["date"] >= start) & (predictions["date"] <= end)]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    truth = p[p["model"] == "Persistence"]
    ax.plot(truth["date"], truth["y_true"], color=INK, lw=2, label="Actual")
    for name, color, ls in (("Persistence", GREY, "--"), ("Climatology", SERIES[2], ":"),
                            ("HistGB + spatial", SERIES[1], "-")):
        g = p[p["model"] == name]
        ax.plot(g["date"], g["y_pred"], color=color, lw=1.6, ls=ls, label=name)
    ax.set_ylabel("Daily mean temperature, °C")
    ax.set_title(f"Next-day forecast vs actual, {city}, {start[:7]} .. {end[:7]} (test period)")
    ax.legend(ncol=4, loc="lower left")
    return _save(fig, "12_forecast_example")


def model_comparison(metrics: pd.DataFrame) -> str:
    """MAE of every model (all cities, test period) + training time."""
    m = metrics[metrics["city"] == "ALL"].sort_values("mae", ascending=False)
    fam_color = {"baseline": GREY, "statistical": SERIES[0], "ml": SERIES[1]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.4, 1]})
    y = np.arange(len(m))
    axes[0].barh(y, m["mae"], color=[fam_color[f] for f in m["family"]], height=0.6)
    axes[0].set_yticks(y, m["model"])
    for i, v in enumerate(m["mae"]):
        axes[0].text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
    axes[0].set_xlabel(f"MAE, °C (test period 2021-2026, {metrics['city'].nunique() - 1} cities)")
    axes[0].set_title("Next-day forecast error")
    axes[0].grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=c, label=f) for f, c in fam_color.items()],
                   loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    t = m["train_seconds"] + m["predict_seconds"]
    axes[1].barh(y, t.clip(lower=0.01), color=[fam_color[f] for f in m["family"]], height=0.6)
    axes[1].set_xscale("log")
    axes[1].set_yticks(y, [""] * len(m))
    axes[1].set_xlabel("Train + predict time, s (log scale)")
    axes[1].set_title("Compute cost")
    axes[1].grid(axis="y", visible=False)
    return _save(fig, "13_model_comparison")


def error_by_month(by_month: pd.DataFrame) -> str:
    """MAE by calendar month for key models."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, color in (("Persistence", GREY), ("ARIMA(2,0,1)", SERIES[0]), ("HistGB + spatial", SERIES[1])):
        if name in by_month:
            ax.plot(by_month.index, by_month[name], marker="o", lw=2, color=color, label=name)
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("MAE, °C")
    ax.set_title("Forecast error by month: winter is hardest")
    ax.legend()
    return _save(fig, "14_error_by_month")


def forecast_interactive(predictions: pd.DataFrame, city: str = "Astana") -> str:
    """Interactive actual vs predicted for the whole test period."""
    p = predictions[predictions["city"] == city]
    fig = go.Figure()
    t = p[p["model"] == "Persistence"]
    fig.add_trace(go.Scatter(x=t["date"], y=t["y_true"], name="Actual", line=dict(color=INK, width=2)))
    for name, color in (("Persistence", GREY), ("ARIMA(2,0,1)", SERIES[0]), ("HistGB + spatial", SERIES[1])):
        g = p[p["model"] == name]
        fig.add_trace(go.Scatter(x=g["date"], y=g["y_pred"].round(2), name=name, line=dict(color=color, width=1.5)))
    fig.update_layout(title=f"{city}: next-day mean temperature, test period (zoom with the range slider)",
                      yaxis_title="°C", hovermode="x unified", height=520,
                      xaxis=dict(rangeslider=dict(visible=True)))
    return _save_html(fig, "15_forecast_interactive")


def make_analysis_figures(daily: pd.DataFrame, res: dict) -> list[str]:
    """All analysis figures; returns their paths."""
    paths = [
        warming_stripes(res["annual"]),
        decadal_anomaly_static(res["annual"]),
        decadal_anomaly_interactive(res["annual"]),
        trend_map(res["trends"]),
        trend_bars(res["trends"]),
        extremes_by_decade(res["wave_counts"], res["indices"]),
        hot_days_small_multiples(res["indices"]),
        stl_plot(res["stl"]),
        acf_pacf_plot(daily),
        correlation_heatmaps(res["weather_corr"], res["city_corr"]),
        corr_vs_distance(res["city_pairs"]),
        monthly_boxplot(daily),
    ]
    return paths


def make_forecast_figures(res: dict) -> list[str]:
    return [
        forecast_example(res["predictions"]),
        model_comparison(res["metrics"]),
        error_by_month(res["errors"]["by_month"]),
        forecast_interactive(res["predictions"]),
    ]
