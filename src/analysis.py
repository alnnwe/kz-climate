"""Climate analysis: EDA, warming trends, climate indices, heat/cold waves,
time-series diagnostics (STL, ADF/KPSS, ACF/PACF) and correlations."""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

import config

LAST_FULL_YEAR = 2025
warnings.filterwarnings("ignore", message=".*(adfuller|kpss) currently returns.*")
warnings.filterwarnings("ignore", message=".*outside of the range of p-values.*")


# ------------------------------------------------------------------ helpers
def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, decade, season and a 365-day day-of-year (Feb 29 -> Feb 28)."""
    out = df.copy()
    d = out["date"]
    out["year"] = d.dt.year.astype("int16")
    out["month"] = d.dt.month.astype("int8")
    out["decade"] = (out["year"] // 10 * 10).astype("int16")
    seasons = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
               6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    out["season"] = out["month"].map(seasons)
    doy = d.dt.dayofyear
    leap_shift = (d.dt.is_leap_year & (d.dt.month > 2)).astype(int)
    out["doy"] = (doy - leap_shift).clip(upper=365).astype("int16")
    return out


def full_years(df: pd.DataFrame) -> pd.DataFrame:
    """Keep complete years only (1940..2025)."""
    return df[(df["year"] >= 1940) & (df["year"] <= LAST_FULL_YEAR)]


# ------------------------------------------------------------------ EDA
def describe_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics of the main daily variables, per city."""
    cols = ["tmean", "tmax", "tmin", "precipitation", "humidity", "dew_point", "pressure"]
    desc = daily.groupby("city", observed=True)[cols].agg(["mean", "std", "min", "max"]).round(2)
    desc.columns = [f"{a}_{b}" for a, b in desc.columns]
    desc.to_csv(config.TABLE_DIR / "describe_daily.csv")
    return desc


def feature_types(daily: pd.DataFrame) -> pd.DataFrame:
    """Type of every column (numeric continuous / categorical / datetime)."""
    rows = []
    for col, dt in daily.dtypes.items():
        if str(dt) == "category":
            kind = "categorical"
        elif np.issubdtype(dt, np.datetime64):
            kind = "datetime"
        elif col == "n_hours":
            kind = "numeric discrete (count)"
        else:
            kind = "numeric continuous"
        rows.append({"column": col, "dtype": str(dt), "type": kind, "missing": int(daily[col].isna().sum())})
    out = pd.DataFrame(rows)
    out.to_csv(config.TABLE_DIR / "feature_types.csv", index=False)
    return out


def daily_iqr_outliers(daily: pd.DataFrame) -> pd.DataFrame:
    """IQR outliers of daily mean temperature per city and month.

    The IQR rule is applied within (city, month) so that seasonality is not
    counted as an outlier. ERA5 reanalysis has no instrument spikes, so these
    points are real extremes: they are reported and kept, not removed.
    """
    d = add_calendar(daily)
    grp = d.groupby(["city", "month"], observed=True)["tmean"]
    q1, q3 = grp.transform("quantile", 0.25), grp.transform("quantile", 0.75)
    iqr = q3 - q1
    d["outlier"] = (d["tmean"] < q1 - 1.5 * iqr) | (d["tmean"] > q3 + 1.5 * iqr)
    d["low"] = d["tmean"] < q1 - 1.5 * iqr
    res = d.groupby("city", observed=True).agg(
        outliers=("outlier", "sum"), low_outliers=("low", "sum"), days=("outlier", "size"))
    res["outlier_pct"] = (100 * res["outliers"] / res["days"]).round(2)
    res.to_csv(config.TABLE_DIR / "daily_iqr_outliers.csv")
    return res


# ------------------------------------------------------------------ trends
def annual_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Annual means and anomalies vs the 1961-1990 baseline, per city."""
    d = full_years(add_calendar(daily))
    annual = d.groupby(["city", "year"], observed=True).agg(
        tmean=("tmean", "mean"), tmax=("tmax", "mean"), tmin=("tmin", "mean"),
        precipitation=("precipitation", "sum")).reset_index()
    b0, b1 = config.BASELINE_PERIOD
    base = annual[annual["year"].between(b0, b1)].groupby("city", observed=True)["tmean"].mean()
    annual["anomaly"] = annual["tmean"] - annual["city"].map(base).astype(float)
    annual["decade"] = annual["year"] // 10 * 10
    annual.to_csv(config.TABLE_DIR / "annual_by_city.csv", index=False)
    return annual


def decade_table(annual: pd.DataFrame) -> pd.DataFrame:
    """Mean anomaly per city and decade (wide table: rows = city)."""
    tab = annual.pivot_table(index="city", columns="decade", values="anomaly", aggfunc="mean", observed=True).round(2)
    tab.to_csv(config.TABLE_DIR / "decade_anomaly_by_city.csv")
    return tab


def linear_trend(years: np.ndarray, values: np.ndarray) -> dict:
    """OLS slope, Theil-Sen slope and Mann-Kendall-type test (Kendall tau), per decade."""
    ols = stats.linregress(years, values)
    ts = stats.theilslopes(values, years)
    tau, p_mk = stats.kendalltau(years, values)
    return {
        "ols_per_decade": ols.slope * 10,
        "ols_p": ols.pvalue,
        "ols_ci95": 1.96 * ols.stderr * 10,
        "theil_sen_per_decade": ts.slope * 10,
        "kendall_tau": tau,
        "mk_p": p_mk,
    }


def warming_trends(annual: pd.DataFrame) -> pd.DataFrame:
    """Trend of annual mean temperature (°C/decade) for 1940-2025 and 1976-2025."""
    rows = []
    for city, g in annual.groupby("city", observed=True):
        for label, y0 in (("1940-2025", 1940), ("1976-2025", 1976)):
            sub = g[g["year"] >= y0]
            tr = linear_trend(sub["year"].to_numpy(), sub["tmean"].to_numpy())
            rows.append({"city": city, "period": label, **{k: round(v, 4) for k, v in tr.items()}})
    out = pd.DataFrame(rows)
    meta = pd.DataFrame(config.CITIES).T[["lat", "lon", "macro_zone"]]
    out = out.merge(meta, left_on="city", right_index=True)
    out.to_csv(config.TABLE_DIR / "warming_trends.csv", index=False)
    return out


def seasonal_trends(daily: pd.DataFrame) -> pd.DataFrame:
    """Trend per season (1976-2025), to see which season warms fastest."""
    d = full_years(add_calendar(daily))
    d = d[d["year"] >= 1976]
    # December belongs to the next year's winter
    d["season_year"] = np.where(d["month"] == 12, d["year"] + 1, d["year"])
    s = d.groupby(["city", "season", "season_year"], observed=True)["tmean"].mean().reset_index()
    s = s[(s["season_year"] > 1976) & (s["season_year"] <= LAST_FULL_YEAR)]
    rows = []
    for (city, season), g in s.groupby(["city", "season"], observed=True):
        tr = linear_trend(g["season_year"].to_numpy(), g["tmean"].to_numpy())
        rows.append({"city": city, "season": season, "trend_per_decade": round(tr["ols_per_decade"], 3),
                     "p_value": round(tr["ols_p"], 4)})
    out = pd.DataFrame(rows).pivot(index="city", columns="season", values="trend_per_decade")
    out = out[["DJF", "MAM", "JJA", "SON"]]
    out.to_csv(config.TABLE_DIR / "seasonal_trends_1976_2025.csv")
    return out


# ------------------------------------------------------------------ extremes
def climate_indices(daily: pd.DataFrame) -> pd.DataFrame:
    """ETCCDI-style annual indices per city (counts of days)."""
    d = full_years(add_calendar(daily))
    d = d.assign(
        frost_days=d["tmin"] < 0,
        ice_days=d["tmax"] < 0,
        summer_days=d["tmax"] > 25,
        hot_days_35=d["tmax"] >= 35,
        tropical_nights=d["tmin"] > 20,
        severe_cold_days=d["tmin"] <= -30,
        heavy_precip_days=d["precipitation"] >= 10,
    )
    cols = ["frost_days", "ice_days", "summer_days", "hot_days_35", "tropical_nights",
            "severe_cold_days", "heavy_precip_days"]
    idx = d.groupby(["city", "year"], observed=True)[cols].sum().reset_index()
    idx["decade"] = idx["year"] // 10 * 10
    idx.to_csv(config.TABLE_DIR / "climate_indices_annual.csv", index=False)
    by_decade = idx.groupby(["city", "decade"], observed=True)[cols].mean().round(1).reset_index()
    by_decade.to_csv(config.TABLE_DIR / "climate_indices_by_decade.csv", index=False)
    return idx


def calendar_percentiles(d: pd.DataFrame, col: str, q: float) -> pd.Series:
    """Calendar-day percentile (doy 1..365) over the baseline period with a ±7-day window."""
    b0, b1 = config.BASELINE_PERIOD
    base = d[d["year"].between(b0, b1)]
    doys = base["doy"].to_numpy()
    vals = base[col].to_numpy()
    half = config.PCTL_WINDOW_DAYS // 2
    out = np.empty(365)
    for day in range(1, 366):
        dist = np.abs(doys - day)
        dist = np.minimum(dist, 365 - dist)                # circular distance
        out[day - 1] = np.nanquantile(vals[dist <= half], q)
    return pd.Series(out, index=np.arange(1, 366))


def runs_of_true(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs of consecutive True runs."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def detect_waves(daily: pd.DataFrame) -> pd.DataFrame:
    """Heat waves (Tmax > p90) and cold waves (Tmin < p10), >= 3 consecutive days."""
    events = []
    d_all = add_calendar(daily)
    for city, d in d_all.groupby("city", observed=True):
        d = d.sort_values("date").reset_index(drop=True)
        specs = {
            "heat_wave": ("tmax", calendar_percentiles(d, "tmax", config.HEAT_PCTL), 1),
            "cold_wave": ("tmin", calendar_percentiles(d, "tmin", config.COLD_PCTL), -1),
        }
        for etype, (col, thr_by_doy, sign) in specs.items():
            thr = d["doy"].map(thr_by_doy).to_numpy()
            vals = d[col].to_numpy()
            exceed = sign * (vals - thr) > 0
            for s, e in runs_of_true(exceed):
                length = e - s + 1
                if length < config.WAVE_MIN_DAYS:
                    continue
                seg = vals[s:e + 1]
                events.append({
                    "city": city, "event_type": etype,
                    "start_date": d.loc[s, "date"], "end_date": d.loc[e, "date"],
                    "duration_days": length,
                    "peak_value": float(seg.max() if sign > 0 else seg.min()),
                    "mean_exceedance": float(np.mean(np.abs(seg - thr[s:e + 1]))),
                })
    ev = pd.DataFrame(events)
    ev["decade"] = ev["start_date"].dt.year // 10 * 10
    ev.to_csv(config.TABLE_DIR / "extreme_events.csv", index=False)
    return ev


def wave_counts_by_decade(events: pd.DataFrame) -> pd.DataFrame:
    """Number of waves and wave-days per decade (all cities, full decades only)."""
    ev = events[events["start_date"].dt.year <= LAST_FULL_YEAR]
    tab = ev.groupby(["decade", "event_type"]).agg(
        events=("duration_days", "size"), wave_days=("duration_days", "sum"),
        mean_duration=("duration_days", "mean")).round(2).reset_index()
    # the 2020s decade is incomplete (2020-2025 = 6 years): normalise per 10 years
    years_in = tab["decade"].map(lambda dec: min(10, LAST_FULL_YEAR - dec + 1))
    tab["events_per_10y"] = (tab["events"] * 10 / years_in).round(1)
    tab["wave_days_per_10y"] = (tab["wave_days"] * 10 / years_in).round(1)
    tab.to_csv(config.TABLE_DIR / "wave_counts_by_decade.csv", index=False)
    return tab


# ------------------------------------------------------------------ time series
def monthly_series(daily: pd.DataFrame, city: str) -> pd.Series:
    """Monthly mean temperature of one city (resample)."""
    s = daily[daily["city"] == city].set_index("date")["tmean"].astype(float)
    return s.resample("MS").mean().loc[: f"{LAST_FULL_YEAR}-12-01"]


def stl_decompose(daily: pd.DataFrame, city: str = "Astana"):
    """Robust STL decomposition of monthly mean temperature (period 12)."""
    monthly = monthly_series(daily, city)
    res = STL(monthly, period=12, robust=True).fit()
    comp = pd.DataFrame({"observed": monthly, "trend": res.trend, "seasonal": res.seasonal, "resid": res.resid})
    comp.to_csv(config.TABLE_DIR / f"stl_{city}.csv")
    strength_trend = max(0, 1 - res.resid.var() / (res.trend + res.resid).var())
    strength_season = max(0, 1 - res.resid.var() / (res.seasonal + res.resid).var())
    return comp, {"trend_strength": round(strength_trend, 3), "seasonal_strength": round(strength_season, 3)}


def daily_anomalies(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily anomaly = tmean minus the city's 1961-1990 calendar-day mean (smoothed)."""
    d = add_calendar(daily)
    b0, b1 = config.BASELINE_PERIOD
    base = d[d["year"].between(b0, b1)]
    clim = base.groupby(["city", "doy"], observed=True)["tmean"].mean().unstack("doy")
    # smooth the climatology with a circular 31-day moving average
    clim = clim.T.pipe(lambda c: pd.concat([c.iloc[-15:], c, c.iloc[:15]]).rolling(31, center=True).mean().iloc[15:-15]).T
    lookup = clim.stack().rename("clim").reset_index()
    d = d.merge(lookup, on=["city", "doy"], how="left")
    d["anomaly"] = d["tmean"] - d["clim"]
    return d


def stationarity_tests(daily: pd.DataFrame, city: str = "Astana") -> pd.DataFrame:
    """ADF and KPSS tests for raw daily, daily anomaly, monthly and annual series."""
    d = daily_anomalies(daily[daily["city"] == city])
    series = {
        "daily tmean (raw)": d.set_index("date")["tmean"].astype(float),
        "daily anomaly": d.set_index("date")["anomaly"].astype(float),
        "monthly mean": monthly_series(daily, city),
        "annual mean": d[d["year"] <= LAST_FULL_YEAR].groupby("year")["tmean"].mean(),
        "annual mean (1st difference)": d[d["year"] <= LAST_FULL_YEAR].groupby("year")["tmean"].mean().diff(),
    }
    rows = []
    for name, s in series.items():
        s = s.dropna()
        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
        try:
            kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
        except (ValueError, OverflowError):
            kpss_stat, kpss_p = np.nan, np.nan
        verdict = "stationary" if adf_p < 0.05 and kpss_p > 0.05 else (
            "non-stationary" if adf_p >= 0.05 and kpss_p <= 0.05 else "mixed (trend/seasonal)")
        rows.append({"series": name, "n": len(s), "adf_stat": round(adf_stat, 3), "adf_p": round(adf_p, 4),
                     "kpss_stat": round(kpss_stat, 3), "kpss_p": round(kpss_p, 4), "verdict": verdict})
    out = pd.DataFrame(rows)
    out.to_csv(config.TABLE_DIR / f"stationarity_{city}.csv", index=False)
    return out


# ------------------------------------------------------------------ correlations
def weather_correlations(daily: pd.DataFrame) -> pd.DataFrame:
    """Correlation of daily temperature anomaly with other weather variables (all cities)."""
    d = daily_anomalies(daily)
    d["pressure_change"] = d.groupby("city", observed=True)["pressure"].diff()
    d["next_day_change"] = d.groupby("city", observed=True)["tmean"].shift(-1) - d["tmean"]
    # remove the seasonal cycle of every other variable too (city x month mean),
    # otherwise the correlations would mostly reflect summer/winter differences
    d["diurnal_range"] = d["tmax"] - d["tmin"]            # proxy for cloudiness (clear sky -> large range)
    others = ["humidity", "dew_point", "pressure", "pressure_range", "precipitation", "precip_hours", "diurnal_range"]
    for col in others:
        d[col] = d[col] - d.groupby(["city", "month"], observed=True)[col].transform("mean")
    d = d.rename(columns={"anomaly": "temp_anomaly"})
    cols = ["temp_anomaly", "next_day_change"] + others + ["pressure_change"]
    corr = d[cols].astype(float).corr(method="spearman").round(3)
    corr.to_csv(config.TABLE_DIR / "weather_correlation_spearman.csv")
    return corr


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two points, km."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def city_anomaly_correlation(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlation of daily anomalies between cities and its decay with distance."""
    d = daily_anomalies(daily)
    wide = d.pivot_table(index="date", columns="city", values="anomaly", observed=True)
    corr = wide.corr().round(3)
    corr.to_csv(config.TABLE_DIR / "city_anomaly_correlation.csv")
    pairs = []
    names = list(corr.columns)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ca, cb = config.CITIES[a], config.CITIES[b]
            pairs.append({"city_a": a, "city_b": b, "corr": corr.loc[a, b],
                          "distance_km": round(haversine_km(ca["lat"], ca["lon"], cb["lat"], cb["lon"]), 0)})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(config.TABLE_DIR / "city_pairs_corr_distance.csv", index=False)
    return corr, pairs


def run_analysis(daily: pd.DataFrame) -> dict:
    """Run every analysis step and return the main tables in a dict."""
    out = {}
    print("  descriptive statistics, feature types, outliers")
    out["describe"] = describe_daily(daily)
    out["feature_types"] = feature_types(daily)
    out["outliers"] = daily_iqr_outliers(daily)
    print("  annual table, decadal anomalies, trends")
    out["annual"] = annual_table(daily)
    out["decades"] = decade_table(out["annual"])
    out["trends"] = warming_trends(out["annual"])
    out["seasonal_trends"] = seasonal_trends(daily)
    print("  climate indices and heat/cold waves")
    out["indices"] = climate_indices(daily)
    out["events"] = detect_waves(daily)
    out["wave_counts"] = wave_counts_by_decade(out["events"])
    print("  STL, stationarity, correlations")
    out["stl"], out["stl_strength"] = stl_decompose(daily, "Astana")
    out["stationarity"] = stationarity_tests(daily, "Astana")
    out["weather_corr"] = weather_correlations(daily)
    out["city_corr"], out["city_pairs"] = city_anomaly_correlation(daily)
    return out
