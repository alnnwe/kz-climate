"""Next-day mean temperature forecast: baselines vs statistical vs ML models.

Target  : y(t+1) = daily mean temperature of the next local day, per city.
Inputs  : only information available at the end of day t (no leakage).
Split   : chronological - train 1950-2015, validation 2016-2020, test 2021+.
          Hyper-parameters of the boosting model are chosen with
          TimeSeriesSplit on train+validation dates; final models are refit on
          train+validation and evaluated once on the test period.

Models
  Persistence        y(t+1) = y(t)                         (naive baseline)
  Seasonal naive     y(t+1) = y(t+1-365)                   (seasonal baseline)
  Climatology        train-period mean for the calendar day
  Moving average 3d  mean of y(t), y(t-1), y(t-2)
  ARIMA(2,0,1)       on anomalies from climatology, per city (statsmodels)
  Ridge (lags)       linear regression on lag/rolling/weather features
  HistGB             gradient boosting on the same features
  HistGB + spatial   + yesterday's anomalies/changes of all 13 cities
"""
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA

import config
from src.analysis import add_calendar
from src.utils import save_json

MODEL_INFO = {
    "Persistence": ("baseline", "y(t+1) = y(t)"),
    "Seasonal naive": ("baseline", "y(t+1) = y(t+1-365 days)"),
    "Climatology": ("baseline", "calendar-day mean of the training period"),
    "Moving average 3d": ("baseline", "mean of the last 3 days"),
    "ARIMA(2,0,1)": ("statistical", "ARIMA on anomalies from climatology, per city"),
    "Ridge (lags)": ("ml", "Ridge regression on lag, rolling, calendar and weather features"),
    "HistGB": ("ml", "HistGradientBoostingRegressor on the same features"),
    "HistGB + spatial": ("ml", "HistGB + previous-day anomalies of all 13 cities"),
}

LAG_DAYS = [1, 2, 3, 4, 5, 6, 7]
WEATHER_FEATURES = ["tmax", "tmin", "t_23h", "humidity", "humidity_min", "dew_point", "pressure",
                    "pressure_23h", "pressure_range", "precipitation", "precip_hours"]


# ------------------------------------------------------------------ metrics
def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def smape_kelvin(y, p):
    """sMAPE in %, computed on the Kelvin scale.

    MAPE/sMAPE are undefined around 0 °C (division by ~0), which happens every
    spring and autumn in Kazakhstan, so the ratio is taken in Kelvin. The
    value is therefore small and mainly useful for comparing models.
    """
    y, p = np.asarray(y) + 273.15, np.asarray(p) + 273.15
    return float(100 * np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p))))


def mase(y, p, scale):
    """MAE scaled by the in-sample MAE of the persistence forecast (training data)."""
    return mae(y, p) / scale


def all_metrics(y, p, scale) -> dict:
    return {"mae": round(mae(y, p), 4), "rmse": round(rmse(y, p), 4),
            "smape": round(smape_kelvin(y, p), 4), "mase": round(mase(y, p, scale), 4)}


# ------------------------------------------------------------------ features
def make_climatology(train: pd.DataFrame) -> pd.DataFrame:
    """Smoothed calendar-day mean temperature per city, from TRAINING rows only."""
    clim = train.groupby(["city", "doy"], observed=True)["tmean"].mean().unstack("doy")
    ext = pd.concat([clim.T.iloc[-15:], clim.T, clim.T.iloc[:15]])
    smooth = ext.rolling(31, center=True).mean().iloc[15:-15].T
    return smooth.stack().rename("clim").reset_index()


def build_features(daily: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """One row per (city, day t) with features known at t and target y = tmean(t+1)."""
    d = add_calendar(daily).sort_values(["city", "date"]).reset_index(drop=True)
    d = d.merge(clim, on=["city", "doy"], how="left")
    g = d.groupby("city", observed=True)
    d["target"] = g["tmean"].shift(-1)
    d["target_date"] = g["date"].shift(-1)
    # climatology of the TARGET day (looked up by its day-of-year, so it also
    # exists for the last observed day) and today's anomaly
    d["next_doy"] = (d["doy"] % 365 + 1).astype("int16")
    nxt = clim.rename(columns={"doy": "next_doy", "clim": "clim_next"})
    d = d.merge(nxt, on=["city", "next_doy"], how="left")
    g = d.groupby("city", observed=True)
    d["anom_0"] = d["tmean"] - d["clim"]
    for k in LAG_DAYS:
        d[f"lag_{k}"] = g["tmean"].shift(k - 1)          # lag_1 = today
    for w in (3, 7, 30):
        d[f"roll_mean_{w}"] = g["tmean"].transform(lambda s: s.rolling(w).mean())
    d["roll_std_7"] = g["tmean"].transform(lambda s: s.rolling(7).std())
    d["diff_1"] = d["lag_1"] - d["lag_2"]
    d["pressure_change"] = g["pressure"].diff()
    d["pressure_trend_23h"] = d["pressure_23h"] - d["pressure"]   # pressure falling/rising late in the day
    d["seasonal_naive"] = g["tmean"].shift(364)        # value 365 days before the target day
    d["doy_sin"] = np.sin(2 * np.pi * d["next_doy"] / 365)
    d["doy_cos"] = np.cos(2 * np.pi * d["next_doy"] / 365)
    return d


def add_spatial_features(feat: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add today's anomaly and 1-day change of every city as extra columns."""
    wide_anom = feat.pivot_table(index="date", columns="city", values="anom_0", observed=True)
    wide_diff = feat.pivot_table(index="date", columns="city", values="diff_1", observed=True)
    wide_anom.columns = [f"sp_anom_{c}" for c in wide_anom.columns]
    wide_diff.columns = [f"sp_diff_{c}" for c in wide_diff.columns]
    wide = pd.concat([wide_anom, wide_diff], axis=1)
    out = feat.merge(wide, left_on="date", right_index=True, how="left")
    return out, list(wide.columns)


BASE_NUMERIC = ([f"lag_{k}" for k in LAG_DAYS] + ["roll_mean_3", "roll_mean_7", "roll_mean_30", "roll_std_7",
                "diff_1", "pressure_change", "pressure_trend_23h", "anom_0", "clim_next", "doy_sin", "doy_cos"] + WEATHER_FEATURES)


def make_pipeline(model, numeric: list[str], scale: bool) -> Pipeline:
    """ColumnTransformer (scaling + one-hot city) followed by the regressor."""
    num = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer([
        ("num", num, numeric),
        ("city", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"]),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def split_masks(feat: pd.DataFrame) -> dict:
    """Chronological masks on the TARGET date."""
    td = feat["target_date"]
    start = pd.Timestamp(config.FORECAST_START)
    return {
        "train": (td >= start) & (td <= config.TRAIN_END),
        "valid": (td > config.TRAIN_END) & (td <= config.VALID_END),
        "test": td > config.VALID_END,
    }


# ------------------------------------------------------------------ models
def fit_arima_per_city(feat: pd.DataFrame, fit_mask: pd.Series, test_mask: pd.Series) -> pd.Series:
    """ARIMA(2,0,1) on anomalies; parameters estimated on fit period, filtered one-step forecasts on test."""
    preds = pd.Series(np.nan, index=feat.index)
    for city, g in feat.groupby("city", observed=True):
        anom = (g["tmean"] - g["clim"]).astype(float)
        fit_idx = g.index[fit_mask.loc[g.index] | test_mask.loc[g.index]]
        hist = anom.loc[g.index[fit_mask.loc[g.index]]]
        # fit on the last 30 years of the fitting period to keep it fast
        hist = hist.iloc[-30 * 365:].reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(hist, order=(2, 0, 1), trend="c").fit()
            # apply the fitted parameters to fit+test anomalies (no re-estimation),
            # then take the one-step-ahead prediction made on day t for day t+1
            full = anom.loc[fit_idx].reset_index(drop=True)
            applied = res.apply(full)
            one_step = applied.predict(start=1, end=len(full))     # prediction for position i+1 made at i
        pred_next_anom = pd.Series(np.asarray(one_step)[: len(full)], index=fit_idx)
        idx_test = g.index[test_mask.loc[g.index]]
        preds.loc[idx_test] = pred_next_anom.loc[idx_test].to_numpy() + g.loc[idx_test, "clim_next"].to_numpy()
    return preds


def tune_histgb(X: pd.DataFrame, y: pd.Series, dates: pd.Series, numeric: list[str]) -> tuple[dict, pd.DataFrame]:
    """Small grid search with TimeSeriesSplit on unique dates (no shuffling)."""
    grid = [
        {"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": 400},
        {"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 300},
        {"learning_rate": 0.05, "max_leaf_nodes": 63, "max_iter": 400},
        {"learning_rate": 0.1, "max_leaf_nodes": 15, "max_iter": 300},
    ]
    uniq = np.sort(dates.unique())
    tss = TimeSeriesSplit(n_splits=4, test_size=365 * 4)
    rows = []
    for params in grid:
        scores = []
        for tr_d, va_d in tss.split(uniq):
            tr = dates.isin(uniq[tr_d]).to_numpy()
            va = dates.isin(uniq[va_d]).to_numpy()
            model = make_pipeline(HistGradientBoostingRegressor(random_state=config.RANDOM_STATE, **params),
                                  numeric, scale=False)
            model.fit(X[tr], y[tr])
            scores.append(mae(y[va], model.predict(X[va])))
        rows.append({**params, "cv_mae_mean": round(np.mean(scores), 4), "cv_mae_std": round(np.std(scores), 4)})
        print(f"    {params} -> CV MAE {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    cv = pd.DataFrame(rows).sort_values("cv_mae_mean")
    best = cv.iloc[0][["learning_rate", "max_leaf_nodes", "max_iter"]].to_dict()
    best["max_leaf_nodes"], best["max_iter"] = int(best["max_leaf_nodes"]), int(best["max_iter"])
    return best, cv


def model_size_mb(model) -> float:
    """Serialized size of a fitted model (joblib), MB."""
    path = config.MODEL_DIR / "_tmp_size.joblib"
    joblib.dump(model, path)
    size = path.stat().st_size / 1024 ** 2
    path.unlink()
    return round(size, 3)


def run_forecasting(daily: pd.DataFrame, tune: bool = True) -> dict:
    """Train/evaluate every model; return predictions, metrics and error analysis."""
    d0 = add_calendar(daily)
    train_rows = d0[(d0["date"] >= config.FORECAST_START) & (d0["date"] <= config.TRAIN_END)]
    clim = make_climatology(train_rows)                     # climatology from training years only
    feat = build_features(daily, clim)
    feat, spatial_cols = add_spatial_features(feat)
    feat = feat.dropna(subset=["target", "lag_7", "roll_mean_30", "seasonal_naive"] + WEATHER_FEATURES)
    masks = split_masks(feat)
    fit_mask = masks["train"] | masks["valid"]               # final fit on train+validation
    test = feat[masks["test"]].copy()
    y_fit = feat.loc[fit_mask, "target"]

    # scale for MASE: in-sample persistence MAE on the training data
    tr = feat[masks["train"]]
    mase_scale = mae(tr["target"], tr["lag_1"])

    preds = {}
    timing = {}
    sizes = {}

    def timed(name, fn_fit, fn_pred):
        t0 = time.perf_counter()
        obj = fn_fit()
        t1 = time.perf_counter()
        p = fn_pred(obj)
        t2 = time.perf_counter()
        timing[name] = {"train_seconds": round(t1 - t0, 3), "predict_seconds": round(t2 - t1, 3)}
        preds[name] = np.asarray(p, dtype=float)
        return obj

    print("  baselines")
    timed("Persistence", lambda: None, lambda _: test["lag_1"])
    timed("Seasonal naive", lambda: None, lambda _: test["seasonal_naive"])
    timed("Climatology", lambda: None, lambda _: test["clim_next"])
    timed("Moving average 3d", lambda: None, lambda _: test["roll_mean_3"])

    print("  ARIMA(2,0,1) per city")
    arima_pred = {}
    timed("ARIMA(2,0,1)", lambda: arima_pred.setdefault("p", fit_arima_per_city(feat, fit_mask, masks["test"])),
          lambda _: arima_pred["p"].loc[test.index])
    # ARIMA fit and filtering are done together - report all time as training time
    timing["ARIMA(2,0,1)"]["train_seconds"] += timing["ARIMA(2,0,1)"].pop("predict_seconds")
    timing["ARIMA(2,0,1)"]["predict_seconds"] = 0.0

    numeric = BASE_NUMERIC
    X_fit = feat.loc[fit_mask, numeric + ["city"]]
    X_test = test[numeric + ["city"]]

    print("  Ridge regression")
    ridge = timed("Ridge (lags)",
                  lambda: make_pipeline(Ridge(alpha=1.0), numeric, scale=True).fit(X_fit, y_fit),
                  lambda m: m.predict(X_test))
    sizes["Ridge (lags)"] = model_size_mb(ridge)

    if tune:
        print("  TimeSeriesSplit tuning of HistGB")
        best, cv = tune_histgb(X_fit, y_fit, feat.loc[fit_mask, "target_date"], numeric)
        cv.to_csv(config.TABLE_DIR / "histgb_cv_results.csv", index=False)
    else:
        best = {"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": 400}
    print(f"  HistGB (best params {best})")
    hgb = timed("HistGB",
                lambda: make_pipeline(HistGradientBoostingRegressor(random_state=config.RANDOM_STATE, **best),
                                      numeric, scale=False).fit(X_fit, y_fit),
                lambda m: m.predict(X_test))
    sizes["HistGB"] = model_size_mb(hgb)

    print("  HistGB + spatial features")
    num_sp = numeric + spatial_cols
    hgb_sp = timed("HistGB + spatial",
                   lambda: make_pipeline(HistGradientBoostingRegressor(random_state=config.RANDOM_STATE, **best),
                                         num_sp, scale=False).fit(feat.loc[fit_mask, num_sp + ["city"]], y_fit),
                   lambda m: m.predict(test[num_sp + ["city"]]))
    sizes["HistGB + spatial"] = model_size_mb(hgb_sp)
    joblib.dump({"model": hgb_sp, "features": num_sp, "clim": clim, "params": best},
                config.MODEL_DIR / "histgb_spatial.joblib")

    # -------------------------------------------------------------- evaluation
    pred_rows, metric_rows = [], []
    y_test = test["target"].to_numpy()
    for name, p in preds.items():
        ok = ~np.isnan(p)
        m = all_metrics(y_test[ok], p[ok], mase_scale)
        metric_rows.append({"model": name, "city": "ALL", "split": "test", **m, **timing[name],
                            "model_size_mb": sizes.get(name, 0.0), "family": MODEL_INFO[name][0]})
        for city in config.CITIES:
            cm = (test["city"] == city).to_numpy() & ok
            if cm.any():
                metric_rows.append({"model": name, "city": city, "split": "test",
                                    **all_metrics(y_test[cm], p[cm], mase_scale), **timing[name],
                                    "model_size_mb": sizes.get(name, 0.0), "family": MODEL_INFO[name][0]})
        pred_rows.append(pd.DataFrame({"city": test["city"].astype(str).to_numpy(), "model": name,
                                       "date": test["target_date"].to_numpy(), "y_true": y_test, "y_pred": p}))
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_rows, ignore_index=True)
    metrics.to_csv(config.TABLE_DIR / "forecast_metrics.csv", index=False)
    predictions.to_parquet(config.PROCESSED_DIR / "forecast_predictions.parquet", index=False)

    overall = metrics[metrics["city"] == "ALL"].sort_values("mae")
    print(overall[["model", "mae", "rmse", "smape", "mase", "train_seconds", "model_size_mb"]].to_string(index=False))

    errors = error_analysis(predictions, test)
    importance = feature_importance_ridge(ridge, numeric)
    save_json({"mase_scale_persistence_train_mae": round(mase_scale, 4), "histgb_params": best,
               "n_fit_rows": int(fit_mask.sum()), "n_test_rows": int(len(test)),
               "test_period": [str(test["target_date"].min().date()), str(test["target_date"].max().date())]},
              config.TABLE_DIR / "forecast_setup.json")
    return {"metrics": metrics, "predictions": predictions, "errors": errors, "ridge_coef": importance}


# ------------------------------------------------------------------ error analysis
def error_analysis(predictions: pd.DataFrame, test: pd.DataFrame) -> dict:
    """MAE by month, by size of the actual day-to-day change, and the worst cases."""
    p = predictions.copy()
    p["abs_err"] = (p["y_pred"] - p["y_true"]).abs()
    p["month"] = pd.to_datetime(p["date"]).dt.month
    by_month = p.pivot_table(index="month", columns="model", values="abs_err", aggfunc="mean").round(3)
    by_month.to_csv(config.TABLE_DIR / "forecast_mae_by_month.csv")

    change = (test["target"] - test["lag_1"]).abs().to_numpy()
    labels = ["<2°C", "2-5°C", "5-10°C", ">=10°C"]
    bins = pd.cut(change, [0, 2, 5, 10, 40], right=False, labels=labels)
    n_models = p["model"].nunique()
    p["actual_change"] = pd.Categorical(np.tile(np.asarray(bins, dtype=object), n_models),
                                        categories=labels, ordered=True)
    by_change = p.pivot_table(index="actual_change", columns="model", values="abs_err",
                              aggfunc="mean", observed=True).round(3)
    by_change["share_of_days_pct"] = (pd.Series(bins).value_counts(normalize=True) * 100).round(1)
    by_change.to_csv(config.TABLE_DIR / "forecast_mae_by_change.csv")

    best = (p[p["model"] == "HistGB + spatial"].nlargest(15, "abs_err")
            [["city", "date", "y_true", "y_pred", "abs_err"]].round({"y_true": 2, "y_pred": 2, "abs_err": 2}))
    best.to_csv(config.TABLE_DIR / "forecast_worst_cases.csv", index=False)
    return {"by_month": by_month, "by_change": by_change, "worst": best}


def feature_importance_ridge(ridge: Pipeline, numeric: list[str]) -> pd.DataFrame:
    """Standardised Ridge coefficients (numeric features) as a simple importance measure."""
    coefs = ridge.named_steps["model"].coef_[: len(numeric)]
    imp = pd.DataFrame({"feature": numeric, "coef_std": coefs}).assign(abs=lambda x: x["coef_std"].abs())
    imp = imp.sort_values("abs", ascending=False).drop(columns="abs").round(4)
    imp.to_csv(config.TABLE_DIR / "ridge_coefficients.csv", index=False)
    return imp


# ------------------------------------------------------------------ inference
def predict_next_day(daily: pd.DataFrame) -> pd.DataFrame:
    """Forecast for the day after the last available date, for every city."""
    path = config.MODEL_DIR / "histgb_spatial.joblib"
    try:
        bundle = joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError("trained model not found - run the forecasting step first (menu item 7)")
    feat = build_features(daily, bundle["clim"])
    feat, _ = add_spatial_features(feat)
    last_date = feat["date"].max()
    today = feat[feat["date"] == last_date].copy()
    today["forecast"] = bundle["model"].predict(today[bundle["features"] + ["city"]])
    out = today[["city", "date", "tmean", "forecast", "clim_next"]].rename(
        columns={"date": "last_observed_day", "tmean": "tmean_last_day", "clim_next": "climate_norm_next_day"})
    out["forecast_day"] = last_date + pd.Timedelta(days=1)
    num = ["tmean_last_day", "forecast", "climate_norm_next_day"]
    out[num] = out[num].astype(float).round(2)
    return out.reset_index(drop=True)
