"""Recursive hourly forecasts; no future observations enter a forecast window."""
from __future__ import annotations

import logging
import joblib
import numpy as np
import pandas as pd
from src.config import MAX_HORIZON, MAX_LAG, BACKTEST_HORIZON
from src.features import next_feature_row, validate_history
from src.models import BASELINE_LAGS
from src.utils import path, ensure_dir, save_json

LOG = logging.getLogger(__name__)


def recursive_forecast(spec: dict, history: pd.Series, hours: int = 24) -> pd.DataFrame:
    if isinstance(hours, bool) or not isinstance(hours, (int, np.integer)) or not 1 <= hours <= MAX_HORIZON:
        raise ValueError(f"hours must be an integer between 1 and {MAX_HORIZON}.")
    validate_history(history)
    past = history.iloc[-MAX_LAG:].copy().astype(float)
    rows = []
    for h in range(1, hours + 1):
        timestamp = past.index[-1] + pd.Timedelta(hours=1)
        if spec["name"] in BASELINE_LAGS:
            prediction = float(past.iloc[-BASELINE_LAGS[spec["name"]]])
        else:
            features = next_feature_row(timestamp, past)
            prediction = float(spec["estimator"].predict(features)[0])
        if not np.isfinite(prediction):
            raise ValueError("The model produced a non-finite prediction.")
        prediction = max(0.0, prediction)
        rows.append({"Datetime": timestamp, "horizon": h, "predicted_kw": prediction,
                     "predicted_kwh": prediction * 1.0})
        # For h>1, lag_1 is a PREDICTION, not the future measured target.
        past = pd.concat([past, pd.Series([prediction], index=[timestamp])]).iloc[-MAX_LAG:]
    return pd.DataFrame(rows).set_index("Datetime")


def load_bundle(run_id: str):
    """Only load trusted local files. Joblib/pickle is unsafe for untrusted input."""
    bundle_path = path("models", run_id, "forecast_bundle.joblib")
    if not bundle_path.exists():
        raise FileNotFoundError(f"No model for {run_id}. Run python main.py --dataset {run_id} first.")
    return joblib.load(bundle_path)


def forecast_bundle(bundle: dict, hours=24):
    forecast = recursive_forecast(bundle["model"], bundle["history"], hours)
    radius = float(bundle["validation_error_radius_kw"])
    forecast["lower_kw"] = np.where(forecast.horizon <= BACKTEST_HORIZON,
                                    np.maximum(0, forecast.predicted_kw - radius), np.nan)
    forecast["upper_kw"] = np.where(forecast.horizon <= BACKTEST_HORIZON,
                                    forecast.predicted_kw + radius, np.nan)
    return forecast


def save_forecast(run_id: str, hours=24):
    bundle = load_bundle(run_id)
    output = ensure_dir(path("outputs", run_id))
    extended = forecast_bundle(bundle, MAX_HORIZON)
    forecast = extended.iloc[:hours].copy()
    forecast.to_csv(output / "forecast.csv", float_format="%.6f")
    extended.to_csv(output / "forecast_168h.csv", float_format="%.6f")
    save_json({"run_id": run_id, "model": bundle["model"]["name"], "cutoff": bundle["cutoff"],
               "forecast_start": str(forecast.index.min()), "forecast_end": str(forecast.index.max()),
               "hours": hours, "total_kwh": float(forecast.predicted_kwh.sum()),
               "peak_kw": float(forecast.predicted_kw.max()),
               "peak_time": str(forecast.predicted_kw.idxmax()),
               "band_label": "Pooled 90th percentile of absolute validation 24-hour backtest errors; empirical, not guaranteed.",
               "band_available_through_horizon": BACKTEST_HORIZON,
               "warning": "No future actuals exist in this forecast. Beyond 24 h is exploratory and has no error band.",
               "source": bundle["source"]}, output / "forecast_meta.json")
    LOG.info("FORECAST | %s hours after %s | %.2f kWh | peak %.2f kW", hours, bundle["cutoff"],
             forecast.predicted_kwh.sum(), forecast.predicted_kw.max())
    return forecast
