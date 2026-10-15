"""Identical training/serving features, all available at forecast issuance time."""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.config import LAGS, MAX_LAG

FEATURE_NAMES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend",
                 *[f"lag_{lag}" for lag in LAGS], "rolling_mean_24", "rolling_std_24", "rolling_mean_168"]


def calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * index.hour / 24),
        "hour_cos": np.cos(2 * np.pi * index.hour / 24),
        "dow_sin": np.sin(2 * np.pi * index.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * index.dayofweek / 7),
        "doy_sin": np.sin(2 * np.pi * index.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * index.dayofyear / 365.25),
        "is_weekend": (index.dayofweek >= 5).astype(float),
    }, index=index)


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Target at t; lags/rolling windows contain only t-1 and earlier."""
    energy = frame["Energy_kW"]
    features = calendar(frame.index)
    for lag in LAGS:
        features[f"lag_{lag}"] = energy.shift(lag)
    past = energy.shift(1)  # critical anti-leakage step
    features["rolling_mean_24"] = past.rolling(24, min_periods=24).mean()
    features["rolling_std_24"] = past.rolling(24, min_periods=24).std(ddof=1)
    features["rolling_mean_168"] = past.rolling(168, min_periods=168).mean()
    mask = features.notna().all(axis=1) & energy.notna() & frame.target_observed
    return features.loc[mask, FEATURE_NAMES].astype(float), energy.loc[mask].astype(float)


def validate_history(history: pd.Series):
    if len(history) < MAX_LAG:
        raise ValueError(f"At least {MAX_LAG} hourly historical values are required.")
    tail = history.iloc[-MAX_LAG:]
    if not np.isfinite(tail.to_numpy(dtype=float)).all():
        raise ValueError("The last 168 hours contain a long gap. Repair input before forecasting.")
    if len(tail) > 1 and not np.all(np.diff(tail.index) == pd.Timedelta(hours=1)):
        raise ValueError("Forecast history must be on an uninterrupted hourly grid.")


def next_feature_row(timestamp: pd.Timestamp, history: pd.Series) -> pd.DataFrame:
    """Efficient feature construction for one recursive step; parity is unit-tested."""
    values = history.iloc[-MAX_LAG:].to_numpy(dtype=float)
    if len(values) < MAX_LAG or not np.isfinite(values).all():
        raise ValueError("168 finite preceding hourly values are required.")
    if timestamp != history.index[-1] + pd.Timedelta(hours=1):
        raise ValueError("Predict only the next contiguous hour; arbitrary hour/day input is insufficient.")
    row = calendar(pd.DatetimeIndex([timestamp]))
    for lag in LAGS:
        row[f"lag_{lag}"] = values[-lag]
    row["rolling_mean_24"] = values[-24:].mean()
    row["rolling_std_24"] = values[-24:].std(ddof=1)
    row["rolling_mean_168"] = values.mean()
    return row[FEATURE_NAMES].astype(float)
