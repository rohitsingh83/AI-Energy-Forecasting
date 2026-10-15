import numpy as np
import pandas as pd
import pytest
from src.features import build_features, next_feature_row, validate_history, FEATURE_NAMES
from src.evaluation import chronological_split, select_origins, rolling_backtest
from src.forecast import recursive_forecast, forecast_bundle
from src.models import fit_candidate, predict_rows


def test_feature_lags_and_rolling_are_past_only(hourly_frame):
    X, y = build_features(hourly_frame)
    t = X.index[0]
    past = hourly_frame.loc[:t-pd.Timedelta(hours=1), "Energy_kW"]
    assert X.loc[t, "lag_1"] == past.iloc[-1]
    assert X.loc[t, "lag_24"] == past.iloc[-24]
    assert X.loc[t, "lag_168"] == past.iloc[-168]
    assert np.isclose(X.loc[t, "rolling_mean_24"], past.tail(24).mean())
    assert np.isclose(X.loc[t, "rolling_std_24"], past.tail(24).std())
    assert X.columns.tolist() == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 15
    assert not any("Temperature" in x or "Occupancy" in x for x in X.columns)


def test_training_serving_feature_parity(hourly_frame):
    X, _ = build_features(hourly_frame)
    for t in X.index[::101]:
        past = hourly_frame.loc[:t-pd.Timedelta(hours=1), "Energy_kW"]
        np.testing.assert_allclose(next_feature_row(t, past).values, X.loc[[t]].values, atol=1e-10)


def test_imputed_target_is_not_training_truth(hourly_frame):
    t = hourly_frame.index[250]
    hourly_frame.loc[t, "target_observed"] = False
    X, y = build_features(hourly_frame)
    assert t not in X.index and t not in y.index


def test_future_target_cannot_change_current_features(hourly_frame):
    X1, _ = build_features(hourly_frame)
    t = X1.index[100]
    changed = hourly_frame.copy()
    changed.loc[t:, "Energy_kW"] = 100000
    X2, _ = build_features(changed)
    np.testing.assert_allclose(X1.loc[t], X2.loc[t])


def test_chronological_splits_do_not_overlap(hourly_frame):
    X, y = build_features(hourly_frame)
    s = chronological_split(X, y)
    assert s["train"][0].index.max() < s["validation"][0].index.min()
    assert s["validation"][0].index.max() < s["test"][0].index.min()
    assert sum(len(v[0]) for v in s.values()) == len(X)


def test_scaling_fits_training_only(hourly_frame):
    X, y = build_features(hourly_frame)
    spec, _, _ = fit_candidate("ridge", X.iloc[:500], y.iloc[:500])
    scaler = spec["estimator"].named_steps["standardscaler"]
    np.testing.assert_allclose(scaler.mean_, X.iloc[:500].mean().values)


@pytest.mark.parametrize("hours", [0, 169, 2.5, True, "24"])
def test_bad_horizon_rejected(hourly_frame, hours):
    with pytest.raises(ValueError):
        recursive_forecast({"name": "previous_hour", "estimator": None}, hourly_frame.Energy_kW, hours)


def test_persistence_forecast_is_constant_and_future(hourly_frame):
    f = recursive_forecast({"name": "previous_hour", "estimator": None}, hourly_frame.Energy_kW, 24)
    assert len(f) == 24 and f.predicted_kw.eq(hourly_frame.Energy_kW.iloc[-1]).all()
    assert f.index[0] == hourly_frame.index[-1]+pd.Timedelta(hours=1)


def test_weekly_baseline_uses_previous_week(hourly_frame):
    f = recursive_forecast({"name": "same_hour_last_week", "estimator": None}, hourly_frame.Energy_kW, 168)
    np.testing.assert_allclose(f.predicted_kw, hourly_frame.Energy_kW.tail(168))


class AddOne:
    def predict(self, X):
        return X.lag_1.to_numpy()+1


def test_recursive_step_reuses_prediction_not_future_truth(hourly_frame):
    f = recursive_forecast({"name": "fixture", "estimator": AddOne()}, hourly_frame.Energy_kW, 3)
    np.testing.assert_allclose(f.predicted_kw, hourly_frame.Energy_kW.iloc[-1]+np.arange(1,4))


def test_long_gap_or_short_history_fails(hourly_frame):
    with pytest.raises(ValueError):
        validate_history(hourly_frame.Energy_kW.head(100))
    bad = hourly_frame.Energy_kW.copy()
    bad.iloc[-10] = np.nan
    with pytest.raises(ValueError):
        validate_history(bad)
    with pytest.raises(ValueError):
        validate_history(hourly_frame.Energy_kW.drop(hourly_frame.index[-3]))


def test_band_stops_after_24_hours(hourly_frame):
    bundle = {"model": {"name": "previous_hour", "estimator": None},
              "history": hourly_frame.Energy_kW, "validation_error_radius_kw": 10}
    f = forecast_bundle(bundle, 48)
    assert f.lower_kw.iloc[:24].notna().all()
    assert f.lower_kw.iloc[24:].isna().all()
    assert (f.lower_kw.dropna() >= 0).all()


def test_backtest_never_uses_actual_future_inside_window(hourly_frame):
    origin = hourly_frame.index[480]
    spec = {"name": "same_hour_yesterday", "estimator": None}
    first = rolling_backtest(spec, hourly_frame, [origin])
    changed = hourly_frame.copy()
    changed.loc[origin:, "Energy_kW"] = 900
    second = rolling_backtest(spec, changed, [origin])
    np.testing.assert_allclose(first.predicted_kw, second.predicted_kw)
    assert not np.allclose(first.actual_kw, second.actual_kw)


def test_origin_windows_stay_within_split(hourly_frame):
    start, end = hourly_frame.index[500], hourly_frame.index[1000]
    origins = select_origins(hourly_frame, start, end, max_origins=4)
    assert len(origins) == 4
    assert min(origins) >= start
    assert max(origins)+pd.Timedelta(hours=23) <= end
