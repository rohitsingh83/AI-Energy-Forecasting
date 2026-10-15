import numpy as np
import pandas as pd
import pytest
from src.simulation import simulate, validate_scenario
from src.utils import compute_metrics, save_json, load_json


@pytest.fixture
def forecast():
    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    vals = np.array([30.]*24)
    vals[17:22] = 90
    return pd.DataFrame({"predicted_kw": vals}, index=idx)


@pytest.mark.parametrize("fraction", [0, .1, .3, .5])
def test_energy_cost_and_peak_constraints(forecast, fraction):
    frame, result = simulate(forecast, {"flexible_fraction": fraction})
    assert np.isclose(frame.baseline_kw.sum(), frame.shifted_kw.sum())
    assert frame.shifted_kw.min() >= 0
    assert frame.shifted_kw.max() <= frame.baseline_kw.max()+1e-9
    assert result["hypothetical_savings_inr"] >= -1e-8
    assert result["emissions_reduction_kg"] == 0
    assert result["baseline_emissions_kg"] == result["shifted_emissions_kg"]


def test_zero_flex_means_no_change(forecast):
    frame, result = simulate(forecast, {"flexible_fraction": 0})
    np.testing.assert_allclose(frame.baseline_kw, frame.shifted_kw)
    assert result["hypothetical_savings_inr"] == 0


def test_no_headroom_means_no_shift(forecast):
    forecast.predicted_kw = 20.0
    _, result = simulate(forecast)
    assert result["moved_energy_kwh"] == 0


@pytest.mark.parametrize("settings", [{"flexible_fraction": -.1}, {"flexible_fraction": .6},
 {"offpeak_rate": 20}, {"peak_rate": -1}, {"standard_rate": float('nan')},
 {"carbon_factor_kg_per_kwh": -1}, {"offpeak_rate": True}, {"unknown": 4}])
def test_invalid_scenario_settings(settings):
    with pytest.raises(ValueError):
        validate_scenario(settings)


def test_simulation_needs_24_hours(forecast):
    with pytest.raises(ValueError):
        simulate(forecast.iloc[:10])


def test_metrics_ignore_missing_truth_and_match_manual_rmse():
    result = compute_metrics([1, 2, np.nan, 4], [1, 3, 100, 5])
    assert result["n"] == 3
    assert np.isclose(result["rmse_kw"], np.sqrt(2/3))
    assert np.isclose(result["mae_kw"], 2/3)


def test_json_never_emits_nan(tmp_path):
    p = tmp_path / 'm.json'
    save_json({"x": np.nan, "y": np.float64(3), "z": [np.inf]}, p)
    assert load_json(p) == {"x": None, "y": 3.0, "z": [None]}
    assert 'NaN' not in p.read_text()
