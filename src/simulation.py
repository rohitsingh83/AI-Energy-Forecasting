"""Transparent load-shifting sandbox. Forecasts inform a hypothetical decision.

Not a control system, tariff quote, realized saving or emission-reduction claim.
Energy is conserved and no hour exceeds the original forecast maximum.
Flexible work may be pre-scheduled or deferred anywhere inside a 24-hour window.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from src.config import DEFAULT_SCENARIO
from src.utils import path, save_json

LOG = logging.getLogger(__name__)


def validate_scenario(overrides=None):
    settings = {**DEFAULT_SCENARIO}
    allowed = {"offpeak_rate", "standard_rate", "peak_rate", "flexible_fraction", "carbon_factor_kg_per_kwh"}
    if overrides:
        if set(overrides) - allowed:
            raise ValueError(f"Unknown settings: {sorted(set(overrides) - allowed)}")
        for k, value in overrides.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{k} must be a finite number.")
            settings[k] = float(value)
    if not 0 <= settings["offpeak_rate"] <= settings["standard_rate"] <= settings["peak_rate"] <= 1000:
        raise ValueError("Rates must satisfy 0 <= offpeak <= standard <= peak <= 1000 INR/kWh.")
    if not 0 <= settings["flexible_fraction"] <= 0.5:
        raise ValueError("flexible_fraction must be between 0 and 0.5.")
    if not 0 <= settings["carbon_factor_kg_per_kwh"] <= 5:
        raise ValueError("carbon factor must be between 0 and 5 kg/kWh.")
    return settings


def simulate(forecast: pd.DataFrame, overrides=None):
    if len(forecast) < 24:
        raise ValueError("A full 24-hour planning window is needed for simulation.")
    settings = validate_scenario(overrides)
    window = forecast.iloc[:24].copy()
    base = window.predicted_kw.to_numpy(dtype=float)
    if not np.isfinite(base).all() or (base < 0).any():
        raise ValueError("Scenario needs finite non-negative forecasts.")
    hours = window.index.hour
    peak = np.isin(hours, settings["peak_hours"])
    offpeak = np.isin(hours, settings["offpeak_hours"])
    rates = np.where(peak, settings["peak_rate"], np.where(offpeak, settings["offpeak_rate"], settings["standard_rate"]))
    requested = float((base[peak] * settings["flexible_fraction"]).sum())
    headroom = np.maximum(base.max() - base[offpeak], 0)
    moved = min(requested, float(headroom.sum()))
    shifted = base.copy()
    if requested > 0 and moved > 0:
        shifted[peak] -= base[peak] * settings["flexible_fraction"] * moved / requested
        shifted[offpeak] += moved * headroom / headroom.sum()
    window["baseline_kw"] = base
    window["shifted_kw"] = shifted
    window["rate_inr_per_kwh"] = rates
    window["baseline_cost_inr"] = base * rates
    window["shifted_cost_inr"] = shifted * rates
    before, after = float((base * rates).sum()), float((shifted * rates).sum())
    carbon = float(base.sum() * settings["carbon_factor_kg_per_kwh"])
    result = {"label": "HYPOTHETICAL forecast-based 24-hour load-shifting scenario",
              "settings": settings, "window_start": str(window.index.min()), "window_end": str(window.index.max()),
              "baseline_energy_kwh": float(base.sum()), "shifted_energy_kwh": float(shifted.sum()),
              "requested_shift_kwh": requested, "moved_energy_kwh": moved,
              "baseline_peak_kw": float(base.max()), "shifted_peak_kw": float(shifted.max()),
              "baseline_cost_inr": before, "shifted_cost_inr": after,
              "hypothetical_savings_inr": before - after,
              "hypothetical_savings_pct": (before - after) / before * 100 if before else 0,
              "baseline_emissions_kg": carbon, "shifted_emissions_kg": carbon,
              "emissions_reduction_kg": 0.0,
              "assumptions": ["Illustrative INR time-of-use rates; not a real utility tariff or billing quote.",
                              "10% of peak-hour forecast load flexible by default; configurable 0-50%.",
                              "Pre-scheduling/deferral allowed anywhere inside this 24-hour window.",
                              "All shifted load fits original forecast peak headroom; no equipment constraints modeled.",
                              "No demand charges, taxes, rebound, discomfort or forecast error in cost calculation.",
                              "Constant illustrative carbon intensity: shifting preserves energy and does NOT reduce modeled emissions.",
                              "Savings are simulated, not measured or guaranteed."]}
    assert np.isclose(base.sum(), shifted.sum()), "Energy-conservation failure"
    assert shifted.min() >= -1e-9 and shifted.max() <= base.max() + 1e-9
    return window, result


def save_simulation(run_id):
    forecast = pd.read_csv(path("outputs", run_id, "forecast_168h.csv"), parse_dates=["Datetime"], index_col="Datetime")
    frame, result = simulate(forecast)
    frame.to_csv(path("outputs", run_id, "simulation.csv"), float_format="%.6f")
    save_json(result, path("outputs", run_id, "simulation.json"))
    LOG.info("SIMULATION | moved %.2f kWh | hypothetical savings INR %.2f | energy conserved; modeled CO2 unchanged.",
             result["moved_energy_kwh"], result["hypothetical_savings_inr"])
    return result
