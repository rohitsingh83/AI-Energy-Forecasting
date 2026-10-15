"""Headless, reproducible evidence figures. No screenshots of invented results."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.utils import path, ensure_dir, load_json
from src.data import load_processed

INK, TEAL, BLUE, AMBER = "#162a3b", "#0c9582", "#4376b8", "#d68a33"


def make_plots(run_id):
    sns.set_theme(style="whitegrid", context="notebook", palette=[TEAL, BLUE, AMBER])
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold",
                         "figure.facecolor": "#f8fafb", "axes.facecolor": "white",
                         "text.color": INK, "axes.labelcolor": INK, "savefig.facecolor": "#f8fafb"})
    out = ensure_dir(path("images", run_id))
    frame = load_processed(run_id)
    metrics = load_json(path("outputs", run_id, "metrics.json"))
    quality = load_json(path("outputs", run_id, "data_quality.json"))
    forecast = pd.read_csv(path("outputs", run_id, "forecast_168h.csv"), parse_dates=["Datetime"], index_col="Datetime")
    one = pd.read_csv(path("outputs", run_id, "one_step_predictions.csv"), parse_dates=["Datetime"], index_col="Datetime")
    bt = pd.read_csv(path("outputs", run_id, "backtest_predictions.csv"), parse_dates=["Datetime", "origin"])
    comp = pd.read_csv(path("outputs", run_id, "model_comparison.csv"))
    title = "SIMULATED BUILDING" if quality["source"]["source_type"] == "synthetic" else "PUBLIC HOUSEHOLD BENCHMARK"

    def save(fig, name, subtitle=None):
        if subtitle:
            fig.text(0.01, 0.005, subtitle, fontsize=8, color="#627588")
        fig.tight_layout(rect=(0, .035, 1, 1))
        fig.savefig(out / name, dpi=155, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    daily = frame.Energy_kW.resample("D").agg(["mean", "max"])
    ax.plot(daily.index, daily["max"], color=BLUE, alpha=.4, lw=.8, label="Daily maximum kW")
    ax.plot(daily.index, daily["mean"], color=TEAL, lw=1.2, label="Daily mean kW")
    ax.axvline(pd.Timestamp(metrics["splits"]["validation"]["start"]), color=AMBER, ls="--", label="Validation starts")
    ax.axvline(pd.Timestamp(metrics["splits"]["test"]["start"]), color=INK, ls="--", label="Test starts")
    ax.set(title=f"{title} / historical demand & chronological splits", ylabel="Average power (kW)")
    ax.legend(ncol=2, frameon=False, fontsize=8)
    save(fig, "energy_history.png", "Imputed observations excluded from training targets and evaluation. Time moves left to right.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    observed = frame.loc[frame.target_observed, "Energy_kW"]
    for weekend, label, color in [(False, "Weekday", TEAL), (True, "Weekend", BLUE)]:
        s = observed.loc[(observed.index.dayofweek >= 5) == weekend]
        pattern = s.groupby(s.index.hour).mean()
        axes[0].plot(pattern.index, pattern.values, marker="o", ms=3, color=color, label=label)
    axes[0].set(title="Hourly behavior", xlabel="Hour", ylabel="Mean kW", xticks=range(0, 24, 3))
    axes[0].legend(frameon=False)
    weekday = observed.groupby(observed.index.dayofweek).mean().reindex(range(7))
    axes[1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], weekday.values, color=[TEAL]*5+[BLUE]*2)
    axes[1].set(title="Weekly behavior", ylabel="Mean kW")
    save(fig, "usage_patterns.png", "Descriptive patterns only. Calendar, weather and human routines can change.")

    fig, ax = plt.subplots(figsize=(10, 4))
    labels = ["Missing before fill", "Filled from past (<=3 h)", "Still missing"]
    values = [quality["missing_target_hours_before_fill"], quality["imputed_target_hours"], quality["remaining_missing_target_hours"]]
    bars = ax.barh(labels, values, color=[AMBER, TEAL, BLUE])
    for b, v in zip(bars, values):
        ax.text(b.get_width() + max(1, max(values)*.02), b.get_y()+b.get_height()/2, f"{v:,}", va="center")
    ax.set(title="Data-quality audit / causal filling only", xlabel="Number of hourly target readings")
    ax.set_xlim(0, max(1, max(values))*1.2)
    save(fig, "data_quality.png", "No backward fill, no future interpolation, no imputed targets used as evaluation truth.")

    fig, ax = plt.subplots(figsize=(12, 5))
    pos = np.arange(len(comp))
    ax.barh(pos+.17, comp.validation_day_ahead_rmse_kw, height=.32, color=BLUE, alpha=.7, label="Validation / used for selection")
    ax.barh(pos-.17, comp.test_day_ahead_rmse_kw, height=.32, color=TEAL, label="Test / report only")
    ax.set_yticks(pos, [x + ("  [selected]" if s else "") for x, s in zip(comp.label, comp.selected)])
    ax.invert_yaxis()
    ax.set(title="Model evidence / recursive 24-hour RMSE (lower is better)", xlabel="RMSE (kW)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "model_comparison.png", "Champion is locked on validation BEFORE test. Both panels use up to 14 sampled daily forecast origins.")

    fig, ax = plt.subplots(figsize=(12, 4.6))
    tail = one.tail(168)
    ax.plot(tail.index, tail.actual_kw, color=INK, lw=1.3, label="Observed")
    ax.plot(tail.index, tail.predicted_kw, color=TEAL, lw=1.1, alpha=.9, label="One-step prediction")
    ax.set(title=f"Untouched test / one-step replay / {metrics['selected_model_label']}", ylabel="kW")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    save(fig, "actual_vs_predicted.png", "One-step evaluation updates meter lags every hour. It is NOT a fixed-origin 7-day forecast.")

    fig, ax = plt.subplots(figsize=(12, 4.8))
    hist = frame.Energy_kW.tail(48)
    future = forecast.head(48)
    ax.plot(hist.index, hist.values, color=INK, lw=1.5, label="Historical meter power")
    ax.plot(future.index, future.predicted_kw, color=TEAL, lw=1.6, label="Recursive future prediction")
    band = future.loc[future.lower_kw.notna()]
    ax.fill_between(band.index, band.lower_kw, band.upper_kw, color=TEAL, alpha=.15, label="Empirical validation error band (first 24 h)")
    ax.axvline(frame.index[-1], color=AMBER, ls="--", lw=1, label="Last observed hour")
    ax.set(title="Forecast from last dataset observation / not a live meter", ylabel="kW")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    save(fig, "forecast.png", "No future actuals are known. The error band is descriptive, not a coverage guarantee; horizons >24 h are exploratory.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    errors = bt.actual_kw - bt.predicted_kw
    axes[0].hist(errors.dropna(), bins=30, color=TEAL, alpha=.85, edgecolor="white")
    axes[0].axvline(0, color=INK, ls="--", lw=1)
    axes[0].set(title="Day-ahead residual distribution", xlabel="Actual - predicted (kW)", ylabel="Count")
    horizons = pd.read_csv(path("outputs", run_id, "horizon_metrics.csv"))
    axes[1].plot(horizons.horizon, horizons.rmse_kw, color=BLUE, marker="o", ms=3)
    axes[1].set(title="Error by forecast lead time", xlabel="Hours ahead", ylabel="RMSE (kW)", xticks=[1, 6, 12, 18, 24])
    save(fig, "residual_diagnostics.png", "Forecast error may grow with horizon; only 14 or fewer origins underpin each lead-time metric.")

    fi = pd.read_csv(path("outputs", run_id, "feature_importance.csv")).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(fi.feature, fi.importance, color=TEAL)
    ax.set(title="Random-forest feature importance / diagnostic", xlabel="Impurity-based relative importance")
    save(fig, "feature_importance.png", "Diagnostic for the RF candidate, not causal importance. Correlated lag features share importance; champion may differ.")

    sim = pd.read_csv(path("outputs", run_id, "simulation.csv"), parse_dates=["Datetime"], index_col="Datetime")
    scenario = load_json(path("outputs", run_id, "simulation.json"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    axes[0].plot(sim.index, sim.baseline_kw, color=INK, label="Forecast baseline")
    axes[0].plot(sim.index, sim.shifted_kw, color=TEAL, label="Hypothetical shifted load")
    axes[0].set(title="Virtual decision sandbox / first 24 hours", ylabel="kW")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    values = [scenario["baseline_cost_inr"], scenario["shifted_cost_inr"]]
    axes[1].bar(["Baseline", "Scenario"], values, color=[BLUE, TEAL])
    axes[1].set(title="Illustrative cost (INR)", ylabel="INR / 24 h")
    save(fig, "simulation.png", "Simulated savings, not achieved savings. Total energy is conserved; modeled CO2 is unchanged with constant carbon intensity.")
    return [str(f.name) for f in sorted(out.glob("*.png"))]
