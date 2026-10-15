"""Chronological validation selection, untouched test reporting, then final refit.

Two different tasks are deliberately reported:
1. One-step: actual meter data are updated every hour.
2. Day-ahead: issue a 24-hour recursive forecast with NO updates within that day.
Never use random train_test_split for this forecasting project.
"""
from __future__ import annotations

import logging
import platform
import time
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
from src.config import (TRAIN_FRACTION, VALIDATION_FRACTION, BACKTEST_HORIZON,
                        BACKTEST_ORIGINS, INTERVAL_COVERAGE, MAX_LAG, SEED)
from src.features import build_features, validate_history, FEATURE_NAMES
from src.forecast import recursive_forecast
from src.models import candidate_names, fit_candidate, predict_rows, MODEL_LABELS
from src.utils import path, ensure_dir, save_json, compute_metrics

LOG = logging.getLogger(__name__)


def chronological_split(X, y):
    if len(X) < 720:
        raise ValueError("At least 720 usable feature rows are needed (recommend >=60 days of hourly data).")
    a = int(len(X) * TRAIN_FRACTION)
    b = int(len(X) * (TRAIN_FRACTION + VALIDATION_FRACTION))
    return {"train": (X.iloc[:a], y.iloc[:a]), "validation": (X.iloc[a:b], y.iloc[a:b]),
            "test": (X.iloc[b:], y.iloc[b:])}


def select_origins(frame, start, end, max_origins=BACKTEST_ORIGINS):
    """Evenly sample eligible daily issuance times across the given time split."""
    last = end - pd.Timedelta(hours=BACKTEST_HORIZON - 1)
    candidates = pd.date_range(pd.Timestamp(start).ceil("D"), last, freq="24h")
    eligible = []
    for origin in candidates:
        history = frame.loc[:origin - pd.Timedelta(hours=1), "Energy_kW"]
        try:
            validate_history(history)
        except ValueError:
            continue
        target = frame.reindex(pd.date_range(origin, periods=BACKTEST_HORIZON, freq="h"))
        if target.target_observed.fillna(False).sum() >= 18:
            eligible.append(origin)
    if not eligible:
        raise ValueError("No complete day-ahead windows in split; use a longer/less-gappy dataset.")
    indices = np.unique(np.linspace(0, len(eligible) - 1, min(max_origins, len(eligible))).astype(int))
    return [eligible[i] for i in indices]


def rolling_backtest(spec, frame, origins):
    blocks = []
    for origin in origins:
        history = frame.loc[:origin - pd.Timedelta(hours=1), "Energy_kW"]
        pred = recursive_forecast(spec, history, BACKTEST_HORIZON).reset_index()
        truth = frame.reindex(pd.DatetimeIndex(pred.Datetime))
        pred["actual_kw"] = truth.Energy_kW.where(truth.target_observed).to_numpy()
        pred["origin"] = origin
        pred["model"] = spec["name"]
        blocks.append(pred)
    return pd.concat(blocks, ignore_index=True)


def train_evaluate(frame, report, run_id, include_mlp=True, quick=False):
    started = time.perf_counter()
    output = ensure_dir(path("outputs", run_id))
    model_dir = ensure_dir(path("models", run_id))
    X, y = build_features(frame)
    splits = chronological_split(X, y)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]
    val_origins = select_origins(frame, y_val.index[0], y_val.index[-1])
    test_origins = select_origins(frame, y_test.index[0], y_test.index[-1])
    split_report = {name: {"rows": len(target), "start": str(target.index[0]), "end": str(target.index[-1])}
                    for name, (_, target) in splits.items()}
    save_json({"splits": split_report, "features": FEATURE_NAMES,
               "eligible_feature_rows": len(X), "dropped_rows_including_lag_warmup": len(frame) - len(X),
               "selection_rule": "Lowest validation 24-hour recursive backtest RMSE; test never selects model.",
               "validation_origins": [str(t) for t in val_origins],
               "test_origins": [str(t) for t in test_origins],
               "backtest_horizon": BACKTEST_HORIZON}, output / "split_report.json")
    X.head(24).join(y.rename("target_kw")).to_csv(output / "feature_preview.csv")
    LOG.info("SPLIT | train=%s | validation=%s | test=%s | features=%s", len(X_train), len(X_val), len(X_test), len(FEATURE_NAMES))

    validation = {}
    fitted_train = {}
    for name in candidate_names(include_mlp):
        LOG.info("VALIDATE | %s", MODEL_LABELS[name])
        spec, seconds, messages = fit_candidate(name, X_train, y_train, quick)
        one_step = compute_metrics(y_val, predict_rows(spec, X_val))
        backtest = rolling_backtest(spec, frame, val_origins)
        day_ahead = compute_metrics(backtest.actual_kw, backtest.predicted_kw)
        validation[name] = {"one_step": one_step, "day_ahead": day_ahead,
                            "fit_seconds": seconds, "warnings": messages, "backtest": backtest}
        fitted_train[name] = spec
        LOG.info("VALIDATION RESULT | %s | one-step RMSE %.3f | day-ahead RMSE %.3f kW",
                 name, one_step["rmse_kw"], day_ahead["rmse_kw"])
    champion = min(validation, key=lambda n: validation[n]["day_ahead"]["rmse_kw"])
    val_bt = validation[champion]["backtest"]
    errors = np.abs(val_bt.actual_kw - val_bt.predicted_kw).dropna()
    # Descriptive validation error band, not a formal time-series conformal guarantee.
    radius = float(np.quantile(errors, INTERVAL_COVERAGE, method="higher"))
    LOG.info("LOCKED SELECTION | %s (validation only). Test evaluation starts now.", champion)
    save_json({"model": champion, "selection_uses": "validation only", "error_radius_kw": radius,
               "coverage_target": INTERVAL_COVERAGE}, output / "selection.json")
    val_bt.to_csv(output / "validation_backtest.csv", index=False, float_format="%.6f")

    X_fit = pd.concat([X_train, X_val])
    y_fit = pd.concat([y_train, y_val])
    rows, all_test_backtests = [], []
    test_champion = None
    one_step_frame = None
    champion_bt = None
    for name in validation:
        LOG.info("TEST | refit train+validation then evaluate %s", name)
        spec, fit_seconds, messages = fit_candidate(name, X_fit, y_fit, quick)
        predictions = predict_rows(spec, X_test)
        one = compute_metrics(y_test, predictions)
        bt = rolling_backtest(spec, frame, test_origins)
        day = compute_metrics(bt.actual_kw, bt.predicted_kw)
        row = {"model": name, "label": MODEL_LABELS[name], "selected": name == champion,
               "validation_one_step_rmse_kw": validation[name]["one_step"]["rmse_kw"],
               "validation_day_ahead_rmse_kw": validation[name]["day_ahead"]["rmse_kw"],
               "test_one_step_rmse_kw": one["rmse_kw"], "test_one_step_mae_kw": one["mae_kw"],
               "test_one_step_r2": one["r2"], "test_day_ahead_rmse_kw": day["rmse_kw"],
               "test_day_ahead_mae_kw": day["mae_kw"], "test_day_ahead_r2": day["r2"],
               "test_day_ahead_wape_pct": day["wape_pct"], "fit_seconds_train_val": fit_seconds}
        rows.append(row)
        all_test_backtests.append(bt)
        if name == "random_forest":
            pd.DataFrame({"feature": FEATURE_NAMES, "importance": spec["estimator"].feature_importances_}).sort_values(
                "importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
        if name == champion:
            test_champion = {"one_step": one, "day_ahead": day}
            one_step_frame = pd.DataFrame({"actual_kw": y_test, "predicted_kw": predictions})
            one_step_frame.index.name = "Datetime"
            one_step_frame["error_kw"] = one_step_frame.actual_kw - one_step_frame.predicted_kw
            champion_bt = bt.copy()
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "model_comparison.csv", index=False, float_format="%.6f")
    pd.concat(all_test_backtests, ignore_index=True).to_csv(output / "backtest_all_models.csv", index=False, float_format="%.6f")
    one_step_frame.to_csv(output / "one_step_predictions.csv", float_format="%.6f")
    champion_bt["absolute_error_kw"] = np.abs(champion_bt.actual_kw - champion_bt.predicted_kw)
    champion_bt.to_csv(output / "backtest_predictions.csv", index=False, float_format="%.6f")
    horizon_rows = []
    for horizon, block in champion_bt.groupby("horizon"):
        horizon_rows.append({"horizon": int(horizon), **compute_metrics(block.actual_kw, block.predicted_kw)})
    pd.DataFrame(horizon_rows).to_csv(output / "horizon_metrics.csv", index=False)
    baseline = comparison.loc[comparison.model == "same_hour_last_week"].iloc[0]
    improvement = (1 - test_champion["day_ahead"]["rmse_kw"] / baseline.test_day_ahead_rmse_kw) * 100 if baseline.test_day_ahead_rmse_kw > 0 else None
    observed_errors = champion_bt.absolute_error_kw.dropna()
    coverage = float((observed_errors <= radius).mean() * 100)
    metrics = {"run_id": run_id, "selected_model": champion, "selected_model_label": MODEL_LABELS[champion],
               "source": report["source"], "seed": SEED, "quick_mode": quick,
               "selection_rule": "Minimum validation day-ahead RMSE, chosen BEFORE test evaluation.",
               "features": FEATURE_NAMES, "splits": split_report,
               "validation_day_ahead": validation[champion]["day_ahead"],
               "test_one_step": test_champion["one_step"], "test_day_ahead": test_champion["day_ahead"],
               "test_backtest_origins": len(test_origins), "test_backtest_horizon": BACKTEST_HORIZON,
               "day_ahead_rmse_improvement_vs_weekly_baseline_pct": improvement,
               "validation_error_radius_kw": radius, "empirical_test_band_coverage_pct": coverage,
               "band_note": "90th percentile absolute validation day-ahead error, pooled across horizons; no guaranteed coverage.",
               "evaluation_note": "One-step uses arriving actual lags; day-ahead recursively uses its own predictions. Metrics exclude imputed targets.",
               "limitations": ["Single site / synthetic data cannot establish grid reliability or realized savings.",
                               "Only 14 or fewer daily origins per split, sampled evenly; not every test day.",
                               "No observed future temperature/occupancy is used.",
                               "UTC/DST/holiday/weather integration and drift monitoring need production work."]}
    save_json(metrics, output / "metrics.json")

    # The test is now finished. Refit on all historical observations for FUTURE use.
    LOG.info("FINAL REFIT | %s on all %s eligible historical rows; reported test scores remain pre-refit.", champion, len(X))
    final_spec, seconds, messages = fit_candidate(champion, X, y, quick)
    validate_history(frame.Energy_kW)
    bundle = {"schema_version": 1, "run_id": run_id, "model": final_spec,
              "history": frame.Energy_kW.iloc[-MAX_LAG:].copy(), "cutoff": str(frame.index[-1]),
              "feature_names": FEATURE_NAMES, "validation_error_radius_kw": radius,
              "source": report["source"], "input_sha256": report["source"]["input_sha256"]}
    joblib.dump(bundle, model_dir / "forecast_bundle.joblib", compress=3)
    metadata = {k: v for k, v in bundle.items() if k not in {"model", "history"}}
    metadata.update({"model_name": champion, "final_refit_rows": len(X),
                     "python": platform.python_version(),
                     "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib"]},
                     "total_train_evaluate_seconds": time.perf_counter() - started,
                     "warnings": messages, "trust_notice": "Never load untrusted joblib/pickle files."})
    save_json(metadata, model_dir / "metadata.json")
    save_json(metadata, output / "run_manifest.json")
    LOG.info("COMPLETE | %s | day-ahead RMSE %.3f kW | R² %.3f | %.1fs", champion,
             test_champion["day_ahead"]["rmse_kw"], test_champion["day_ahead"]["r2"], time.perf_counter() - started)
    return metrics
