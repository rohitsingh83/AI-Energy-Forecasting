# Complete source code — copy-paste ready

Generated from the actual project files. Each heading is its target filename, followed by its purpose and full content. For beginner explanations read [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md). Edit source files and regenerate this book, not the other way around. Dataset/model binaries are separate artifacts, not hidden code.

The executed notebook is included separately; its code cells appear at the end without bulky embedded image/base64 outputs. All runtime logic is provided below.

## `generate_data.py`

Deterministic virtual-building data and honest synthetic metadata.

````python
"""Deterministic virtual-building data. This is simulation, not corporate logs.

Run: python generate_data.py --days 730 --seed 42
The target is average power during each hour (kW), not cumulative meter kWh.
Temperature and occupancy are EDA-only: future measurements are not assumed known.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import PROJECT_ROOT, save_json

OUT_PATH = PROJECT_ROOT / "data/raw/energy_data.csv"


def make_dataset(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    """Each call with the same arguments returns exactly the same dataset."""
    if n_days < 21:
        raise ValueError("At least 21 days are required for weekly-lag experiments.")
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days * 24, freq="h")
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    month = idx.month.to_numpy()
    doy = idx.dayofyear.to_numpy()
    n = len(idx)

    # A cosine peaks at 14:00. A sine with the same phase would peak at 20:00.
    daily_temp = 8 * np.cos(2 * np.pi * (hour - 14) / 24)
    seasonal_temp = 12 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    temperature = 20 + seasonal_temp + daily_temp + rng.normal(0, 2, n)
    daily_pattern = (30 * np.exp(-((hour - 8) ** 2) / 6)
                     + 45 * np.exp(-((hour - 19) ** 2) / 8) + 10)
    weekend_factor = np.where(dow >= 5, 0.85, 1.0)
    seasonal_factor = 1 + 0.15 * np.abs(np.cos(2 * np.pi * (month - 1) / 12))
    hvac = np.maximum(temperature - 24, 0) * 3.5 + np.maximum(16 - temperature, 0) * 2.5
    occupancy = np.clip(0.2 + 0.6 * ((hour >= 8) & (hour <= 18))
                        - 0.3 * (dow >= 5) + rng.normal(0, 0.05, n), 0, 1)
    energy = np.clip(50 + daily_pattern * weekend_factor * seasonal_factor
                     + hvac + occupancy * 15 + rng.normal(0, 4, n), 5, None)
    # Deliberately missing meter readings; never use imputed targets as test truth.
    energy[rng.choice(n, size=int(n * 0.005), replace=False)] = np.nan
    return pd.DataFrame({"Datetime": idx, "Energy_kW": np.round(energy, 6),
                         "Temperature_C": np.round(temperature, 2),
                         "Occupancy": np.round(occupancy, 3),
                         "IsWeekend": (dow >= 5).astype(int)})


def generate(n_days: int = 730, seed: int = 42, output: Path = OUT_PATH) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = make_dataset(n_days, seed)
    df.to_csv(output, index=False)
    save_json({"source_type": "synthetic", "title": "Virtual mixed-use building",
               "generator": "generate_data.py", "seed": seed, "days": n_days,
               "rows": len(df), "missing_target_rows": int(df.Energy_kW.isna().sum()),
               "unit": "hour-average kW", "start": str(df.Datetime.min()),
               "end": str(df.Datetime.max()), "timezone": "naive simulation clock",
               "warning": "Generated data. No real building, customer or corporate system."},
              output.with_suffix(".metadata.json"))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=OUT_PATH)
    args = p.parse_args()
    output = generate(args.days, args.seed, args.output)
    print(f"Saved {args.days * 24:,} simulated hourly rows -> {output}")
    print(pd.read_csv(output).head().to_string(index=False))


if __name__ == "__main__":
    main()
````

## `main.py`

CLI orchestration and phase selection; main entry point.

````python
"""Student-friendly CLI. Run everything with: python main.py

Examples:
  python main.py --dataset synthetic
  python main.py --dataset uci
  python main.py prepare --dataset synthetic
  python main.py train --dataset synthetic
  python main.py evaluate --dataset synthetic
  python main.py forecast --dataset synthetic --horizon 48
  python main.py simulate --dataset synthetic
  python main.py visualize --dataset synthetic
  python main.py --dataset csv --csv data/raw/my_building.csv --run-id building
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from src.config import DATASETS, MAX_HORIZON
from src.utils import configure_logging, path, load_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="run", choices=["run", "prepare", "train", "evaluate", "forecast", "simulate", "visualize"])
    p.add_argument("--dataset", choices=DATASETS, default="synthetic")
    p.add_argument("--csv", help="CSV containing Datetime and Energy_kW; used only for dataset=csv")
    p.add_argument("--run-id", help="Separate artifact name; defaults to dataset (or custom for CSV)")
    p.add_argument("--horizon", type=int, default=24, help="Future hourly predictions, 1..168 (default 24)")
    p.add_argument("--refresh-data", action="store_true", help="Regenerate/download input intentionally")
    p.add_argument("--no-mlp", action="store_true", help="Skip optional neural-network candidate")
    p.add_argument("--quick", action="store_true", help="Smaller models for smoke checks; not benchmark-equivalent")
    args = p.parse_args(argv)
    args.run_id = args.run_id if args.run_id is not None else ("custom" if args.dataset == "csv" else args.dataset)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", args.run_id):
        p.error("run-id must be 1..40 letters/digits/hyphens/underscores; no paths.")
    if not 1 <= args.horizon <= MAX_HORIZON:
        p.error("horizon must be between 1 and 168.")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Evaluation is read-only: never evaluate the final all-data-refit model on its training data.
    if args.command == "evaluate":
        print(json.dumps(load_json(path("outputs", args.run_id, "metrics.json")), indent=2))
        return 0
    configure_logging(args.run_id)
    try:
        if args.command in {"run", "prepare"}:
            from src.data import prepare
            frame, report = prepare(args.dataset, args.run_id, args.csv, args.refresh_data)
        if args.command in {"run", "train"}:
            from src.data import load_processed
            from src.evaluation import train_evaluate
            if args.command == "train":
                frame = load_processed(args.run_id)
                report = load_json(path("outputs", args.run_id, "data_quality.json"))
            metrics = train_evaluate(frame, report, args.run_id, not args.no_mlp, args.quick)
        if args.command in {"run", "forecast"}:
            from src.forecast import save_forecast
            forecast = save_forecast(args.run_id, args.horizon)
            print("\nFuture predictions (first five):\n", forecast.head().round(3).to_string())
        if args.command in {"run", "simulate"}:
            from src.simulation import save_simulation
            save_simulation(args.run_id)
        if args.command in {"run", "visualize"}:
            from src.visualization import make_plots
            figures = make_plots(args.run_id)
            logging.info("FIGURES | %s", ", ".join(figures))
        logging.info("DONE | artifacts: outputs/%s | images/%s | models/%s", args.run_id, args.run_id, args.run_id)
        if args.command == "run":
            print(f"\nSelected: {metrics['selected_model_label']}")
            print(f"Test day-ahead RMSE: {metrics['test_day_ahead']['rmse_kw']:.3f} kW")
            print(f"Test one-step R²: {metrics['test_one_step']['r2']:.3f}")
            print("Open the dashboard: python app.py")
        return 0
    except (ValueError, FileNotFoundError, OSError) as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
````

## `app.py`

Validated Flask JSON API, trusted bundle cache, downloads and dashboard.

````python
"""Flask API + dependency-free dashboard. Start: python app.py

The UI calls relative URLs, so it works behind a preview proxy. This is a
local educational server, not an authenticated, production grid-control API.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from src.forecast import load_bundle, forecast_bundle
from src.simulation import simulate
from src.utils import PROJECT_ROOT, path, load_json, json_safe

DOWNLOADS = {"forecast.csv", "forecast_168h.csv", "metrics.json", "model_comparison.csv",
             "one_step_predictions.csv", "backtest_predictions.csv", "simulation.csv", "simulation.json",
             "data_quality.json", "run_manifest.json", "horizon_metrics.csv", "feature_preview.csv"}
IMAGES = {"energy_history.png", "usage_patterns.png", "data_quality.png", "model_comparison.png",
          "actual_vs_predicted.png", "forecast.png", "residual_diagnostics.png", "feature_importance.png", "simulation.png"}


def available_runs():
    root = path("outputs")
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir()
                  and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", p.name)
                  and (p / "metrics.json").is_file()
                  and path("models", p.name, "forecast_bundle.joblib").is_file())


def select_run(value=None):
    run = value if value is not None else "synthetic"
    if not isinstance(run, str) or run not in available_runs():
        raise ValueError(f"Unknown/untrained dataset. Available: {', '.join(available_runs()) or 'none'}")
    return run


def int_parameter(value, name, low, high):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer in {low}..{high}.")
    if isinstance(value, str):
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError(f"{name} must be an integer in {low}..{high}.")
        number = int(value)
    elif isinstance(value, int):
        number = value
    else:
        raise ValueError(f"{name} must be an integer in {low}..{high}.")
    if not low <= number <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}.")
    return number


@lru_cache(maxsize=4)
def cached_bundle(run, mtime):
    return load_bundle(run)


@lru_cache(maxsize=12)
def cached_forecast(run, mtime, hours):
    return forecast_bundle(cached_bundle(run, mtime), hours)


def get_forecast(run, hours):
    mtime = path("models", run, "forecast_bundle.joblib").stat().st_mtime_ns
    return cached_bundle(run, mtime), cached_forecast(run, mtime, hours).copy()


def records(frame):
    return json_safe(frame.reset_index().to_dict(orient="records"))


def create_app():
    app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"))
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    app.json.sort_keys = False

    def response(value, status=200):
        return jsonify(json_safe(value)), status

    @app.get("/")
    def dashboard():
        return render_template("dashboard.html")

    @app.get("/health")
    def health():
        return response({"status": "ok", "ready_runs": available_runs(),
                         "mode": "historical dataset replay; not a live energy feed"})

    @app.get("/api/runs")
    def runs():
        items = []
        for run in available_runs():
            m = load_json(path("outputs", run, "metrics.json"))
            q = load_json(path("outputs", run, "data_quality.json"))
            items.append({"id": run, "title": m["source"]["title"], "source_type": m["source"]["source_type"],
                          "rows": q["hourly_rows"], "cutoff": q["end"],
                          "model": m["selected_model_label"]})
        return response({"runs": items})

    @app.get("/api/summary")
    def summary():
        run = select_run(request.args.get("dataset"))
        return response({"run_id": run, "metrics": load_json(path("outputs", run, "metrics.json")),
                         "quality": load_json(path("outputs", run, "data_quality.json")),
                         "forecast": load_json(path("outputs", run, "forecast_meta.json")),
                         "simulation": load_json(path("outputs", run, "simulation.json"))})

    @app.get("/api/history")
    def history():
        run = select_run(request.args.get("dataset"))
        hours = int_parameter(request.args.get("hours", "168"), "hours", 24, 8760)
        from src.data import load_processed
        frame = load_processed(run).tail(hours)
        return response({"run_id": run, "unit": "kW", "clock": "original dataset clock, no timezone conversion",
                         "data": records(frame[["Energy_kW", "target_observed"]])})

    def prediction_response(run, hours):
        bundle, forecast = get_forecast(run, hours)
        return response({"dataset": run, "model": bundle["model"]["name"], "cutoff": bundle["cutoff"],
                         "hours": hours, "unit": "kW (hour-average); kWh for each one-hour interval",
                         "total_kwh": float(forecast.predicted_kwh.sum()),
                         "peak_kw": float(forecast.predicted_kw.max()),
                         "peak_time": str(forecast.predicted_kw.idxmax()),
                         "band_note": "Empirical validation error band; only first 24 h, not guaranteed coverage.",
                         "source_type": bundle["source"]["source_type"], "predictions": records(forecast)})

    @app.get("/api/forecast")
    def forecast_get():
        run = select_run(request.args.get("dataset"))
        hours = int_parameter(request.args.get("hours", "24"), "hours", 1, 168)
        return prediction_response(run, hours)

    @app.post("/predict")
    def forecast_post():
        body = request.get_json(silent=False)
        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object with dataset and hours.")
        if set(body) - {"dataset", "hours"}:
            raise ValueError("Only dataset and hours are accepted. Forecasts require saved lag history, not arbitrary hour/day inputs.")
        run = select_run(body.get("dataset"))
        hours = int_parameter(body.get("hours", 24), "hours", 1, 168)
        return prediction_response(run, hours)

    @app.get("/api/evaluation")
    def evaluation():
        run = select_run(request.args.get("dataset"))
        one = pd.read_csv(path("outputs", run, "one_step_predictions.csv")).tail(168)
        bt = pd.read_csv(path("outputs", run, "backtest_predictions.csv"))
        last = bt.loc[bt.origin == bt.origin.max()]
        comp = pd.read_csv(path("outputs", run, "model_comparison.csv"))
        horizons = pd.read_csv(path("outputs", run, "horizon_metrics.csv"))
        return response({"run_id": run, "comparison": json_safe(comp.to_dict("records")),
                         "one_step": json_safe(one.to_dict("records")),
                         "day_ahead_example": json_safe(last.to_dict("records")),
                         "horizon_metrics": json_safe(horizons.to_dict("records"))})

    @app.post("/api/simulate")
    def scenario():
        body = request.get_json(silent=False)
        if not isinstance(body, dict) or set(body) - {"dataset", "settings"}:
            raise ValueError("JSON object may contain only dataset and settings.")
        run = select_run(body.get("dataset"))
        settings = body.get("settings", {})
        if not isinstance(settings, dict):
            raise ValueError("settings must be an object.")
        _, forecast = get_forecast(run, 24)
        window, result = simulate(forecast, settings)
        return response({"dataset": run, "result": result, "hourly": records(window)})

    @app.get("/download/<run>/<filename>")
    def download(run, filename):
        select_run(run)
        if filename not in DOWNLOADS:
            raise ValueError("This artifact is not available for download.")
        return send_file(path("outputs", run, filename), as_attachment=True, download_name=f"{run}_{filename}")

    @app.get("/images/<run>/<filename>")
    def image(run, filename):
        select_run(run)
        if filename not in IMAGES:
            raise ValueError("Unknown evidence image.")
        return send_file(path("images", run, filename))

    @app.errorhandler(ValueError)
    @app.errorhandler(BadRequest)
    def bad_input(exc):
        return response({"error": "invalid_request", "message": str(exc)}, 400)

    @app.errorhandler(FileNotFoundError)
    def not_ready(exc):
        return response({"error": "not_ready", "message": "Run python main.py for the requested dataset first."}, 409)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(exc):
        return response({"error": "request_too_large", "message": "JSON body must be <=16 KiB."}, 413)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
````

## `src/__init__.py`

Source package/version marker.

````python
"""AI-Powered Energy Consumption Forecasting - source package."""
__version__ = "1.0.0"
````

## `src/config.py`

Visible seeds, horizons, lag/split/fill settings and scenario assumptions.

````python
"""Small, explicit settings. No cloud account, private keys or GPU required."""
SEED = 42
LAGS = (1, 2, 24, 48, 168)
MAX_LAG = max(LAGS)
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
DEFAULT_HORIZON = 24
MAX_HORIZON = 168
BACKTEST_HORIZON = 24
BACKTEST_ORIGINS = 14
INTERVAL_COVERAGE = 0.90
MAX_FORWARD_FILL_HOURS = 3
DATASETS = ("synthetic", "uci", "csv")

# Illustrative INR/kWh values, not any real utility's approved tariff.
DEFAULT_SCENARIO = {
    "offpeak_rate": 5.0, "standard_rate": 8.0, "peak_rate": 12.0,
    "flexible_fraction": 0.10, "carbon_factor_kg_per_kwh": 0.70,
    "peak_hours": [17, 18, 19, 20, 21], "offpeak_hours": [0, 1, 2, 3, 4, 5],
}
````

## `src/data.py`

Official public archive adapter, schema validation, causal cleaning and masks.

````python
"""Download/ingest real or simulated data and clean it without future leakage."""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from src.config import MAX_FORWARD_FILL_HOURS
from src.utils import path, ensure_dir, save_json, load_json, sha256_file

UCI_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption"
LOG = logging.getLogger(__name__)


def download_uci(refresh: bool = False) -> Path:
    """Stream the official archive, aggregate in chunks, do not extract 126 MB.

    Require >=54 observed minutes per hour (90% coverage). Incomplete hours
    remain missing; the cleaner may use up to 3 causal forward-fill hours.
    Mean kW equals kWh only for a complete one-hour duration; the 90% rule is
    an explicit approximation for this student benchmark, not a billing method.
    """
    output = path("data", "raw", "uci_household_hourly.csv")
    if output.exists() and not refresh:
        LOG.info("Using cached public hourly dataset: %s", output)
        return output
    cache = ensure_dir(path(".cache", "downloads"))
    archive = cache / "uci_household.zip"
    if not archive.exists() or refresh:
        LOG.info("Downloading official UCI archive (~20 MB); internet needed once.")
        partial = archive.with_suffix(".part")
        with requests.get(UCI_URL, stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            with partial.open("wb") as f:
                size = 0
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        size += len(chunk)
                        if size > 30 * 1024 * 1024:
                            raise ValueError("Unexpected archive size; download refused.")
                        f.write(chunk)
        partial.replace(archive)
    parts = []
    total_minutes = 0
    with zipfile.ZipFile(archive) as z:
        with z.open("household_power_consumption.txt") as f:
            reader = pd.read_csv(f, sep=";", usecols=["Date", "Time", "Global_active_power"],
                                 na_values=["?", ""], chunksize=250_000)
            for chunk in reader:
                dt = pd.to_datetime(chunk.Date + " " + chunk.Time, format="%d/%m/%Y %H:%M:%S")
                p = pd.to_numeric(chunk.Global_active_power, errors="coerce")
                p = p.where((p >= 0) & np.isfinite(p))
                s = pd.Series(p.to_numpy(), index=dt)
                hourly = pd.DataFrame({"power_sum": s.resample("h").sum(min_count=1),
                                       "Observed_minutes": s.resample("h").count(),
                                       "Source_minutes": pd.Series(1, index=dt).resample("h").sum()})
                parts.append(hourly)
                total_minutes += len(chunk)
    # Some hours span chunk boundaries: combine sums/counts before averaging.
    grouped = pd.concat(parts).groupby(level=0).sum(min_count=1).sort_index()
    grouped["Energy_kW"] = (grouped.power_sum / grouped.Observed_minutes).where(
        grouped.Observed_minutes >= 54)
    out = grouped[["Energy_kW", "Observed_minutes", "Source_minutes"]]
    out.index.name = "Datetime"
    ensure_dir(output.parent)
    out.to_csv(output, float_format="%.6f")
    save_json({"source_type": "public_measured", "title": "UCI Individual Household Electric Power Consumption",
               "creators": "Georges Hebrail and Alice Berard", "source_url": UCI_PAGE,
               "doi": "10.24432/C58K54", "license": "CC BY 4.0",
               "license_url": "https://creativecommons.org/licenses/by/4.0/",
               "location": "One household in Sceaux, France", "raw_minute_rows": total_minutes,
               "hourly_rows": len(out), "minimum_observed_minutes_per_hour": 54,
               "archive_sha256": sha256_file(archive), "hourly_sha256": sha256_file(output),
               "transformation": "Hour-average of Global_active_power (kW); <54 valid minutes => missing.",
               "timezone": "Original local calendar labels; timezone/DST not reconstructed.",
               "warning": "Historical single-household benchmark, not current grid or corporate data."},
              output.with_suffix(".metadata.json"))
    LOG.info("Aggregated %s minute rows into %s hours.", f"{total_minutes:,}", f"{len(out):,}")
    return output


def resolve_dataset(dataset: str, csv_path=None, refresh=False) -> tuple[Path, dict]:
    if dataset == "synthetic":
        raw = path("data", "raw", "energy_data.csv")
        if refresh or not raw.exists():
            from generate_data import generate
            generate(output=raw)
    elif dataset == "uci":
        raw = download_uci(refresh)
    elif dataset == "csv":
        if csv_path is None:
            raise ValueError("--csv PATH is required with --dataset csv.")
        raw = Path(csv_path).expanduser().resolve()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if not raw.is_file():
        raise FileNotFoundError(raw)
    meta_path = raw.with_suffix(".metadata.json")
    meta = load_json(meta_path) if meta_path.exists() else {
        "source_type": "user_supplied", "title": raw.name,
        "warning": "User must verify permission, power units and timestamp semantics."}
    meta["input_sha256"] = sha256_file(raw)
    meta["input_file"] = raw.name
    return raw, meta


def clean_data(raw: Path, max_ffill: int = MAX_FORWARD_FILL_HOURS) -> tuple[pd.DataFrame, dict]:
    """Mean power on an hourly grid. Past-only bounded filling, never bfill.

    The target_observed mask survives imputation. Evaluation and fitting use
    real target observations only. Long gaps stay NaN and invalidate lag rows.
    """
    df = pd.read_csv(raw)
    required = {"Datetime", "Energy_kW"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain Datetime and Energy_kW (hour-average kW).")
    rows_in = len(df)
    if rows_in == 0:
        raise ValueError("Empty input dataset.")
    dt = pd.to_datetime(df["Datetime"], errors="coerce")
    if dt.isna().any():
        raise ValueError(f"{int(dt.isna().sum())} invalid timestamp(s). Fix the input CSV.")
    if getattr(dt.dt, "tz", None) is not None:
        raise ValueError("Use a timezone-naive, documented site clock or normalize to UTC before import.")
    df = df.assign(Datetime=dt).set_index("Datetime").sort_index()
    numeric = [c for c in ["Energy_kW", "Temperature_C", "Occupancy"] if c in df]
    df = df[numeric].apply(pd.to_numeric, errors="coerce")
    invalid_power = int(((df.Energy_kW < 0) | np.isinf(df.Energy_kW)).sum())
    df.loc[(df.Energy_kW < 0) | ~np.isfinite(df.Energy_kW), "Energy_kW"] = np.nan
    duplicates = int(df.index.duplicated().sum())
    # Mean, not sum: target is average power, not per-record energy.
    hourly = df.resample("h").mean()
    if hourly.Energy_kW.notna().sum() == 0:
        raise ValueError("No valid non-negative power observations.")
    # Do not forecast from partial/missing trailing hours; drop unobserved edges.
    hourly = hourly.loc[hourly.Energy_kW.first_valid_index():hourly.Energy_kW.last_valid_index()].copy()
    observed = hourly.Energy_kW.notna()
    missing_before = int((~observed).sum())
    hourly[numeric] = hourly[numeric].ffill(limit=max_ffill)
    hourly["target_observed"] = observed.astype(bool)
    hourly["Energy_kWh"] = hourly.Energy_kW * 1.0  # exactly a one-hour interval
    hourly.index.name = "Datetime"
    report = {"input_rows": rows_in, "hourly_rows": len(hourly),
              "start": str(hourly.index.min()), "end": str(hourly.index.max()),
              "duplicate_timestamps_aggregated": duplicates, "invalid_power_readings": invalid_power,
              "missing_target_hours_before_fill": missing_before,
              "imputed_target_hours": int((~observed & hourly.Energy_kW.notna()).sum()),
              "remaining_missing_target_hours": int(hourly.Energy_kW.isna().sum()),
              "max_causal_forward_fill_hours": max_ffill,
              "observed_target_hours": int(observed.sum()),
              "fill_policy": "Past-only forward-fill <=3 hours; no interpolation/backfill; imputed targets not scored.",
              "unit": "hour-average kW; hourly kWh = kW x 1 h"}
    return hourly, report


def prepare(dataset: str, run_id: str, csv_path=None, refresh=False):
    raw, meta = resolve_dataset(dataset, csv_path, refresh)
    frame, report = clean_data(raw)
    output = ensure_dir(path("outputs", run_id))
    processed = ensure_dir(path("data", "processed", run_id))
    frame.to_csv(processed / "hourly_energy.csv", float_format="%.6f")
    report.update({"dataset": dataset, "run_id": run_id, "source": meta})
    save_json(report, output / "data_quality.json")
    frame.head(24).to_csv(output / "dataset_preview.csv")
    LOG.info("DATA | %s: %s hours; %s filled; %s still missing.", run_id, len(frame),
             report["imputed_target_hours"], report["remaining_missing_target_hours"])
    return frame, report


def load_processed(run_id):
    p = path("data", "processed", run_id, "hourly_energy.csv")
    if not p.exists():
        raise FileNotFoundError(f"Run prepare first: {p}")
    frame = pd.read_csv(p, parse_dates=["Datetime"], index_col="Datetime")
    if frame.target_observed.dtype != bool:
        frame["target_observed"] = frame.target_observed.astype(str).str.lower().eq("true")
    return frame
````

## `src/evaluation.py`

Ordered selection, recursive backtests, untouched test reports and final refit.

````python
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
````

## `src/features.py`

Past-only vectorized training features and identical inference row.

````python
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
    if len(tail) > 1 and not np.all(np.diff(tail.index.asi8) == pd.Timedelta(hours=1).value):
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
````

## `src/forecast.py`

Recursive future prediction, trusted bundle loading and limited error bands.

````python
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
````

## `src/models.py`

Naive baselines, Ridge, random forest, scaled MLP and fit warnings.

````python
"""CPU-only candidates: three honest baselines and three small ML models."""
from __future__ import annotations

import logging
import time
import warnings
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.config import SEED

BASELINE_LAGS = {"previous_hour": 1, "same_hour_yesterday": 24, "same_hour_last_week": 168}
MODEL_LABELS = {"previous_hour": "Previous-hour baseline", "same_hour_yesterday": "Yesterday baseline",
                "same_hour_last_week": "Last-week baseline", "ridge": "Ridge regression",
                "random_forest": "Random forest", "mlp": "MLP neural network"}
LOG = logging.getLogger(__name__)


def candidate_names(include_mlp=True):
    return [*BASELINE_LAGS, "ridge", "random_forest"] + (["mlp"] if include_mlp else [])


def make_estimator(name: str, quick=False):
    if name in BASELINE_LAGS:
        return None
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=35 if quick else 90, max_depth=12,
                                     min_samples_leaf=6, max_features=0.9,
                                     n_jobs=-1, random_state=SEED)
    if name == "mlp":
        # Both X and y are standardized using TRAINING data only.
        # early_stopping=False avoids a randomly held-out internal validation set.
        regressor = make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(48, 24), activation="relu", solver="adam",
            max_iter=60 if quick else 220, batch_size=256, learning_rate_init=0.001,
            early_stopping=False, shuffle=False, random_state=SEED,
            tol=1e-4, n_iter_no_change=20))
        return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())
    raise ValueError(f"Unknown model: {name}")


def fit_candidate(name, X, y, quick=False):
    estimator = make_estimator(name, quick)
    started = time.perf_counter()
    messages = []
    if estimator is not None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(X, y)
        messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        for message in messages:
            LOG.warning("%s: %s", name, message)
        # Parallel training helps; a one-row forecast is faster without thread setup.
        if name == "random_forest":
            estimator.n_jobs = 1
    return {"name": name, "estimator": estimator}, time.perf_counter() - started, messages


def predict_rows(spec, X):
    """One-step evaluation: actual lag readings arrive between predictions."""
    import numpy as np
    if spec["name"] in BASELINE_LAGS:
        return np.maximum(0, X[f"lag_{BASELINE_LAGS[spec['name']]}"] .to_numpy())
    return np.maximum(0, spec["estimator"].predict(X))
````

## `src/simulation.py`

Transparent energy-conserving/headroom-limited load-shifting heuristic.

````python
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
````

## `src/utils.py`

Shared paths, finite metrics, strict JSON, hashes and real logs.

````python
"""Shared paths, strict JSON serialization, reproducible fingerprints and metrics."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def path(*parts) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_safe(value):
    """JSON has no NaN/Infinity. Convert these to null, including nested values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_json(obj, filepath):
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    filepath.write_text(json.dumps(json_safe(obj), indent=2, allow_nan=False), encoding="utf-8")


def load_json(filepath):
    return json.loads(Path(filepath).read_text(encoding="utf-8"))


def sha256_file(filepath) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_metrics(actual, predicted) -> dict:
    """Ignore unobserved targets; R² is not a percentage accuracy."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    if len(a) == 0:
        raise ValueError("No observed targets available for evaluation.")
    nonzero = np.abs(a) > 1e-8
    return {"n": int(len(a)), "mae_kw": float(mean_absolute_error(a, p)),
            "rmse_kw": float(np.sqrt(mean_squared_error(a, p))),
            "r2": float(r2_score(a, p)) if len(a) > 1 else None,
            "mape_pct": float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100)
            if nonzero.any() else None,
            "wape_pct": float(np.abs(a - p).sum() / np.abs(a).sum() * 100)
            if np.abs(a).sum() else None}


def configure_logging(run_id: str):
    log_dir = ensure_dir(path("outputs", run_id, "logs"))
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    root.setLevel(logging.INFO)
    for handler in [logging.StreamHandler(), logging.FileHandler(log_dir / "pipeline.log", mode="a", encoding="utf-8")]:
        handler.setFormatter(fmt)
        root.addHandler(handler)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
````

## `src/visualization.py`

Nine reproducible analytical plots from actual saved evidence.

````python
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
````

## `templates/dashboard.html`

Complete self-contained responsive five-view UI and offline snapshot logic.

````html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Volt / Energy Forecasting Lab</title>
<style>
:root{--paper:#f3f6f5;--white:#fff;--ink:#172c36;--muted:#71818a;--line:#e3eae7;--teal:#098974;--teal-light:#e8f5ef;--blue:#668bb7;--amber:#b78235;--dark:#152c36;--radius:15px}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px;line-height:1.5}button,input,select{font:inherit}button,a,select,input[type=range]{-webkit-tap-highlight-color:transparent}button,a{transition:background .18s,color .18s}button{cursor:pointer}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #52b9a5;outline-offset:3px}a{color:var(--teal);text-decoration:none}svg{display:block}button:disabled{cursor:wait;opacity:.6}.layout{display:flex;min-height:100vh}.sidebar{width:224px;background:var(--dark);color:#b8c8cd;position:fixed;inset:0 auto 0 0;display:flex;flex-direction:column;padding:30px 18px 22px;z-index:10}.brand{display:flex;align-items:center;gap:12px;padding:0 14px;color:white}.brand-mark{width:35px;height:39px}.brand-name{font-size:25px;font-weight:760;letter-spacing:-1.1px}.brand-label{font-size:9px;letter-spacing:2.5px;color:#91aaaF;display:block;line-height:1.2}.nav-label{font-size:10px;letter-spacing:1.8px;color:#68838e;margin:47px 18px 14px}.nav{display:flex;flex-direction:column;gap:7px}.nav button{border:0;background:none;color:#a7bdc5;text-align:left;display:flex;align-items:center;gap:12px;padding:12px 15px;border-radius:9px;font-size:13px}.nav button svg{width:18px;height:18px;opacity:.8}.nav button:hover{background:#203e49;color:white}.nav button.active{background:#26464e;color:#fff}.nav button.active svg{color:#7bdec2;opacity:1}.sidebar-bottom{margin-top:auto;padding:20px 14px 0}.lab-badge{border:1px solid #39525c;border-radius:11px;padding:15px 13px}.lab-badge span{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#89cbb5}.lab-badge p{font-size:11px;color:#89a5af;line-height:1.7;margin:8px 0 0}.sidebar-sign{font-size:10px;color:#617f8a;padding-top:21px}.main{margin-left:224px;width:calc(100% - 224px)}.topbar{height:77px;padding:0 36px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;background:#fafcfa}.crumb{font-size:12px;color:var(--muted)}.crumb strong{font-weight:550;color:var(--ink);margin-left:8px}.top-right{display:flex;gap:17px;align-items:center}.status{font-size:11px;display:flex;align-items:center;gap:7px;color:#58796d}.dot{height:6px;width:6px;background:var(--teal);border-radius:50%;display:inline-block}.avatar{background:#e9eeeb;border:1px solid #dbe4df;width:31px;height:31px;border-radius:50%;display:grid;place-items:center;font-size:10px;color:#61786e;font-weight:700}.content{max-width:1500px;margin:auto;padding:31px 36px 20px}.page{display:none}.page.active{display:block}.eyebrow{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--teal);display:flex;align-items:center;gap:8px}.heading-row{display:flex;justify-content:space-between;align-items:flex-end;gap:18px;margin-bottom:24px}h1{font-size:31px;letter-spacing:-1.1px;line-height:1.25;margin:9px 0 9px;font-weight:640}h2{font-size:16px;letter-spacing:-.2px;font-weight:640;margin:0}h3{font-size:13px;margin:0;font-weight:650}.subtitle{color:var(--muted);font-size:12px;margin:0;max-width:630px;line-height:1.8}.btn{border:1px solid #d5dfd9;background:white;color:var(--ink);padding:10px 15px;border-radius:8px;font-size:11px;font-weight:600;display:inline-flex;align-items:center;justify-content:center;gap:8px;white-space:nowrap}.btn:hover{background:#eaf3ed}.btn.primary{background:var(--teal);border-color:var(--teal);color:white}.btn.primary:hover{background:#087562}.btn svg{width:14px;height:14px}.source-bar{border:1px solid var(--line);border-radius:10px;background:#edf3f0;display:flex;align-items:center;justify-content:space-between;padding:11px 16px;gap:14px;margin-bottom:22px}.source-left{display:flex;align-items:center;gap:13px}.source-text{font-size:11px;color:#5b726a}.pill{border:1px solid #cee3d6;background:#f7fffa;color:#367655;font-size:9px;font-weight:700;letter-spacing:.8px;border-radius:5px;padding:4px 7px;white-space:nowrap}.source-bar select{border:0;border-left:1px solid #d5e0d9;padding:4px 0 4px 17px;background:transparent;color:var(--ink);font-size:11px;max-width:330px;cursor:pointer}.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:23px}.card{background:var(--white);border:1px solid var(--line);border-radius:var(--radius);box-shadow:0 3px 13px rgba(15,47,34,.018)}.kpi{padding:19px 21px 17px;min-height:128px}.kpi-top{display:flex;justify-content:space-between;align-items:center}.kpi-label{font-size:11px;color:var(--muted);font-weight:500}.kpi-icon{width:25px;height:25px;border-radius:7px;background:#f0f6f2;display:grid;place-items:center;color:#6a9380;font-size:12px}.kpi-value{font-size:27px;letter-spacing:-.8px;font-weight:650;margin-top:7px;line-height:1.25}.kpi-value small{font-size:13px;font-weight:450;color:#83908a;margin-left:4px;letter-spacing:0}.kpi-foot{font-size:10px;color:#88958e;margin-top:9px;display:flex;align-items:center;gap:6px}.kpi-foot strong{font-weight:600;color:var(--teal)}.overview-grid{display:grid;grid-template-columns:minmax(0,3.3fr) minmax(218px,1fr);gap:18px;margin-bottom:20px}.chart-card{padding:23px 23px 15px;min-width:0}.card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;margin-bottom:14px}.card-sub{font-size:10px;color:var(--muted);margin-top:5px}.seg{display:flex;padding:3px;border:1px solid var(--line);border-radius:7px;background:#fafcfb;gap:2px}.seg button{border:0;background:transparent;color:#8c9994;font-size:10px;padding:5px 10px;border-radius:4px;white-space:nowrap}.seg button.active{background:white;color:var(--teal);box-shadow:0 1px 4px #dbe3de}.legend{display:flex;align-items:center;gap:16px;color:#76847d;font-size:10px;margin:14px 0 2px;flex-wrap:wrap}.legend span{display:flex;align-items:center;gap:6px}.swatch{width:16px;height:2px;border-radius:2px;background:var(--ink)}.swatch.teal{background:var(--teal)}.swatch.band{height:8px;background:#e0f1e7;border:1px solid #c8e5d4}.chart-wrap{width:100%;overflow:hidden;min-height:240px}.chart-wrap svg{width:100%;height:auto}.chart-note{font-size:9px;color:#94a29a;margin-top:6px}.brief{padding:23px 21px;display:flex;flex-direction:column;background:linear-gradient(145deg,#fff,#f7fbf8)}.brief-top{font-size:10px;color:var(--teal);font-weight:700;letter-spacing:1px;text-transform:uppercase;display:flex;align-items:center;gap:7px}.brief-time{font-size:36px;line-height:1.2;letter-spacing:-1px;margin:20px 0 2px;font-weight:650}.brief-info{font-size:11px;color:#7d8c84}.brief-line{height:1px;background:var(--line);margin:21px 0 17px}.brief p{font-size:11px;line-height:1.8;color:#687f73;margin:8px 0 20px}.brief .btn{margin-top:auto;font-size:10px;width:100%}.brief-model{font-size:11px;color:#3d6150;font-weight:650;margin-top:6px}.lower-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.mini-card{padding:20px 21px}.mini-card .mini-icon{font-size:9px;color:var(--teal);margin-bottom:11px;font-weight:750;letter-spacing:1px}.mini-value{font-size:20px;font-weight:650;letter-spacing:-.5px;margin:7px 0}.mini-description{font-size:10px;color:#89998f;line-height:1.7}.progress{height:4px;background:#edf3ee;border-radius:4px;margin:13px 0 9px;overflow:hidden}.progress div{height:100%;background:var(--teal);border-radius:4px}.feature-chips{display:flex;gap:5px;flex-wrap:wrap;margin:14px 0 8px}.feature-chips span{background:#f0f5f2;border:1px solid #e5ede7;font-size:9px;color:#6c8173;padding:4px 6px;border-radius:4px}.footer{display:flex;justify-content:space-between;border-top:1px solid var(--line);margin-top:27px;padding:17px 1px 2px;color:#97a59c;font-size:9px;gap:12px}.footer strong{color:#6e897b;font-weight:550}.notice{font-size:11px;color:#7a6748;background:#faf6ec;border:1px solid #eae0c9;padding:12px 15px;border-radius:9px;line-height:1.7;margin:18px 0}.notice.info{color:#5e7e6d;background:#edf6f0;border-color:#d9e9de}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:20px}.table-card{margin-top:20px;overflow:auto;padding:22px}.table-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:15px}table{border-collapse:collapse;width:100%;font-size:11px;text-align:left}th{font-size:9px;letter-spacing:.55px;text-transform:uppercase;color:#8b9b90;font-weight:650;padding:11px 12px;background:#f8fbf9;border-bottom:1px solid var(--line)}td{padding:11px 12px;border-bottom:1px solid #edf2ee;color:#5c7668}tr.selected td{background:#f0faf4;color:#22684e}td strong{font-weight:650;color:var(--ink)}.selected-badge{font-size:8px;background:#dcf0e4;color:#367554;border-radius:4px;padding:3px 5px;margin-left:7px}.controls-card{padding:24px}.sandbox-grid{display:grid;grid-template-columns:300px minmax(0,1fr);gap:20px}.field{margin:21px 0}.field label{font-size:11px;color:#61776b;display:flex;justify-content:space-between;margin-bottom:8px}.field label strong{color:var(--teal)}.field input[type=number]{width:100%;border:1px solid #dce5de;border-radius:7px;background:#f9fcfa;padding:9px 11px;color:var(--ink);font-size:12px}.field input[type=range]{width:100%;accent-color:var(--teal)}.field-note{font-size:9px;color:#95a195;line-height:1.7}.scenario-kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:18px}.scenario-kpis .kpi{padding:17px;min-height:113px}.scenario-kpis .kpi-value{font-size:23px}.metric-row{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding:11px 0;font-size:11px;color:#738778}.metric-row strong{color:var(--ink);font-weight:550}.runbook{display:grid;grid-template-columns:1.25fr 1fr;gap:20px}.codebox{margin-top:18px;background:#142f35;border-radius:10px;padding:18px;color:#bde0cb;font-size:11px;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;line-height:1.9;overflow:auto;white-space:pre}.codebox .comment{color:#71998a}.steps{margin:17px 0;padding-left:19px;color:#6e8375;font-size:11px;line-height:2.2}.artifact-grid{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:17px}.artifact-grid a{padding:10px;border:1px solid var(--line);border-radius:7px;font-size:10px;color:#5e7c6c;background:#f8fcfa}.artifact-grid a:hover{background:#eaf5ed}.loading{opacity:.65;pointer-events:none}.error{display:none;position:fixed;bottom:22px;right:25px;z-index:100;max-width:440px;border:1px solid #edc5b1;background:#fff9f2;color:#925733;border-radius:10px;padding:15px 19px;box-shadow:0 3px 22px #162b3622;font-size:12px}.empty{padding:45px;text-align:center;color:#6d8175;font-size:13px}.hidden{display:none!important}.inline-label{font-size:9px;color:#8e9f93;letter-spacing:.7px;text-transform:uppercase}.small-link{font-size:10px;font-weight:600}.page-heading{margin-bottom:22px}.page-heading h1{font-size:28px}.evidence-top{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.evidence-top .kpi{min-height:112px}.small-chart{padding:23px}.small-chart .chart-wrap{min-height:180px}.quality-list{margin-top:11px}.sources-note{font-size:10px;color:#7f9084;line-height:1.8;padding-top:14px}.clock{font-size:10px;color:#7e9286}.chart-tooltip{position:absolute;pointer-events:none;display:none;font-size:11px;background:#15362ff2;color:#eaf7ed;border-radius:7px;padding:9px 12px;box-shadow:0 2px 12px #243d2822;z-index:20;white-space:nowrap}.chart-wrap{position:relative}.range-readout{font-size:10px;color:#799081;margin-top:4px}.page .full-chart{padding:25px}.muted{color:var(--muted)}
@media(min-width:1550px){.content{padding-top:40px}.chart-wrap{min-height:280px}.kpi{min-height:144px}}
@media(max-width:1150px){.sidebar{width:192px;padding-left:12px;padding-right:12px}.main{margin-left:192px;width:calc(100% - 192px)}.content{padding:26px 25px 20px}.topbar{padding:0 25px}.kpi{padding:16px}.kpi-value{font-size:24px}.overview-grid{grid-template-columns:minmax(0,2.7fr) minmax(180px,1fr)}.chart-card{padding:19px 17px 13px}.brief{padding:19px 17px}.brief-time{font-size:31px}.sandbox-grid{grid-template-columns:260px 1fr}.source-text{font-size:10px}.source-bar select{max-width:245px}.lower-grid{gap:12px}.mini-card{padding:17px}}
@media(max-width:930px){.sidebar{width:78px;padding:26px 10px}.brand{padding:0 10px}.brand>div{display:none}.brand-mark{width:30px}.nav-label{display:none}.nav{margin-top:45px}.nav button{justify-content:center;padding:14px}.nav button span{display:none}.sidebar-bottom{display:none}.main{margin-left:78px;width:calc(100% - 78px)}.overview-grid{grid-template-columns:minmax(0,1fr)}.brief{display:none}.kpi-grid{gap:10px}.kpi-value{font-size:22px}.kpi-label{font-size:10px}.source-text{max-width:275px}.lower-grid{grid-template-columns:repeat(3,1fr)}.sandbox-grid{grid-template-columns:240px 1fr}.scenario-kpis{grid-template-columns:1fr}.scenario-kpis .kpi{min-height:80px;display:flex;align-items:center;justify-content:space-between}.scenario-kpis .kpi-foot{display:none}.runbook{grid-template-columns:1fr 1fr}.heading-row h1{font-size:27px}.two-col{grid-template-columns:1fr}.evidence-top{grid-template-columns:repeat(2,1fr)}}
@media(max-width:650px){.layout{display:block}.sidebar{position:sticky;inset:0;top:0;width:100%;height:62px;padding:0 15px;display:flex;flex-direction:row;align-items:center;justify-content:space-between}.brand{padding:0 3px}.brand-mark{width:26px;height:28px}.nav{flex-direction:row;margin:0;gap:0}.nav button{padding:12px 13px}.nav button svg{width:17px;height:17px}.main{width:100%;margin-left:0}.topbar{height:52px;padding:0 19px}.top-right{gap:8px}.avatar{display:none}.status{font-size:9px}.content{padding:24px 17px 17px}.heading-row{align-items:flex-start;margin-bottom:20px}.heading-row h1{font-size:24px;letter-spacing:-.7px}.heading-row .btn{padding:9px;font-size:0}.heading-row .btn svg{width:16px;height:16px}.subtitle{font-size:11px}.eyebrow{font-size:9px}.source-bar{flex-direction:column;align-items:stretch;gap:9px;padding:12px}.source-bar select{border-left:0;padding:4px 0;max-width:100%;border-top:1px solid #d5e0d9;padding-top:8px}.source-text{max-width:100%;font-size:9px}.kpi-grid{grid-template-columns:1fr 1fr;gap:11px}.kpi{padding:15px;min-height:117px}.kpi-value{font-size:26px}.kpi-foot{font-size:9px}.card-head{flex-direction:column;gap:9px}.card-head .seg{align-self:flex-start}.overview-grid{margin-bottom:14px}.chart-card{padding:18px 14px 13px}.chart-wrap{min-height:160px}.lower-grid{grid-template-columns:1fr;gap:13px}.mini-card{padding:18px}.mini-card .mini-value{font-size:22px}.mini-description{font-size:11px}.sandbox-grid{grid-template-columns:1fr}.scenario-kpis{grid-template-columns:repeat(3,1fr);gap:8px}.scenario-kpis .kpi{display:block;min-height:120px;padding:12px}.scenario-kpis .kpi-value{font-size:17px}.scenario-kpis .kpi-label{font-size:9px}.scenario-kpis .kpi-foot{display:block;font-size:8px}.runbook{grid-template-columns:1fr}.two-col{margin-top:15px;gap:15px}.evidence-top{gap:10px}.evidence-top .kpi{padding:13px}.evidence-top .kpi-value{font-size:22px}.table-card{padding:17px}.table-card th,.table-card td{padding:10px 9px;font-size:10px;white-space:nowrap}.legend{font-size:9px;gap:12px}.footer{font-size:8px}.footer span:last-child{display:none}.notice{font-size:10px}.page .full-chart,.small-chart{padding:18px 14px}.page-heading h1{font-size:25px}.clock{font-size:9px}.error{left:17px;right:17px;font-size:11px}.source-left{gap:8px}}
</style>
</head>
<body>
<div class="layout">
<aside class="sidebar" aria-label="Main navigation">
 <a class="brand" href="#overview" aria-label="Volt Energy Lab home"><svg class="brand-mark" viewBox="0 0 40 44" fill="none"><path d="M5 5h11v19H5z" fill="#91dabb"/><path d="M16 14h11v24H16z" fill="#50b99f"/><path d="M27 2h8v28h-8z" fill="#d3eee1"/><path d="M5 29h11v9H5z" fill="#518c7c"/></svg><div><span class="brand-name">volt<span style="color:#70cbb0">.</span></span><span class="brand-label">ENERGY LAB</span></div></a>
 <div class="nav-label">WORKSPACE</div>
 <nav class="nav">
  <button class="active" data-page="overview" aria-label="Overview"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg><span>Overview</span></button>
  <button data-page="forecast" aria-label="Forecast explorer"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 4v16h18M6 15l4-5 4 3 6-8"/><path d="M16 5h4v4"/></svg><span>Forecast explorer</span></button>
  <button data-page="sandbox" aria-label="Decision sandbox"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M9 3h6M10 3v6l-6 10c-.5 1 .3 2 1.5 2h13c1.2 0 2-1 1.5-2L14 9V3M7 15h10"/></svg><span>Decision sandbox</span></button>
  <button data-page="evidence" aria-label="Model evidence"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M5 4h14v17H5zM8 8h8M8 12h8M8 16h3"/><path d="m14 16 1 1 3-3"/></svg><span>Model evidence</span></button>
  <button data-page="runbook" aria-label="Project runbook"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M8 6 3 12l5 6M16 6l5 6-5 6M14 3l-4 18"/></svg><span>Project runbook</span></button>
 </nav>
 <div class="sidebar-bottom"><div class="lab-badge"><span>Dataset-driven simulation</span><p>Real forecasting code.<br>Transparent assumptions.<br>No live grid connection.</p></div><div class="sidebar-sign">STUDENT PROOF OF WORK / v1.0</div></div>
</aside>
<main class="main">
 <header class="topbar"><div class="crumb">Workspace <span style="color:#b7c3bc;margin-left:10px">/</span><strong id="crumb">Overview</strong></div><div class="top-right"><div class="status"><i class="dot"></i><span id="engine-status">Connecting to model engine</span></div><div class="avatar">EF</div></div></header>
 <div class="content" id="workspace">
  <div class="heading-row"><div><div class="eyebrow"><i class="dot"></i> AI-POWERED ENERGY FORECASTING</div><h1>Plan demand. Reduce surprises.</h1><p class="subtitle">From historical meter readings to a clearer view of tomorrow. A practical forecasting lab for smarter energy decisions.</p></div><a class="btn primary artifact" data-file="forecast_168h.csv" href="#"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3v12m-4-4 4 4 4-4M4 16v5h16v-5"/></svg>Export forecast</a></div>
  <div class="source-bar"><div class="source-left"><span class="pill" id="source-pill">LOADING</span><span class="source-text" id="source-info">Loading dataset provenance…</span></div><label><span class="hidden">Select dataset</span><select id="dataset" aria-label="Select forecast dataset"><option>Loading datasets…</option></select></label></div>

  <section class="page active" id="page-overview">
   <div class="kpi-grid">
    <div class="card kpi"><div class="kpi-top"><span class="kpi-label">Forecast consumption</span><span class="kpi-icon">↗</span></div><div class="kpi-value" id="kwh">—</div><div class="kpi-foot"><strong id="horizon-label">Next 24 hours</strong><span>· forecast, not actual</span></div></div>
    <div class="card kpi"><div class="kpi-top"><span class="kpi-label">Predicted peak demand</span><span class="kpi-icon">ϟ</span></div><div class="kpi-value" id="peak">—</div><div class="kpi-foot">Expected at <strong id="peak-at">—</strong></div></div>
    <div class="card kpi"><div class="kpi-top"><span class="kpi-label">Day-ahead test error</span><span class="kpi-icon">≈</span></div><div class="kpi-value" id="rmse">—</div><div class="kpi-foot"><strong>RMSE</strong><span>· 24-hour rolling backtest</span></div></div>
    <div class="card kpi"><div class="kpi-top"><span class="kpi-label">Hypothetical bill saving</span><span class="kpi-icon">₹</span></div><div class="kpi-value" id="saving">—</div><div class="kpi-foot"><strong id="saving-pct">—</strong><span>· first 24 h scenario only</span></div></div>
   </div>
   <div class="overview-grid">
    <div class="card chart-card"><div class="card-head"><div><h2>Demand outlook</h2><div class="card-sub">48 hours of meter context. A recursive forecast from the last known reading.</div></div><div class="seg horizon-buttons"><button class="active" data-hours="24">24 h</button><button data-hours="48">48 h</button><button data-hours="168">7 days</button></div></div><div class="legend"><span><i class="swatch"></i>Historical power</span><span><i class="swatch teal"></i>Forecast</span><span><i class="swatch band"></i>Empirical error band · first 24 h</span></div><div class="chart-wrap" id="overview-chart"></div><div class="chart-note">DATASET CLOCK · kW = average power · no future actuals · >24 h is exploratory</div></div>
    <div class="card brief"><div class="brief-top">✧ Planning brief</div><div class="brief-time" id="brief-time">—</div><div class="brief-info">forecast peak / <span id="brief-date">—</span></div><div class="brief-line"></div><div class="inline-label">Selected on validation</div><div class="brief-model" id="brief-model">—</div><p>Explore shifting flexible peak-hour work into lower-cost hours. Decisions remain a simulation, not automatic equipment control.</p><button class="btn" data-goto="sandbox">Open decision sandbox <span>↗</span></button></div>
   </div>
   <div class="lower-grid">
    <div class="card mini-card"><div class="mini-icon">01 / DATA INTEGRITY</div><h3>Observed meter readings</h3><div class="mini-value" id="observed-count">—</div><div class="progress"><div id="quality-progress" style="width:0"></div></div><div class="mini-description" id="quality-note">—</div></div>
    <div class="card mini-card"><div class="mini-icon">02 / FORECAST FEATURES</div><h3>History + human routines</h3><div class="feature-chips"><span>1 h lag</span><span>24 h lag</span><span>168 h lag</span><span>Cyclic calendar</span><span>Rolling demand</span></div><div class="mini-description">15 past-only features. No future temperature or occupancy measurements are assumed known.</div></div>
    <div class="card mini-card"><div class="mini-icon">03 / VERIFIED EVALUATION</div><h3>Time moves forward</h3><div class="mini-value" style="font-size:19px">70 / 15 / 15</div><div class="mini-description">Chronological train / validation / test.<br>Three baselines + three ML candidates.</div><div style="margin-top:13px"><button class="btn" data-goto="evidence" style="padding:7px 10px;font-size:9px">Inspect model evidence ↗</button></div></div>
   </div>
  </section>

  <section class="page" id="page-forecast">
   <div class="page-heading"><div class="eyebrow">EXPLORE / FUTURE DEMAND</div><h1>The next hours, made visible.</h1><p class="subtitle">Forecasts start after the dataset's last observation, not after today's date. Longer horizons recursively reuse predictions.</p></div>
   <div class="card full-chart"><div class="card-head"><div><h2>Forecast explorer</h2><div class="card-sub" id="forecast-window">—</div></div><div class="seg horizon-buttons"><button class="active" data-hours="24">24 h</button><button data-hours="48">48 h</button><button data-hours="168">7 days</button></div></div><div class="legend"><span><i class="swatch teal"></i>Forecast power</span><span><i class="swatch band"></i>Validation error band · first 24 h only</span></div><div class="chart-wrap" id="forecast-chart"></div></div>
   <div class="notice">The shaded band is the pooled 90th percentile of absolute errors from validation day-ahead forecasts. It is <strong>not a guaranteed 90% interval</strong>. No band is shown beyond 24 hours.</div>
   <div class="card table-card"><div class="table-head"><h2>Hourly forecast</h2><a class="small-link artifact" data-file="forecast_168h.csv" href="#">Download full 168-hour CSV ↗</a></div><table><thead><tr><th>Dataset timestamp</th><th>Lead time</th><th>Power (kW)</th><th>Energy (kWh)</th><th>Lower band</th><th>Upper band</th></tr></thead><tbody id="forecast-table"></tbody></table><div class="sources-note" id="table-count">Showing first 24 hours.</div></div>
  </section>

  <section class="page" id="page-sandbox">
   <div class="page-heading"><div class="eyebrow">SIMULATE / PLANNING DECISIONS</div><h1>A forecast is useful when it informs a decision.</h1><p class="subtitle">Move a configurable share of flexible peak-hour demand into off-peak headroom. This sandbox always uses the first 24 forecast hours.</p></div>
   <div class="sandbox-grid">
    <div class="card controls-card"><h2>Scenario settings</h2><div class="card-sub">Illustrative rates · INR per kWh</div><div class="field"><label for="flex">Flexible peak load <strong id="flex-label">10%</strong></label><input id="flex" type="range" min="0" max="50" step="1" value="10"><div class="field-note">Only peak-hour energy is eligible for shifting.</div></div><div class="field"><label for="off-rate">Off-peak / 00:00–05:59</label><input id="off-rate" type="number" value="5" min="0" max="1000" step="0.5"></div><div class="field"><label for="std-rate">Standard / all other hours</label><input id="std-rate" type="number" value="8" min="0" max="1000" step="0.5"></div><div class="field"><label for="peak-rate">Peak / 17:00–21:59</label><input id="peak-rate" type="number" value="12" min="0" max="1000" step="0.5"></div><button class="btn primary" id="apply-scenario" style="width:100%">Run virtual scenario ↗</button><div class="field-note" style="margin-top:13px">No comfort, equipment, demand-charge or rebound constraints. Jobs may be pre-scheduled or deferred anywhere in the planning window.</div></div>
    <div><div class="scenario-kpis"><div class="card kpi"><div class="kpi-label">Baseline cost</div><div class="kpi-value" id="base-cost">—</div><div class="kpi-foot">Forecast-based / 24 h</div></div><div class="card kpi"><div class="kpi-label">Scenario cost</div><div class="kpi-value" id="shift-cost">—</div><div class="kpi-foot">After flexible shifting</div></div><div class="card kpi"><div class="kpi-label">Hypothetical saving</div><div class="kpi-value" id="scenario-saving" style="color:var(--teal)">—</div><div class="kpi-foot" id="scenario-pct">—</div></div></div><div class="card chart-card"><div class="card-head"><div><h2>Load shifting, not load deletion</h2><div class="card-sub" id="scenario-window">—</div></div></div><div class="legend"><span><i class="swatch"></i>Baseline forecast</span><span><i class="swatch teal"></i>Shifted scenario</span></div><div class="chart-wrap" id="scenario-chart"></div></div><div class="notice info" id="energy-conserved">Total energy is conserved. With constant carbon intensity, modeled emissions do not change.</div></div>
   </div>
  </section>

  <section class="page" id="page-evidence">
   <div class="page-heading"><div class="eyebrow">VERIFY / MEASURED RESULTS</div><h1>Evidence over impressive claims.</h1><p class="subtitle">Validation chooses the champion. Test reports its performance. One-step replay and fixed-origin day-ahead forecasting are deliberately separated.</p></div>
   <div class="evidence-top"><div class="card kpi"><div class="kpi-label">Day-ahead RMSE</div><div class="kpi-value" id="ev-rmse">—</div><div class="kpi-foot">Lower is better</div></div><div class="card kpi"><div class="kpi-label">Day-ahead MAE</div><div class="kpi-value" id="ev-mae">—</div><div class="kpi-foot">Mean absolute error</div></div><div class="card kpi"><div class="kpi-label">Day-ahead R²</div><div class="kpi-value" id="ev-r2">—</div><div class="kpi-foot">Not percentage accuracy</div></div><div class="card kpi"><div class="kpi-label">Vs last-week baseline</div><div class="kpi-value" id="ev-improvement">—</div><div class="kpi-foot">Day-ahead RMSE reduction</div></div></div>
   <div class="card table-card"><div class="table-head"><h2>Candidate comparison</h2><a class="small-link artifact" data-file="model_comparison.csv" href="#">Export evidence CSV ↗</a></div><table><thead><tr><th>Candidate</th><th>Validation 24 h RMSE ↓</th><th>Test 24 h RMSE ↓</th><th>Test one-step RMSE ↓</th><th>Test 24 h R²</th></tr></thead><tbody id="comparison-table"></tbody></table><div class="sources-note">All errors in kW. The lowest test error may belong to a different model; the champion is never changed using test results.</div></div>
   <div class="two-col"><div class="card small-chart"><h2>One-step test replay</h2><div class="card-sub">Actual lags update each hour · last 168 eligible test rows</div><div class="legend"><span><i class="swatch"></i>Observed</span><span><i class="swatch teal"></i>Predicted</span></div><div class="chart-wrap" id="one-step-chart"></div></div><div class="card small-chart"><h2>A day-ahead test example</h2><div class="card-sub">Last sampled origin · no actual updates inside the 24-hour window</div><div class="legend"><span><i class="swatch"></i>Observed</span><span><i class="swatch teal"></i>Forecast</span></div><div class="chart-wrap" id="day-ahead-chart"></div></div></div>
   <div class="two-col"><div class="card small-chart"><h2>Error by hours ahead</h2><div class="card-sub">Same held-out origins · each lead time is a separate error estimate</div><div class="chart-wrap" id="horizon-chart"></div></div><div class="card mini-card"><h2>Evaluation audit</h2><div id="audit-list" class="quality-list"></div><div class="sources-note">Reported scores are from train+validation refits on untouched test data. The served model is subsequently refit on all historical data for future forecasting.</div></div></div>
  </section>

  <section class="page" id="page-runbook">
   <div class="page-heading"><div class="eyebrow">BUILD / REPRODUCIBLE PROOF</div><h1>From this lab to your GitHub.</h1><p class="subtitle">The repository contains the complete code, dataset attribution, executed notebook, automated tests and an A–M beginner's guide.</p></div>
   <div class="runbook"><div class="card controls-card"><h2>Run it yourself</h2><div class="card-sub">Python 3.11–3.13 · CPU only · no corporate access</div><div class="codebox"><span class="comment"># Activate your virtual environment first</span>
python -m pip install -r requirements.txt

<span class="comment"># Offline virtual-building experiment</span>
python main.py --dataset synthetic

<span class="comment"># Real public household benchmark</span>
python main.py --dataset uci

<span class="comment"># Start the interactive dashboard + API</span>
python app.py

<span class="comment"># Verify code and leakage checks</span>
python -m pip install -r requirements-dev.txt
python -m pytest -q</div><ol class="steps"><li>Read START_HERE.md, then docs/PROJECT_GUIDE.md.</li><li>Inspect data source and measurement units.</li><li>Reproduce chronological evaluation and inspect baselines.</li><li>Capture real outputs; explain limitations honestly.</li><li>Publish with the Git commands in the guide.</li></ol></div><div class="card controls-card"><h2>Download evidence</h2><div class="card-sub">Machine-readable artifacts from this dataset run</div><div class="artifact-grid"><a class="artifact" data-file="metrics.json" href="#">Metrics JSON ↗</a><a class="artifact" data-file="data_quality.json" href="#">Data-quality audit ↗</a><a class="artifact" data-file="forecast_168h.csv" href="#">168 h forecast CSV ↗</a><a class="artifact" data-file="backtest_predictions.csv" href="#">Day-ahead backtest ↗</a><a class="artifact" data-file="model_comparison.csv" href="#">Model comparison ↗</a><a class="artifact" data-file="simulation.json" href="#">Scenario assumptions ↗</a><a class="artifact" data-file="simulation.csv" href="#">Scenario hourly CSV ↗</a><a class="artifact" data-file="run_manifest.json" href="#">Run manifest ↗</a></div><div class="notice info">A portfolio experiment, not a production grid controller. Do not claim blackout prevention, realized savings, salary premiums or corporate deployment from this work.</div><h3 style="margin-top:21px">Data provenance</h3><div class="sources-note" id="provenance">—</div><div class="sources-note">See docs/DATA_CARD.md, docs/MODEL_CARD.md and docs/CLAIMS_AUDIT.md for assumptions, external sources and honest reporting guidance.</div></div></div>
  </section>
  <footer class="footer"><span><strong>VOLT / ENERGY LAB</strong> &nbsp; · &nbsp; Python · scikit-learn · Flask</span><span>Historical dataset replay. Transparent simulation. No live grid connection.</span></footer>
 </div>
</main>
</div>
<div class="error" id="error-box" role="alert"></div>
<script>
'use strict';
const SNAPSHOT = {{ snapshot | default(none) | tojson }};
const state={dataset:'synthetic',hours:24,page:'overview',summary:null,forecast:null,history:null,evaluation:null,scenario:null,requestId:0};
const $=id=>document.getElementById(id);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(x,d=1)=>Number(x).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const date=t=>new Date(String(t).replace(' ','T'));
const clock=t=>date(t).toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit'});
const shortDate=t=>date(t).toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'});
const money=x=>'₹'+num(x,2);
function error(message){$('error-box').textContent=message;$('error-box').style.display='block';clearTimeout(error.timer);error.timer=setTimeout(()=>$('error-box').style.display='none',6500)}
function offlineScenario(run,overrides){
 const source=SNAPSHOT.datasets[run], rows=source.forecast.predictions.slice(0,24), old=source.summary.simulation;
 const s={...old.settings,...overrides};
 if(!(s.offpeak_rate>=0&&s.offpeak_rate<=s.standard_rate&&s.standard_rate<=s.peak_rate&&s.peak_rate<=1000))throw Error('Rates must satisfy 0 ≤ off-peak ≤ standard ≤ peak ≤ 1000.');
 if(!(s.flexible_fraction>=0&&s.flexible_fraction<=.5))throw Error('Flexible share must be 0–50%.');
 const base=rows.map(r=>r.predicted_kw),max=Math.max(...base),peak=rows.map(r=>s.peak_hours.includes(date(r.Datetime).getHours())),off=rows.map(r=>s.offpeak_hours.includes(date(r.Datetime).getHours()));
 const requested=base.reduce((a,v,i)=>a+(peak[i]?v*s.flexible_fraction:0),0),head=base.map((v,i)=>off[i]?Math.max(0,max-v):0),capacity=head.reduce((a,b)=>a+b,0),moved=Math.min(requested,capacity);
 const shifted=base.map((v,i)=>v-(peak[i]&&requested?v*s.flexible_fraction*moved/requested:0)+(off[i]&&capacity?moved*head[i]/capacity:0));
 const rates=base.map((v,i)=>peak[i]?s.peak_rate:off[i]?s.offpeak_rate:s.standard_rate),before=base.reduce((a,v,i)=>a+v*rates[i],0),after=shifted.reduce((a,v,i)=>a+v*rates[i],0),energy=base.reduce((a,b)=>a+b,0);
 const result={...old,settings:s,baseline_energy_kwh:energy,shifted_energy_kwh:shifted.reduce((a,b)=>a+b,0),requested_shift_kwh:requested,moved_energy_kwh:moved,baseline_peak_kw:max,shifted_peak_kw:Math.max(...shifted),baseline_cost_inr:before,shifted_cost_inr:after,hypothetical_savings_inr:before-after,hypothetical_savings_pct:before?(before-after)/before*100:0};
 return {dataset:run,result,hourly:rows.map((r,i)=>({...r,baseline_kw:base[i],shifted_kw:shifted[i],rate_inr_per_kwh:rates[i],baseline_cost_inr:base[i]*rates[i],shifted_cost_inr:shifted[i]*rates[i]}))};
}
async function api(url,body){
 if(SNAPSHOT){
  const u=new URL(url,'https://snapshot.invalid'),run=body?.dataset||u.searchParams.get('dataset')||state.dataset;
  if(u.pathname==='/api/runs')return SNAPSHOT.runs;
  const d=SNAPSHOT.datasets[run];if(!d)throw Error('Unknown dataset.');
  if(u.pathname==='/api/summary')return d.summary;
  if(u.pathname==='/api/history')return d.history;
  if(u.pathname==='/api/evaluation')return d.evaluation;
  if(u.pathname==='/api/simulate')return offlineScenario(run,body.settings||{});
  if(u.pathname==='/api/forecast'){
   const h=Number(u.searchParams.get('hours')||24),pred=d.forecast.predictions.slice(0,h),peak=pred.reduce((a,b)=>a.predicted_kw>=b.predicted_kw?a:b);
   return {...d.forecast,hours:h,predictions:pred,total_kwh:pred.reduce((a,b)=>a+b.predicted_kwh,0),peak_kw:peak.predicted_kw,peak_time:peak.Datetime};
  }
  throw Error('Artifact not available in snapshot.');
 }
 const res=await fetch(url,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});
 const value=await res.json();if(!res.ok)throw Error(value.message||'Request failed');return value;
}
function chart(id,series,options={}){
 const wrap=$(id);if(!wrap)return;
 const times=options.times||series[0].values.map((v,i)=>i),N=times.length,W=820,H=options.small?248:300,L=49,R=19,T=28,B=40,iw=W-L-R,ih=H-T-B;
 let all=series.flatMap(s=>s.values).filter(v=>v!=null&&Number.isFinite(Number(v)));
 if(options.band)all=all.concat(options.band.lower.filter(v=>v!=null),options.band.upper.filter(v=>v!=null));
 if(!all.length){wrap.innerHTML='<div class="empty">No observed values available.</div>';return;}
 const max=Math.max(...all)*1.10||1,min=0,x=i=>L+(i/Math.max(1,N-1))*iw,y=v=>T+ih-(v-min)/(max-min)*ih;
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(options.label||'Energy chart')}"><defs><linearGradient id="fill-${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#098974" stop-opacity=".10"/><stop offset="100%" stop-color="#098974" stop-opacity="0"/></linearGradient></defs><text x="${L}" y="13" font-size="9" fill="#94a298" font-family="inherit">${esc(options.unit||'POWER / kW')}</text>`;
 for(let j=0;j<5;j++){const v=max*j/4,yy=y(v);svg+=`<line x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}" stroke="#eaf0ec" stroke-dasharray="3 4"/><text x="${L-10}" y="${yy+3}" font-size="9" fill="#8fa196" text-anchor="end" font-family="inherit">${num(v,max<10?1:0)}</text>`;}
 if(options.band){const points=options.band.upper.map((v,i)=>v==null?null:[x(i),y(v)]).filter(Boolean).concat(options.band.lower.map((v,i)=>v==null?null:[x(i),y(v)]).filter(Boolean).reverse());if(points.length)svg+=`<polygon points="${points.map(p=>p.join(',')).join(' ')}" fill="#e2f2e8" opacity=".8"/>`;}
 series.forEach((s,si)=>{
  let d='',valid=[];s.values.forEach((v,i)=>{if(v==null||!Number.isFinite(Number(v))){valid.push(null);return;}valid.push([x(i),y(v)]);});
  let previous=false;valid.forEach(p=>{if(!p){previous=false;return;}d+=(previous?'L':'M')+p[0].toFixed(2)+','+p[1].toFixed(2)+' ';previous=true;});
  if(s.fill&&valid.filter(Boolean).length){const ps=valid.filter(Boolean);svg+=`<path d="M${ps[0][0]},${T+ih} ${ps.map(p=>'L'+p.join(',')).join(' ')} L${ps[ps.length-1][0]},${T+ih}Z" fill="url(#fill-${id})"/>`;}
  svg+=`<path d="${d}" stroke="${s.color||'#098974'}" stroke-width="${s.width||1.8}" stroke-linejoin="round" stroke-linecap="round" fill="none" ${s.dash?'stroke-dasharray="5 4"':''}/>`;
 });
 if(options.cutoff!=null){const xx=x(options.cutoff);svg+=`<line x1="${xx}" y1="${T}" x2="${xx}" y2="${T+ih}" stroke="#c4b78b" stroke-dasharray="4 4"/><rect x="${Math.min(xx-36,W-89)}" y="${T+3}" width="76" height="17" rx="4" fill="#f5f1e6"/><text x="${Math.min(xx+2,W-51)}" y="${T+14}" text-anchor="middle" fill="#998357" font-size="8" font-family="inherit">FORECAST START</text>`;}
 const ticks=options.small?4:5;for(let j=0;j<ticks;j++){let i=Math.round(j*(N-1)/(ticks-1)),label=options.numeric?`${times[i]} h`:N>96?shortDate(times[i]).slice(0,6):clock(times[i]);svg+=`<text x="${x(i)}" y="${H-15}" text-anchor="${j===0?'start':j===ticks-1?'end':'middle'}" fill="#8c9f91" font-size="9" font-family="inherit">${esc(label)}</text>`;}
 svg+='</svg><div class="chart-tooltip"></div>';wrap.innerHTML=svg;
 const svgEl=wrap.querySelector('svg'),tip=wrap.querySelector('.chart-tooltip');
 svgEl.addEventListener('pointermove',e=>{const box=svgEl.getBoundingClientRect(),xx=(e.clientX-box.left)/box.width*W,i=Math.max(0,Math.min(N-1,Math.round((xx-L)/iw*(N-1))));const vals=series.filter(s=>s.values[i]!=null).map(s=>`${esc(s.name)}: ${num(s.values[i],2)} ${options.numeric?'kW':'kW'}`);if(!vals.length)return;tip.innerHTML=`<b>${esc(options.numeric?times[i]+' hours ahead':shortDate(times[i])+' · '+clock(times[i]))}</b><br>${vals.join('<br>')}`;tip.style.display='block';tip.style.left=Math.min(e.clientX-box.left+8,Math.max(0,box.width-210))+'px';tip.style.top=Math.max(12,e.clientY-box.top-80)+'px';});
 svgEl.addEventListener('pointerleave',()=>tip.style.display='none');
}
function navigate(page){
 if(!['overview','forecast','sandbox','evidence','runbook'].includes(page))page='overview';
 state.page=page;document.querySelectorAll('.page').forEach(e=>e.classList.toggle('active',e.id==='page-'+page));document.querySelectorAll('.nav button').forEach(e=>e.classList.toggle('active',e.dataset.page===page));
 $('crumb').textContent={overview:'Overview',forecast:'Forecast explorer',sandbox:'Decision sandbox',evidence:'Model evidence',runbook:'Project runbook'}[page];
 // Sandboxed file viewers can have an opaque origin and disallow History API updates.
 try{if(location.hash!=='#'+page)history.replaceState(null,'','#'+page);}catch(_){/* Navigation still works without URL changes. */}
}
function updateArtifacts(){document.querySelectorAll('.artifact').forEach(a=>{a.href='/download/'+encodeURIComponent(state.dataset)+'/'+a.dataset.file;});}
function renderSummary(){
 const s=state.summary,m=s.metrics,q=s.quality,f=state.forecast,sim=state.scenario?.result||s.simulation;
 $('source-pill').textContent=m.source.source_type==='synthetic'?'SIMULATED BUILDING':'PUBLIC MEASURED DATA';
 $('source-info').textContent=`${num(q.hourly_rows,0)} hourly readings · data through ${shortDate(q.end)} · historical replay`;
 $('kwh').innerHTML=num(f.total_kwh,0)+'<small>kWh</small>';$('peak').innerHTML=num(f.peak_kw,2)+'<small>kW</small>';$('rmse').innerHTML=num(m.test_day_ahead.rmse_kw,3)+'<small>kW</small>';
 $('saving').innerHTML='₹'+num(sim.hypothetical_savings_inr,2);$('saving-pct').textContent=num(sim.hypothetical_savings_pct,1)+'% lower modeled cost';$('horizon-label').textContent='Next '+state.hours+' hours';$('peak-at').textContent=clock(f.peak_time)+' · '+shortDate(f.peak_time).slice(0,6);
 $('brief-time').textContent=clock(f.peak_time);$('brief-date').textContent=shortDate(f.peak_time);$('brief-model').textContent=m.selected_model_label;
 $('observed-count').textContent=num(q.observed_target_hours,0)+' / '+num(q.hourly_rows,0);$('quality-progress').style.width=(q.observed_target_hours/q.hourly_rows*100)+'%';$('quality-note').textContent=`${q.imputed_target_hours} hours causally filled · ${q.remaining_missing_target_hours} long-gap hours excluded. Imputed targets are never test truth.`;
 const hist=state.history.data.slice(-48),future=f.predictions,times=hist.map(r=>r.Datetime).concat(future.map(r=>r.Datetime)),n=hist.length;
 chart('overview-chart',[{name:'Historical',values:hist.map(r=>r.Energy_kW).concat(future.map(()=>null)),color:'#234334'},{name:'Forecast',values:hist.map(()=>null).concat(future.map(r=>r.predicted_kw)),color:'#098974',fill:true}],{times,cutoff:n,band:{lower:hist.map(()=>null).concat(future.map(r=>r.lower_kw)),upper:hist.map(()=>null).concat(future.map(r=>r.upper_kw))},label:'Historical power and recursive future forecast'});
 chart('forecast-chart',[{name:'Forecast',values:future.map(r=>r.predicted_kw),color:'#098974',fill:true}],{times:future.map(r=>r.Datetime),band:{lower:future.map(r=>r.lower_kw),upper:future.map(r=>r.upper_kw)},label:'Future hourly energy demand'});
 $('forecast-window').textContent=`${shortDate(future[0].Datetime)} ${clock(future[0].Datetime)} → ${shortDate(future[future.length-1].Datetime)} ${clock(future[future.length-1].Datetime)} · issued after ${s.forecast.cutoff}`;
 const visible=future.slice(0,48);$('forecast-table').innerHTML=visible.map(r=>`<tr><td>${esc(r.Datetime)}</td><td>+${r.horizon} h</td><td><strong>${num(r.predicted_kw,3)}</strong></td><td>${num(r.predicted_kwh,3)}</td><td>${r.lower_kw==null?'—':num(r.lower_kw,3)}</td><td>${r.upper_kw==null?'—':num(r.upper_kw,3)}</td></tr>`).join('');$('table-count').textContent=`Showing ${visible.length} of ${future.length} selected forecast hours; CSV export contains the full 168-hour exploratory forecast.`;
 const src=m.source;$('provenance').innerHTML=src.source_type==='synthetic'?`Seeded virtual mixed-use building · seed 42.<br>${esc(src.warning)}`:`Georges Hebrail & Alice Berard · one household in Sceaux, France.<br>UCI dataset 235 · DOI 10.24432/C58K54 · CC BY 4.0.<br>Minute-averaged kW aggregated hourly with ≥54 valid minutes.<br><a href="https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption" target="_blank" rel="noopener">Official source ↗</a>`;
 document.querySelectorAll('[data-hours]').forEach(b=>b.classList.toggle('active',Number(b.dataset.hours)===state.hours));updateArtifacts();
}
function renderEvidence(){
 const e=state.evaluation,m=state.summary.metrics;
 $('ev-rmse').innerHTML=num(m.test_day_ahead.rmse_kw,3)+'<small>kW</small>';$('ev-mae').innerHTML=num(m.test_day_ahead.mae_kw,3)+'<small>kW</small>';$('ev-r2').textContent=num(m.test_day_ahead.r2,3);$('ev-improvement').textContent=num(m.day_ahead_rmse_improvement_vs_weekly_baseline_pct,1)+'%';
 $('comparison-table').innerHTML=e.comparison.map(r=>`<tr class="${r.selected?'selected':''}"><td><strong>${esc(r.label)}</strong>${r.selected?'<span class="selected-badge">VALIDATION WINNER</span>':''}</td><td>${num(r.validation_day_ahead_rmse_kw,3)}</td><td>${num(r.test_day_ahead_rmse_kw,3)}</td><td>${num(r.test_one_step_rmse_kw,3)}</td><td>${num(r.test_day_ahead_r2,3)}</td></tr>`).join('');
 ['one_step','day_ahead_example'].forEach((kind,i)=>{const rows=e[kind];chart(i?'day-ahead-chart':'one-step-chart',[{name:'Observed',values:rows.map(r=>r.actual_kw),color:'#234334'},{name:'Forecast',values:rows.map(r=>r.predicted_kw),color:'#098974'}],{times:rows.map(r=>r.Datetime),small:true,label:i?'24-hour day-ahead backtest example':'One-step held-out test predictions'});});
 chart('horizon-chart',[{name:'RMSE',values:e.horizon_metrics.map(r=>r.rmse_kw),color:'#668bb7',fill:true}],{times:e.horizon_metrics.map(r=>r.horizon),numeric:true,unit:'ERROR / kW',small:true,label:'Forecast RMSE by lead time'});
 const rows=[['Champion',m.selected_model_label],['Observed one-step targets',num(m.test_one_step.n,0)],['Sampled test origins',m.test_backtest_origins+' × 24 h'],['Scored day-ahead targets',num(m.test_day_ahead.n,0)],['Empirical test band coverage',num(m.empirical_test_band_coverage_pct,1)+'%'],['Selection', 'Validation only']];$('audit-list').innerHTML=rows.map(([k,v])=>`<div class="metric-row"><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`).join('');
}
function renderScenario(){
 const d=state.scenario,r=d.result;
 $('base-cost').textContent=money(r.baseline_cost_inr);$('shift-cost').textContent=money(r.shifted_cost_inr);$('scenario-saving').textContent=money(r.hypothetical_savings_inr);$('scenario-pct').textContent=num(r.hypothetical_savings_pct,2)+'% lower hypothetical cost';
 $('scenario-window').textContent=`${shortDate(r.window_start)} ${clock(r.window_start)} → ${shortDate(r.window_end)} ${clock(r.window_end)}`;
 chart('scenario-chart',[{name:'Baseline',values:d.hourly.map(x=>x.baseline_kw),color:'#234334'},{name:'Shifted',values:d.hourly.map(x=>x.shifted_kw),color:'#098974',fill:true}],{times:d.hourly.map(x=>x.Datetime),label:'Energy-conserving hypothetical load-shifting scenario'});
 $('energy-conserved').innerHTML=`<strong>Energy conserved:</strong> ${num(r.baseline_energy_kwh,2)} kWh before and after · ${num(r.moved_energy_kwh,2)} kWh moved.<br>With a constant illustrative carbon factor, modeled CO₂ reduction is <strong>0 kg</strong>. Savings are hypothetical, not achieved.`;
 if(state.summary&&state.forecast){$('saving').textContent=money(r.hypothetical_savings_inr);$('saving-pct').textContent=num(r.hypothetical_savings_pct,1)+'% lower modeled cost';}
}
async function runScenario(){
 const button=$('apply-scenario');button.disabled=true;
 try{const settings={flexible_fraction:Number($('flex').value)/100,offpeak_rate:Number($('off-rate').value),standard_rate:Number($('std-rate').value),peak_rate:Number($('peak-rate').value)};state.scenario=await api('/api/simulate',{dataset:state.dataset,settings});renderScenario();}catch(e){error(e.message)}finally{button.disabled=false}
}
async function loadRun(){
 const id=++state.requestId;const run=state.dataset;state.scenario=null;$('workspace').classList.add('loading');
 try{const [summary,hist,forecast,evaluation,scenario]=await Promise.all([api('/api/summary?dataset='+run),api('/api/history?dataset='+run+'&hours=168'),api('/api/forecast?dataset='+run+'&hours='+state.hours),api('/api/evaluation?dataset='+run),api('/api/simulate',{dataset:run,settings:{}})]);if(id!==state.requestId)return;Object.assign(state,{summary,history:hist,forecast,evaluation,scenario});$('flex').value=10;$('flex-label').textContent='10%';$('off-rate').value=5;$('std-rate').value=8;$('peak-rate').value=12;renderSummary();renderEvidence();renderScenario();$('engine-status').textContent=SNAPSHOT?'Interactive snapshot · offline ready':'Model API ready · historical replay';}catch(e){error(e.message);$('engine-status').textContent='Run python main.py to prepare data';}finally{if(id===state.requestId)$('workspace').classList.remove('loading')}
}
async function changeHours(hours){
 state.hours=hours;try{state.forecast=await api('/api/forecast?dataset='+state.dataset+'&hours='+hours);renderSummary();}catch(e){error(e.message)}
}
function snapshotDownload(a){
 const d=SNAPSHOT.datasets[state.dataset],file=a.dataset.file;
 let obj=null,type='application/json',text='';
 const map={'metrics.json':d.summary.metrics,'data_quality.json':d.summary.quality,'simulation.json':state.scenario.result,'run_manifest.json':d.manifest};
 if(file in map){obj=map[file];text=JSON.stringify(obj,null,2);}else{
  type='text/csv';const r=file.startsWith('forecast')?d.forecast.predictions:file==='model_comparison.csv'?d.evaluation.comparison:file==='simulation.csv'?state.scenario.hourly:file==='backtest_predictions.csv'?d.backtest:null;
  if(!r){error('This file is included in the project archive.');return;}
  const keys=Object.keys(r[0]);const cell=v=>'"'+String(v??'').replace(/"/g,'""')+'"';text=keys.map(cell).join(',')+'\n'+r.map(row=>keys.map(k=>cell(row[k])).join(',')).join('\n');
 }
 const url=URL.createObjectURL(new Blob([text],{type})),link=document.createElement('a');link.href=url;link.download=state.dataset+'_'+file;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
document.querySelectorAll('.nav button').forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.page)));
document.querySelectorAll('[data-goto]').forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.goto)));
document.querySelectorAll('[data-hours]').forEach(b=>b.addEventListener('click',()=>changeHours(Number(b.dataset.hours))));
$('dataset').addEventListener('change',()=>{state.dataset=$('dataset').value;loadRun()});$('flex').addEventListener('input',()=>{$('flex-label').textContent=$('flex').value+'%'});$('apply-scenario').addEventListener('click',runScenario);
document.querySelectorAll('.artifact').forEach(a=>a.addEventListener('click',e=>{if(SNAPSHOT){e.preventDefault();snapshotDownload(a);}}));
window.addEventListener('hashchange',()=>navigate(location.hash.slice(1)));navigate(location.hash.slice(1)||'overview');
(async()=>{try{const result=await api('/api/runs');if(!result.runs.length)throw Error('No trained dataset. Run python main.py first.');$('dataset').innerHTML=result.runs.map(r=>`<option value="${esc(r.id)}">${esc(r.id==='synthetic'?'Virtual mixed-use building':r.id==='uci'?'UCI · measured household':r.title)}</option>`).join('');state.dataset=result.runs.some(r=>r.id==='synthetic')?'synthetic':result.runs[0].id;$('dataset').value=state.dataset;await loadRun();}catch(e){error(e.message);$('engine-status').textContent='No trained dataset found';}})();
</script>
</body>
</html>
````

## `scripts/browser_check.py`

Executable evidence helper; see its docstring.

````python
"""Optional browser smoke checks and genuine UI screenshots.

Extra tooling (not needed to run the project):
  python -m pip install playwright
  python -m playwright install chromium
Start python app.py in another terminal, then:
  python scripts/browser_check.py --url http://127.0.0.1:5000
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import path, ensure_dir, save_json


def main():
    from playwright.sync_api import sync_playwright
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:5000")
    args = p.parse_args()
    images = ensure_dir(path("images", "screenshots"))
    errors, checks = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1512, "height": 1080}, device_scale_factor=1)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function("document.querySelector('#kwh').textContent !== '—'")
        checks.append({"check": "Overview renders with model forecast", "passed": page.locator('#overview-chart svg').count()==1})
        page.screenshot(path=str(images / "01_overview_synthetic.png"), full_page=True)
        page.click('[data-page="forecast"]')
        page.click('#page-forecast [data-hours="48"]')
        page.wait_for_function("document.querySelector('#forecast-table').children.length === 48")
        checks.append({"check": "48-hour explorer and table update", "passed": True})
        page.click('[data-page="sandbox"]')
        before_saving = page.locator('#scenario-saving').inner_text()
        page.locator('#flex').evaluate("el => { el.value = '20'; }")
        page.locator('#flex').dispatch_event('input')
        page.click('#apply-scenario')
        page.wait_for_function("document.querySelector('#apply-scenario').disabled === false")
        checks.append({"check": "Scenario recalculates after slider change", "passed": 'Energy conserved' in page.locator('#energy-conserved').inner_text() and page.locator('#scenario-saving').inner_text() != before_saving and page.evaluate('state.scenario.result.settings.flexible_fraction') == 0.20})
        page.screenshot(path=str(images / "02_decision_sandbox.png"), full_page=True)
        page.click('[data-page="evidence"]')
        checks.append({"check": "Six candidate rows visible", "passed": page.locator('#comparison-table tr').count()==6})
        page.screenshot(path=str(images / "03_model_evidence.png"), full_page=True)
        page.select_option('#dataset','uci')
        page.wait_for_function("document.querySelector('#source-pill').textContent === 'PUBLIC MEASURED DATA'")
        page.wait_for_function("document.querySelector('#workspace').classList.contains('loading') === false")
        checks.append({"check": "Switch to real UCI benchmark", "passed": 'Random forest' in page.locator('#brief-model').inner_text()})
        page.click('[data-page="overview"]')
        page.click('#page-overview [data-hours="24"]')
        page.wait_for_function("document.querySelector('#horizon-label').textContent === 'Next 24 hours'")
        page.screenshot(path=str(images / "04_overview_uci.png"), full_page=True)
        with page.expect_download() as download:
            page.locator('.heading-row .artifact').click()
        checks.append({"check": "Forecast CSV downloads", "passed": download.value.suggested_filename.endswith('forecast_168h.csv')})
        page.set_viewport_size({"width":390,"height":844})
        page.screenshot(path=str(images / "05_mobile.png"),full_page=True)
        checks.append({"check": "Mobile layout has no horizontal overflow", "passed": page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')})
        snapshot = path('outputs','dashboard.html')
        if snapshot.exists():
            offline = browser.new_page(viewport={"width":1512,"height":1080})
            offline.on('pageerror',lambda e:errors.append(str(e)))
            offline.goto(snapshot.as_uri(),wait_until='load')
            offline.wait_for_function("document.querySelector('#kwh').textContent !== '—'")
            checks.append({"check":"Standalone HTML works without server/network", "passed": 'offline ready' in offline.locator('#engine-status').inner_text()})
            offline.click('[data-page="sandbox"]')
            offline.locator('#flex').evaluate("el => { el.value = '25'; el.dispatchEvent(new Event('input')); }")
            offline.click('#apply-scenario')
            offline.wait_for_function('state.scenario.result.settings.flexible_fraction === 0.25')
            checks.append({"check":"Offline scenario remains interactive","passed": 'Energy conserved' in offline.locator('#energy-conserved').inner_text()})
        browser.close()
    checks.append({"check":"No browser JavaScript exceptions","passed":not errors})
    report={"passed":all(c['passed'] for c in checks),"checks":checks,"browser_errors":errors,
            "note":"Genuine browser smoke checks; screenshots are captures of this app, not generated mockups."}
    save_json(report,path('outputs','browser-checks.json'))
    for c in checks:
        print(('PASS' if c['passed'] else 'FAIL')+' | '+c['check'])
    if errors: print(errors)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    sys.exit(main())
````

## `scripts/export_dashboard.py`

Executable evidence helper; see its docstring.

````python
"""Create a standalone, offline interactive HTML from actual run artifacts.

Run from project root: python scripts/export_dashboard.py
No CDN, fonts, network connection or Python server is needed to view the export.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from flask import render_template
from app import app, available_runs
from src.utils import path, load_json, json_safe


def main():
    snapshot = {"runs": None, "datasets": {}}
    with app.test_client() as client:
        snapshot["runs"] = client.get("/api/runs").get_json()
        for run in available_runs():
            def get(endpoint):
                separator = "&" if "?" in endpoint else "?"
                res = client.get(endpoint + separator + "dataset=" + run)
                if res.status_code != 200:
                    raise RuntimeError(res.get_data(as_text=True))
                return res.get_json()
            snapshot["datasets"][run] = {
                "summary": get("/api/summary"), "history": get("/api/history?hours=168"),
                "forecast": get("/api/forecast?hours=168"), "evaluation": get("/api/evaluation"),
                "manifest": load_json(path("outputs", run, "run_manifest.json")),
                "backtest": pd.read_csv(path("outputs", run, "backtest_predictions.csv")).to_dict("records")}
    with app.app_context():
        html = render_template("dashboard.html", snapshot=json_safe(snapshot))
    target = path("outputs", "dashboard.html")
    target.write_text(html, encoding="utf-8")
    print(f"Exported offline dashboard: {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
````

## `scripts/export_results.py`

Executable evidence helper; see its docstring.

````python
"""Regenerate the human-readable result report from real saved artifacts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.utils import path, load_json


def main():
    text = ["# Actual execution results\n",
            "This report is generated from saved pipeline artifacts, not target or invented scores. "
            "Run `python scripts/export_results.py` after a verified rerun to refresh it. "
            "The supplied runs use seed 42, full settings, CPU execution and pinned direct dependencies.\n",
            "## At a glance\n",
            "| Dataset | Validation-selected model | Test one-step RMSE (kW) | Test one-step R² | Test 24 h RMSE (kW) | Test 24 h MAE (kW) | Test 24 h R² |",
            "|---|---|---:|---:|---:|---:|---:|"]
    runs = []
    for run in ["synthetic", "uci"]:
        m = load_json(path("outputs",run,"metrics.json"))
        q = load_json(path("outputs",run,"data_quality.json"))
        s = load_json(path("outputs",run,"simulation.json"))
        f = load_json(path("outputs",run,"forecast_meta.json"))
        manifest = load_json(path("outputs",run,"run_manifest.json"))
        runs.append((run,m,q,s,f,manifest))
        a,b=m['test_one_step'],m['test_day_ahead']
        text.append(f"| {run} | {m['selected_model_label']} | {a['rmse_kw']:.3f} | {a['r2']:.3f} | {b['rmse_kw']:.3f} | {b['mae_kw']:.3f} | {b['r2']:.3f} |")
    text.append("\nOne-step replay updates actual historical lags between predictions. A day-ahead origin recursively forecasts 24 hours without actual updates inside that window. Scores are not interchangeable. R² is not percentage accuracy. Different load scales make cross-site absolute RMSE comparisons inappropriate.\n")
    for run,m,q,s,f,manifest in runs:
        a,b=m['test_one_step'],m['test_day_ahead']
        text += [f"## {run}: {q['source']['title']}\n",
                 f"- Source type: **{q['source']['source_type']}**.",
                 f"- Cleaned grid: **{q['hourly_rows']:,} hours**, {q['start']} → {q['end']}.",
                 f"- Missing target hours before fill: {q['missing_target_hours_before_fill']}; causal fills: {q['imputed_target_hours']}; still missing: {q['remaining_missing_target_hours']}.",
                 f"- Selected: **{m['selected_model_label']}**, chosen on validation 24 h RMSE **{m['validation_day_ahead']['rmse_kw']:.3f} kW**.",
                 f"- Test observed one-step targets: **{a['n']:,}**.",
                 f"- Test day-ahead windows: **{m['test_backtest_origins']} sampled daily origins × 24 h**, **{b['n']} observed targets scored**; imputed targets excluded.",
                 f"- Test day-ahead WAPE: **{b['wape_pct']:.2f}%**; MAPE: {b['mape_pct']:.2f}% (unstable at low load).",
                 f"- Day-ahead RMSE reduction vs weekly-naive baseline: **{m['day_ahead_rmse_improvement_vs_weekly_baseline_pct']:.2f}%**; this is forecast-error reduction, not energy saved.",
                 f"- Empirical validation error radius: **{m['validation_error_radius_kw']:.3f} kW**; observed test band coverage **{m['empirical_test_band_coverage_pct']:.2f}%** (no guarantee).",
                 f"- Train/evaluate/final-refit stage elapsed here: {manifest['total_train_evaluate_seconds']:.1f} s; excludes data preparation, graph rendering and API startup. Hardware runtimes vary.",
                 f"- Input SHA256: `{q['source']['input_sha256']}`.\n",
                 "### Actual candidate comparison\n",
                 "| Candidate | Selected | Validation 24 h RMSE | Test 24 h RMSE | Test one-step RMSE | Test 24 h R² |",
                 "|---|---|---:|---:|---:|---:|"]
        comp=pd.read_csv(path('outputs',run,'model_comparison.csv'))
        for _,r in comp.iterrows():
            text.append(f"| {r['label']} | {'Yes' if r['selected'] else '—'} | {r['validation_day_ahead_rmse_kw']:.3f} | {r['test_day_ahead_rmse_kw']:.3f} | {r['test_one_step_rmse_kw']:.3f} | {r['test_day_ahead_r2']:.3f} |")
        text += ["\nAll RMSE values are kW on that same source/task. Selection is locked before the test loop.\n",
                 "### Genuine future forecast, not future actuals\n",
                 f"- Forecast cutoff: **{f['cutoff']}**.",
                 f"- Stored primary horizon: {f['hours']} hours, {f['forecast_start']} → {f['forecast_end']}.",
                 f"- Forecast energy: **{f['total_kwh']:.2f} kWh**; forecast peak **{f['peak_kw']:.2f} kW** at {f['peak_time']}.",
                 "- `forecast_168h.csv` provides exploratory longer output. The empirical band stops after hour 24, because longer horizons were not validated here.\n",
                 "### Default hypothetical first-24-hour business scenario\n",
                 f"- Energy before / after: **{s['baseline_energy_kwh']:.2f} / {s['shifted_energy_kwh']:.2f} kWh**.",
                 f"- Eligible energy actually moved in the heuristic: **{s['moved_energy_kwh']:.2f} kWh**.",
                 f"- Baseline / shifted illustrative costs: **INR {s['baseline_cost_inr']:.2f} / {s['shifted_cost_inr']:.2f}**.",
                 f"- Hypothetical saving: **INR {s['hypothetical_savings_inr']:.2f} ({s['hypothetical_savings_pct']:.2f}%)**.",
                 f"- Original / shifted peak: **{s['baseline_peak_kw']:.2f} / {s['shifted_peak_kw']:.2f} kW**.",
                 "- Modeled carbon reduction: **0 kg**, because total energy is conserved under a constant illustrative intensity.",
                 "- Rates (INR 5 / 8 / 12), flexibility (10%), job timing and carbon factor are assumptions, not sourced tariffs or measured outcomes.\n",
                 f"![Forecast from the last dataset observation](../images/{run}/forecast.png)\n",
                 f"![Model evidence](../images/{run}/model_comparison.png)\n"]
    text += ["## What should not be hidden\n",
             "The measured UCI household is much less predictable in relative terms than the virtual building. Its selected model's day-ahead WAPE is 48.44%; the supplied evidence does not establish operational readiness. The test-best UCI MLP did not replace RF because RF had already won validation. That MLP's train+validation fit emitted a max-iteration convergence warning, visible in the real log.\n",
             "Only up to 14 daily origins per split are sampled, with eligible history/target coverage. Results do not evaluate every test day, 48/168-hour accuracy, multisite transfer, current households or city-grid reliability. The final served model is refit on all history after held-out reporting; stored test scores are not in-sample scores from that final refit.\n",
             "## Evidence locations\n",
             "- `outputs/<run>/metrics.json`, `model_comparison.csv`, `split_report.json`.",
             "- `one_step_predictions.csv`, `backtest_predictions.csv`, `horizon_metrics.csv`.",
             "- `forecast.csv`, `forecast_168h.csv`, `forecast_meta.json`.",
             "- `simulation.csv`, `simulation.json`, `logs/pipeline.log`.",
             "- `models/<run>/metadata.json`, `data/raw/*.metadata.json`.",
             "- `outputs/test-results.txt`, `verification.json`, `browser-checks.json`.",
             "- Actual analytical plots and browser captures in `images/`.\n",
             "See [MODEL_CARD.md](MODEL_CARD.md), [DATA_CARD.md](DATA_CARD.md) and [CLAIMS_AUDIT.md](CLAIMS_AUDIT.md) for scope. No business intervention, realized saving or live grid connection is claimed.\n"]
    target=path('docs','RESULTS.md')
    target.write_text('\n'.join(text),encoding='utf-8')
    print('Wrote',target)


if __name__=='__main__':
    main()
````

## `scripts/export_source.py`

Executable evidence helper; see its docstring.

````python
"""Collect the actual implementation and README into a copy-paste source book.

Run: python scripts/export_source.py
This uses only Python's standard library. Edit individual files, then regenerate.
"""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
DESCRIPTIONS={
    'main.py':'CLI orchestration and phase selection; main entry point.',
    'app.py':'Validated Flask JSON API, trusted bundle cache, downloads and dashboard.',
    'generate_data.py':'Deterministic virtual-building data and honest synthetic metadata.',
    'src/__init__.py':'Source package/version marker.',
    'src/config.py':'Visible seeds, horizons, lag/split/fill settings and scenario assumptions.',
    'src/utils.py':'Shared paths, finite metrics, strict JSON, hashes and real logs.',
    'src/data.py':'Official public archive adapter, schema validation, causal cleaning and masks.',
    'src/features.py':'Past-only vectorized training features and identical inference row.',
    'src/models.py':'Naive baselines, Ridge, random forest, scaled MLP and fit warnings.',
    'src/evaluation.py':'Ordered selection, recursive backtests, untouched test reports and final refit.',
    'src/forecast.py':'Recursive future prediction, trusted bundle loading and limited error bands.',
    'src/simulation.py':'Transparent energy-conserving/headroom-limited load-shifting heuristic.',
    'src/visualization.py':'Nine reproducible analytical plots from actual saved evidence.',
    'templates/dashboard.html':'Complete self-contained responsive five-view UI and offline snapshot logic.',
    'requirements.txt':'Pinned tested direct runtime dependencies.',
    'requirements-dev.txt':'Optional automated tests and notebook development environment.',
    'pyproject.toml':'Pytest discovery/warning rules and code-style settings.',
    '.gitignore':'Excludes private credentials, environments, caches and heavy derived artifacts.',
    '.github/workflows/ci.yml':'Ready-to-run CPU checks on Python 3.11/3.12/3.13 after publication.',
    'README.md':'The complete copy-paste GitHub landing page requested in section K.',
}


def main():
    files=[ROOT/'generate_data.py',ROOT/'main.py',ROOT/'app.py']
    files += sorted((ROOT/'src').glob('*.py'))
    files += [ROOT/'templates/dashboard.html']
    files += sorted((ROOT/'scripts').glob('*.py'))
    files += sorted((ROOT/'tests').glob('*.py'))
    files += [ROOT/p for p in ['requirements.txt','requirements-dev.txt','pyproject.toml','.gitignore','.github/workflows/ci.yml','README.md']]
    lines=['# Complete source code — copy-paste ready\n',
           'Generated from the actual project files. Each heading is its target filename, followed by its purpose and full content. For beginner explanations read [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md). Edit source files and regenerate this book, not the other way around. Dataset/model binaries are separate artifacts, not hidden code.\n',
           'The executed notebook is included separately; its code cells appear at the end without bulky embedded image/base64 outputs. All runtime logic is provided below.\n']
    fence=chr(96)*4
    languages={'.py':'python','.html':'html','.toml':'toml','.yml':'yaml','.md':'markdown'}
    for f in files:
        rel=f.relative_to(ROOT).as_posix()
        desc=DESCRIPTIONS.get(rel, 'Executable evidence helper; see its docstring.' if rel.startswith('scripts/') else 'Isolated automated checks; fixtures are not portfolio measurements.')
        language=languages.get(f.suffix,'text')
        lines += [f'## `{rel}`\n',desc+'\n',fence+language,f.read_text(encoding='utf-8').rstrip(),fence+'\n']
    nb_path=ROOT/'notebooks/01_energy_forecasting.ipynb'
    if nb_path.exists():
        nb=json.loads(nb_path.read_text(encoding='utf-8'))
        lines += ['## `notebooks/01_energy_forecasting.ipynb` — code cells\n','An executed reading/learning path using the same modular pipeline. Open the notebook for narrative, actual outputs and figures.\n']
        count=0
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                count+=1
                source=cell['source']
                if isinstance(source,list):source=''.join(source)
                lines += [f'### Code cell {count}\n',fence+'python',source.rstrip(),fence+'\n']
    target=ROOT/'docs/FULL_SOURCE.md'
    target.write_text('\n'.join(lines),encoding='utf-8')
    print(f'Wrote {target} from {len(files)} text files ({target.stat().st_size:,} bytes).')


if __name__=='__main__':
    main()
````

## `scripts/package_project.py`

Executable evidence helper; see its docstring.

````python
"""Package the completed repository, including trusted demo bundles, for download.

Run: python scripts/package_project.py
Output: ../AI-Energy-Forecasting.zip
Caches, environments, .git and unneeded raw archives are never included.
Git publication uses .gitignore instead; a ZIP is not a fabricated Git history.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'.git','.cache','.venv','venv','__pycache__','.pytest_cache','.ruff_cache',
           '.ipynb_checkpoints','node_modules','.idea','.vscode'}
MANIFEST = ROOT / 'outputs/delivery_manifest.json'


def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    files=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or p==MANIFEST:
            continue
        rel=p.relative_to(ROOT)
        if any(part in EXCLUDE for part in rel.parts):
            continue
        if p.suffix in {'.pyc','.zip','.part','.pem','.key'} or p.name.startswith('.env'):
            continue
        files.append(p)
    files=sorted(files)
    entries=[{'file':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in files]
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'project':'AI-Powered Energy Consumption Forecasting System',
              'publication_status':'Local deliverable; not published to a student GitHub account.',
              'scope':'Complete code, measured and synthetic experiments, trusted demo bundles and real local evidence.',
              'file_count_excluding_this_manifest':len(entries),'files':entries}
    MANIFEST.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    target=ROOT.parent/'AI-Energy-Forecasting.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [*files,MANIFEST]:
            z.write(p,arcname=f'{ROOT.name}/{p.relative_to(ROOT).as_posix()}')
    with zipfile.ZipFile(target) as z:
        bad=z.testzip()
        if bad: raise ValueError(f'Archive CRC failure: {bad}')
    print(f'Packaged {len(files)+1} files → {target} ({target.stat().st_size:,} bytes)')
    print('ZIP SHA256:',sha256(target))
    return target


if __name__=='__main__':
    main()
````

## `scripts/validate_project.py`

Executable evidence helper; see its docstring.

````python
"""Verify actual artifacts and provenance without retraining or any network.

Run: python scripts/validate_project.py
Failure exits nonzero; results are saved in outputs/verification.json.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from app import app, available_runs
from src.utils import path, load_json, save_json, sha256_file, compute_metrics


def main():
    checks = []
    def check(name, condition, run="project"):
        checks.append({"run": run, "check": name, "passed": bool(condition)})
    with app.test_client() as client:
        check("Dashboard renders", client.get("/").status_code == 200)
        for run in available_runs():
            out = path("outputs", run)
            metrics = load_json(out / "metrics.json")
            comp = pd.read_csv(out / "model_comparison.csv")
            one = pd.read_csv(out / "one_step_predictions.csv")
            bt = pd.read_csv(out / "backtest_predictions.csv")
            fc = pd.read_csv(out / "forecast_168h.csv", parse_dates=["Datetime"])
            sim = load_json(out / "simulation.json")
            meta = load_json(path("models", run, "metadata.json"))
            quality = load_json(out / "data_quality.json")
            raw = path("data", "raw", quality["source"]["input_file"])
            check("Raw input SHA256 matches run", raw.is_file() and sha256_file(raw)==quality["source"]["input_sha256"], run)
            check("Champion chosen by validation only", comp.loc[comp.validation_day_ahead_rmse_kw.idxmin(), "model"]==metrics["selected_model"], run)
            for label, rows in [("one_step", one), ("day_ahead", bt)]:
                measured = compute_metrics(rows.actual_kw, rows.predicted_kw)
                check(f"Stored {label} RMSE matches saved predictions", np.isclose(measured["rmse_kw"],metrics[f"test_{label}"]["rmse_kw"],atol=1e-5), run)
                check(f"Stored {label} sample count matches observations", measured["n"]==metrics[f"test_{label}"]["n"], run)
            check("Future begins exactly after cutoff", fc.Datetime.iloc[0]==pd.Timestamp(meta["cutoff"])+pd.Timedelta(hours=1), run)
            check("168 nonnegative finite hourly forecasts", len(fc)==168 and np.isfinite(fc.predicted_kw).all() and (fc.predicted_kw>=0).all(), run)
            check("No band beyond validated 24-hour horizon", fc.lower_kw.iloc[24:].isna().all(), run)
            check("Simulation conserves energy", np.isclose(sim["baseline_energy_kwh"],sim["shifted_energy_kwh"]), run)
            check("Simulation respects original forecast peak", sim["shifted_peak_kw"]<=sim["baseline_peak_kw"]+1e-8, run)
            check("No invented carbon benefit from shifting", sim["emissions_reduction_kg"]==0, run)
            response = client.post("/predict",json={"dataset":run,"hours":24})
            check("Saved model serves valid JSON", response.status_code==200 and len(response.json.get("predictions",[]))==24, run)
            check("Nine evidence figures exist", len(list(path("images",run).glob("*.png")))>=9, run)
    summary={"checked_at_utc":datetime.now(timezone.utc).isoformat(),"passed":all(c["passed"] for c in checks),
             "total_checks":len(checks),"checks":checks,
             "scope":"Artifact/provenance checks. Automated unit/API tests and browser checks are reported separately."}
    save_json(summary,path("outputs","verification.json"))
    for c in checks:
        print(f"{'PASS' if c['passed'] else 'FAIL'} | {c['run']} | {c['check']}")
    print(f"{sum(c['passed'] for c in checks)}/{len(checks)} artifact checks passed")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
````

## `tests/conftest.py`

Isolated automated checks; fixtures are not portfolio measurements.

````python
"""Fixtures are isolated in pytest temp directories, never bundled evidence."""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def hourly_frame():
    idx = pd.date_range("2024-01-01", periods=1200, freq="h")
    values = 50 + 12*np.sin(2*np.pi*idx.hour/24) + np.arange(len(idx))*.005
    return pd.DataFrame({"Energy_kW": values, "target_observed": True}, index=idx)


@pytest.fixture(scope="module")
def integration_run(tmp_path_factory):
    """A small genuine offline end-to-end run; no shipped model is required."""
    from src import utils
    from generate_data import generate
    from src.data import prepare
    from src.evaluation import train_evaluate
    from src.forecast import save_forecast
    from src.simulation import save_simulation
    from app import create_app, cached_bundle, cached_forecast
    root = tmp_path_factory.mktemp("energy_integration")
    original = utils.PROJECT_ROOT
    utils.PROJECT_ROOT = root
    cached_bundle.cache_clear()
    cached_forecast.cache_clear()
    try:
        generate(60, 42, root / "data/raw/energy_data.csv")
        frame, report = prepare("synthetic", "synthetic")
        metrics = train_evaluate(frame, report, "synthetic", include_mlp=False, quick=True)
        forecast = save_forecast("synthetic")
        simulation = save_simulation("synthetic")
        app = create_app()
        app.config.update(TESTING=True)
        yield {"root": root, "frame": frame, "metrics": metrics, "forecast": forecast,
               "simulation": simulation, "client": app.test_client()}
    finally:
        utils.PROJECT_ROOT = original
        cached_bundle.cache_clear()
        cached_forecast.cache_clear()
````

## `tests/test_api_and_pipeline.py`

Isolated automated checks; fixtures are not portfolio measurements.

````python
import json
from pathlib import Path
import numpy as np
import pytest
from main import parse_args


def test_end_to_end_artifacts(integration_run):
    run = integration_run
    out = run["root"] / "outputs/synthetic"
    for name in ['metrics.json','data_quality.json','model_comparison.csv','forecast.csv',
                 'forecast_168h.csv','backtest_predictions.csv','simulation.json','run_manifest.json']:
        assert (out/name).is_file()
    assert run['metrics']['selection_rule'].startswith('Minimum validation')
    assert run['forecast'].index.min() > run['frame'].index.max()
    assert run['simulation']['baseline_energy_kwh'] == pytest.approx(run['simulation']['shifted_energy_kwh'])
    assert (run['root']/'models/synthetic/forecast_bundle.joblib').exists()


def test_health_and_dashboard(integration_run):
    client = integration_run['client']
    assert client.get('/health').json['ready_runs'] == ['synthetic']
    assert client.get('/').status_code == 200
    assert b'No live grid connection' in client.get('/').data


def test_valid_forecast_post_and_get(integration_run):
    client = integration_run['client']
    post = client.post('/predict', json={'dataset':'synthetic','hours':24})
    assert post.status_code == 200
    value = post.json
    assert value['hours'] == 24 and len(value['predictions']) == 24
    get = client.get('/api/forecast?dataset=synthetic&hours=24')
    assert get.json['predictions'] == value['predictions']
    assert value['source_type'] == 'synthetic'


@pytest.mark.parametrize('body', [{'hours':0},{'hours':169},{'hours':True},{'hours':24.3},
 {'hours':'foo'},{'dataset':'../../etc'},{'dataset':[]},{'hour':14,'day':2},[],None])
def test_bad_predict_returns_400(integration_run, body):
    res = integration_run['client'].post('/predict', data=json.dumps(body), content_type='application/json')
    assert res.status_code == 400
    assert res.json['error'] == 'invalid_request'


def test_malformed_json_and_large_request(integration_run):
    client = integration_run['client']
    assert client.post('/predict', data='{bad', content_type='application/json').status_code == 400
    assert client.post('/predict', data=' '*17000, content_type='application/json').status_code == 413


def test_api_summary_history_and_evidence(integration_run):
    client = integration_run['client']
    for endpoint in ['/api/runs','/api/summary','/api/history','/api/evaluation']:
        res = client.get(endpoint)
        assert res.status_code == 200
        json.loads(res.data, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def test_simulation_endpoint_and_validation(integration_run):
    client = integration_run['client']
    result = client.post('/api/simulate', json={'dataset':'synthetic','settings':{'flexible_fraction':.2}})
    assert result.status_code == 200
    r = result.json['result']
    assert r['baseline_energy_kwh'] == pytest.approx(r['shifted_energy_kwh'])
    assert client.post('/api/simulate',json={'settings':{'offpeak_rate':100}}).status_code == 400


def test_download_allowlist(integration_run):
    client = integration_run['client']
    res = client.get('/download/synthetic/metrics.json')
    assert res.status_code == 200
    assert 'attachment' in res.headers['Content-Disposition']
    assert client.get('/download/synthetic/forecast_bundle.joblib').status_code == 400
    assert client.get('/images/synthetic/unknown.png').status_code == 400


@pytest.mark.parametrize('args', [['--horizon','0'],['--run-id','../../secret'],['--run-id','']])
def test_cli_input_validation(args):
    with pytest.raises(SystemExit):
        parse_args(args)
````

## `tests/test_data.py`

Isolated automated checks; fixtures are not portfolio measurements.

````python
from io import StringIO
import zipfile
import numpy as np
import pandas as pd
import pytest
from generate_data import make_dataset
from src.data import clean_data


def write_csv(tmp_path, frame):
    p = tmp_path / "input.csv"
    frame.to_csv(p, index=False)
    return p


def test_generator_repeatable():
    pd.testing.assert_frame_equal(make_dataset(30, 42), make_dataset(30, 42))
    assert not make_dataset(30, 42).equals(make_dataset(30, 43))


def test_generator_shape_and_units():
    frame = make_dataset(30)
    assert len(frame) == 720
    assert frame.Energy_kW.dropna().min() >= 0
    assert frame.Datetime.is_monotonic_increasing
    assert frame.Energy_kW.isna().sum() == int(720*.005)


def test_generator_requires_history():
    with pytest.raises(ValueError):
        make_dataset(7)


def test_causal_bounded_fill_and_mask(tmp_path):
    raw = pd.DataFrame({"Datetime": pd.date_range("2024-01-01", periods=8, freq="h"),
                        "Energy_kW": [10, np.nan, np.nan, np.nan, np.nan, 20, -1, 30]})
    clean, report = clean_data(write_csv(tmp_path, raw))
    assert clean.Energy_kW.iloc[1:4].eq(10).all()
    assert np.isnan(clean.Energy_kW.iloc[4])
    assert clean.Energy_kW.iloc[6] == 20
    assert not clean.target_observed.iloc[1:5].any()
    assert not clean.target_observed.iloc[6]
    assert report["invalid_power_readings"] == 1
    assert report["remaining_missing_target_hours"] == 1
    assert np.allclose(clean.Energy_kWh, clean.Energy_kW, equal_nan=True)


def test_duplicate_and_subhourly_power_is_mean_not_sum(tmp_path):
    raw = pd.DataFrame({"Datetime": ["2024-01-01 01:00", "2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 00:30"],
                        "Energy_kW": [8, 2, 4, 6]})
    clean, report = clean_data(write_csv(tmp_path, raw))
    assert clean.Energy_kW.iloc[0] == 4
    assert clean.index.is_monotonic_increasing and clean.index.is_unique
    assert report["duplicate_timestamps_aggregated"] == 1


def test_future_change_does_not_change_earlier_cleaning(tmp_path):
    frame = pd.DataFrame({"Datetime": pd.date_range("2024-01-01", periods=6, freq="h"),
                          "Energy_kW": [4, np.nan, np.nan, 9, 10, 11]})
    first, _ = clean_data(write_csv(tmp_path, frame))
    frame.loc[3:, "Energy_kW"] = 1000
    second, _ = clean_data(write_csv(tmp_path, frame))
    pd.testing.assert_series_equal(first.Energy_kW.iloc[:3], second.Energy_kW.iloc[:3])


@pytest.mark.parametrize("frame", [pd.DataFrame({"wrong": [1]}),
    pd.DataFrame({"Datetime": ["nonsense"], "Energy_kW": [1]}),
    pd.DataFrame({"Datetime": ["2024-01-01"], "Energy_kW": [np.nan]}),
    pd.DataFrame(columns=["Datetime", "Energy_kW"]),
    pd.DataFrame({"Datetime": ["2024-01-01T00:00:00+00:00"], "Energy_kW": [1]})])
def test_bad_csv_fails_clearly(tmp_path, frame):
    with pytest.raises(ValueError):
        clean_data(write_csv(tmp_path, frame))


def test_uci_aggregation_uses_valid_minute_coverage(tmp_path, monkeypatch):
    """Tiny local source archive; no web request and no 126 MB extraction."""
    import src.data as module
    monkeypatch.setattr(module, "path", lambda *p: tmp_path.joinpath(*p))
    cache = tmp_path / ".cache/downloads"
    cache.mkdir(parents=True)
    dt = pd.date_range("2007-01-01", periods=125, freq="min")
    df = pd.DataFrame({"Date": dt.strftime("%d/%m/%Y"), "Time": dt.strftime("%H:%M:%S"),
                       "Global_active_power": [1.0]*60+[2.0]*60+[4.0]*5})
    with zipfile.ZipFile(cache / "uci_household.zip", "w") as z:
        z.writestr("household_power_consumption.txt", df.to_csv(index=False, sep=";"))
    p = module.download_uci()
    out = pd.read_csv(p)
    assert len(out) == 3
    assert out.Energy_kW.iloc[0] == 1.0 and out.Energy_kW.iloc[1] == 2.0
    assert np.isnan(out.Energy_kW.iloc[2])  # a 5-minute partial hour is NOT an hourly truth
    assert not (tmp_path / "household_power_consumption.txt").exists()
    assert p.with_suffix(".metadata.json").exists()
````

## `tests/test_forecasting.py`

Isolated automated checks; fixtures are not portfolio measurements.

````python
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
````

## `tests/test_simulation.py`

Isolated automated checks; fixtures are not portfolio measurements.

````python
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
````

## `requirements.txt`

Pinned tested direct runtime dependencies.

````text
# Tested on CPython 3.13.16. Recommended: Python 3.11 or 3.12 (64-bit).
# Exact runtime versions used for the bundled evidence runs.
numpy==2.3.5
pandas==2.2.3
scipy==1.17.1
scikit-learn==1.6.1
matplotlib==3.10.9
seaborn==0.13.2
joblib==1.5.3
Flask==3.1.3
requests==2.33.0
````

## `requirements-dev.txt`

Optional automated tests and notebook development environment.

````text
-r requirements.txt
pytest==9.0.3
nbformat==5.11.1
nbclient==0.11.0
ipykernel==6.31.0
jupyterlab>=4.3,<5
````

## `pyproject.toml`

Pytest discovery/warning rules and code-style settings.

````toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
filterwarnings = ["error::FutureWarning"]

[tool.ruff]
line-length = 110
target-version = "py311"
````

## `.gitignore`

Excludes private credentials, environments, caches and heavy derived artifacts.

````text
# Environments / credentials (never publish these)
.venv/
venv/
.env
.env.*
!.env.example
*.pem
*.key
# Python and tooling
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.mypy_cache/
.ipynb_checkpoints/
.cache/
.DS_Store
Thumbs.db
.vscode/
.idea/
# Regenerate large/derived artifacts instead of bloating Git history.
# The downloadable project includes trained bundles for instant demonstration.
# A fresh Git clone should run python main.py for each desired dataset.
models/**/*.joblib
models/**/*.pkl
data/processed/**/*.csv
data/raw/*.zip
data/raw/household_power_consumption.txt
# Keep small, actual evidence CSV/JSON/PNG files and executed notebooks tracked.
# Do not add private user data; see DATA_LICENSE.md and SECURITY.md.
outputs/ci-tests.xml
````

## `.github/workflows/ci.yml`

Ready-to-run CPU checks on Python 3.11/3.12/3.13 after publication.

````yaml
name: Energy forecasting checks
on:
  push:
  pull_request:
permissions:
  contents: read
jobs:
  tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install CPU runtime and tests
        run: python -m pip install -r requirements.txt pytest==9.0.3
      - name: Syntax checks
        run: python -m compileall -q src main.py app.py generate_data.py
      - name: Unit, leakage, API and offline pipeline checks
        run: python -m pytest -q --junitxml=outputs/ci-tests.xml
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-report-python-${{ matrix.python-version }}
          path: outputs/ci-tests.xml
````

## `README.md`

The complete copy-paste GitHub landing page requested in section K.

````markdown
# AI-Powered Energy Consumption Forecasting System

### Historical demand → reliable evaluation → practical virtual decisions

A complete, CPU-friendly machine-learning project for forecasting hourly electricity
use in homes/buildings. Built as student proof of work: **real public data + a
reproducible virtual building**, not a claimed corporate deployment.

**Python · pandas · scikit-learn · Flask · matplotlib · pytest**

[Start here](START_HERE.md) · [Complete A–M guide](docs/PROJECT_GUIDE.md) ·
[Full source](docs/FULL_SOURCE.md) · [Executed notebook](notebooks/01_energy_forecasting.ipynb) ·
[Actual results](docs/RESULTS.md) · [API](docs/API.md)

![Forecasting dashboard: synthetic building](images/screenshots/01_overview_synthetic.png)

> **Transparency:** this is historical-data replay and virtual planning, not a live
> grid connection. Predictions follow the dataset cutoff, not today's date. Savings
> are hypothetical. R² is not percentage accuracy. No blackout-prevention, actual
> energy-saving or emissions-reduction outcome is claimed.

## Overview

Energy demand changes with time, weather and human activity. Facility planners need
a defensible estimate of future demand to schedule flexible work, anticipate peaks
and budget energy. This repository implements the whole experiment rather than a
single model-fitting notebook: ingest, clean, engineer, compare, evaluate, forecast,
simulate, visualize, serve and verify.

### Problem statement

Uncertain hourly demand can make supply planning and peak-time electricity use
more difficult. The project estimates upcoming power and explores one possible
business response: shifting eligible work from illustrative peak-rate hours into
lower-rate headroom. **Forecasting is decision support, not a guarantee of grid
reliability or an automatic energy optimizer.**

### Industry relevance

| Context | Decision forecasting can support | Additional real-system needs |
|---|---|---|
| Smart homes/buildings | HVAC/flexible-appliance planning, consumption budgets | Comfort constraints and real tariffs |
| Smart cities | Aggregate facility demand and peak awareness | Representative multi-site data and network models |
| Electricity boards | Demand planning and procurement inputs | Grid constraints, reserves, weather and safety processes |
| Manufacturing | Scheduling flexible process loads | Production deadlines, machine limits, shift calendars |
| Data centers | Capacity/cooling planning inputs | SLA/thermal safety constraints and approved controls |
| Renewable energy | Demand/net-load planning | Separate solar/wind forecasts and storage constraints |

These are potential applications, not deployments performed by this project.

## What is implemented

- [x] Deterministic virtual-building generator: 17,520 hourly rows, realistic routines,
  temperature effects, occupancy proxies, noise and deliberately missing readings.
- [x] Genuine UCI household benchmark: 2,075,259 minute rows aggregated to hourly kW;
  bundled adaptation includes source/coverage/license metadata and SHA256 hashes.
- [x] Causal cleaning: sorted hourly grid, duplicate averaging, bounded past-only
  fill, long-gap preservation, original-observation mask.
- [x] 15 calendar/lag/rolling features with training–serving parity checks.
- [x] Previous-hour, yesterday and last-week baselines; Ridge, random forest and MLP.
- [x] Chronological 70/15/15 splits; champion locked on validation day-ahead RMSE.
- [x] Separate one-step and fixed-origin recursive 24-hour test evaluations.
- [x] Final all-history refit, reusable joblib bundle, 1–168-hour forecasts; >24 h
  explicitly exploratory, empirical error bands only for the first 24 h.
- [x] Energy-conserving, headroom-constrained load-shifting sandbox with transparent
  illustrative tariff assumptions and no invented carbon reduction.
- [x] Flask API, responsive five-view dashboard, standalone offline HTML export.
- [x] Nine analytical PNG figures per dataset, actual browser screenshots, CSV/JSON
  evidence, executed notebook and actual pipeline logs.
- [x] Automated unit/leakage/API/offline pipeline tests; artifact/browser checks;
  GitHub Actions workflow ready for your repository.

## Tech stack

| Layer | Choice |
|---|---|
| Python | 3.11–3.13; build evidence executed on 3.13.16 |
| Data | pandas, NumPy, public UCI CSV adaptation, seeded synthetic CSV |
| ML | scikit-learn Ridge, RandomForestRegressor, MLPRegressor |
| Scaling/serialization | StandardScaler pipelines, joblib |
| Evaluation | MAE, RMSE, R², MAPE, WAPE; rolling forecast origins |
| Visuals | matplotlib / seaborn; dependency-free SVG web charts |
| Serving | Flask JSON endpoints and inline HTML/CSS/JS |
| Quality | pytest; CI matrix for Python 3.11/3.12/3.13 |
| Hardware | CPU; no GPU, cloud account or paid service required |

Direct runtime dependencies are pinned in [requirements.txt](requirements.txt).
Three beginner-to-advanced alternatives are explained in the [guide](docs/PROJECT_GUIDE.md#b-tech-stack-options).

## Dataset

### 1. Synthetic virtual building — default, offline

`data/raw/energy_data.csv`, seed 42, 730 days starting 2023-01-01. `Energy_kW`
means **hour-average power**, despite the column's legacy name. Temperature and
occupancy are for exploration only; their future realized values are not assumed
known by the forecasting model. This is generated data, not smart-grid client logs.

### 2. Public measured household — real benchmark

Georges Hebrail & Alice Berard, **Individual Household Electric Power Consumption**,
UCI Machine Learning Repository, DOI [10.24432/C58K54](https://doi.org/10.24432/C58K54).
One household in Sceaux, France, December 2006–November 2010. Data: **CC BY 4.0**.

[Official dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) ·
[Attribution & adaptations](DATA_LICENSE.md) · [Data card](docs/DATA_CARD.md)

The bundled hourly adaptation averages minute power only for hours with at least
54 valid source minutes. Unobserved boundary hours are trimmed. Final cleaned grid:
34,587 hours; 411 long-gap hours remain missing. Do not call this a city-scale grid
benchmark or a representative sample of households.

**Units:** hourly energy (kWh) = average power (kW) × 1 hour. The 24-hour forecast
energy is the sum of interval kWh, not the mean or sum of arbitrary-duration samples.

## Architecture

```text
 Public UCI minute archive ──> coverage-aware hourly aggregation ─┐
 Seeded virtual building ────────────────────────────────────────┼─> Hourly CSV
 Your permitted hourly CSV ──────────────────────────────────────┘
       │
       v
 Causal cleaning + observation mask + data quality / source hashes
       │
       v
 Past-only lags / rolling statistics + known cyclic calendar
       │
       v
 Ordered 70% train | 15% validation | 15% test
       │
       ├─> 3 naive baselines + Ridge + Random forest + MLP
       ├─> Validation 24-hour backtests → LOCK champion
       ├─> Train+validation refit → untouched test evidence
       └─> Final historical refit → trusted local model bundle
                                     │
                                     v
                        Recursive future hourly forecasting
                                     │
                      ┌──────────────┴───────────────┐
                      v                              v
                CSV / JSON / plots        Virtual load-shifting sandbox
                      └──────────────┬───────────────┘
                                     v
                           Flask API + web dashboard
```

![System architecture](images/architecture.svg)

## Installation

### Windows — PowerShell

Install 64-bit Python 3.11, 3.12 or 3.13 first. From the extracted project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use `py -3.12` or `py -3.13` if that is the version you installed. If activation is
blocked, either use CMD's `.venv\Scripts\activate.bat` or run the environment's
Python directly: `.\.venv\Scripts\python.exe main.py`.

### macOS / Linux

```bash
python3 --version  # must be 3.11, 3.12 or 3.13
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No GPU configuration is required. Detailed setup, common mistakes and Colab steps
are in the [full guide](docs/PROJECT_GUIDE.md#f-installation-and-environment-setup).

## Usage

### Run the full experiment

```bash
python main.py                         # default: synthetic building
python main.py --dataset uci           # real measured benchmark
python app.py                         # http://127.0.0.1:5000
```

The downloadable archive includes models and processed data for immediate viewing.
A fresh Git clone omits these derived artifacts by `.gitignore`; run the experiment
first. Both small hourly source CSVs are bundled. `--refresh-data` intentionally
regenerates/downloads source data and requires internet for UCI only.

### Run phase by phase

```bash
python main.py prepare --dataset synthetic
python main.py train --dataset synthetic
python main.py evaluate --dataset synthetic
python main.py forecast --dataset synthetic --horizon 48
python main.py simulate --dataset synthetic
python main.py visualize --dataset synthetic
python scripts/export_dashboard.py
```

`train` includes validation selection and test evaluation. `evaluate` prints the
saved pre-final-refit test report; it does **not** incorrectly rescore the final
all-history model on data it has now trained on. Serving forecasts require saved
168-hour lag history, not merely arbitrary `hour` and `day` inputs.

For a faster learning smoke run use a separate ID so you preserve benchmark files:

```bash
python main.py --dataset synthetic --run-id smoke --quick --no-mlp
```

### Use your own permitted data

```bash
python main.py --dataset csv --csv data/raw/my_building.csv --run-id building
```

Required columns: `Datetime,Energy_kW`; at least 60 days recommended. Do not upload
private meter data, cumulative register kWh, or irregular-duration energy as power.
See the [data card](docs/DATA_CARD.md) for the schema and limitations.

### API example

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"dataset":"synthetic","hours":24}'
```

Browser-friendly GET:
`http://127.0.0.1:5000/api/forecast?dataset=synthetic&hours=24`.
See [API.md](docs/API.md) for response fields, simulation requests and PowerShell.

## Actual results

These figures come from the **executed bundled runs**, not desired target scores.
Model selection uses validation, not whichever test metric looks best.

| Dataset | Selected on validation | Test one-step RMSE | Test one-step R² | Test 24 h RMSE | Test 24 h MAE | Test 24 h R² |
|---|---|---:|---:|---:|---:|---:|
| Synthetic building | MLP | 6.711 kW | 0.831 | 6.888 kW | 5.485 kW | 0.829 |
| UCI household | Random forest | 0.466 kW | 0.567 | 0.598 kW | 0.448 kW | 0.253 |

- One-step updates actual lag readings between predictions; day-ahead recursively
  predicts 24 hours without updates inside each window.
- Day-ahead test: **14 sampled daily origins** per dataset, 335 observed synthetic
  targets / 336 UCI targets. Not all test days and not a seven-day evaluation.
- Day-ahead RMSE reduction vs the same-hour-last-week baseline: **24.01%** synthetic,
  **29.59%** UCI. These are forecast-error reductions, **not energy or bill savings**.
- The UCI MLP scored better on sampled test windows but was not selected: RF had
  won validation. The champion remains locked to avoid test-set cherry-picking.
- UCI day-ahead WAPE is **48.44%**; this honest limitation rules out claiming
  operational-quality household predictions from this benchmark alone.

### Virtual business scenario — first 24 forecast hours

At illustrative INR 5 / 8 / 12 per kWh, moving 10% of peak-hour forecast energy:

| Dataset | Energy before / after | Hypothetical saving | Modeled CO₂ reduction |
|---|---:|---:|---:|
| Synthetic | 2,543.53 / 2,543.53 kWh | ₹438.55 / 2.09% | 0 kg |
| UCI | 31.30 / 31.30 kWh | ₹6.70 / 2.40% | 0 kg |

This is not a real tariff quote or achieved saving. Demand charges, tax, comfort,
physical device schedules, rebound and forecast error are not modeled. Energy is
conserved. A constant carbon factor means load shifting alone does not reduce the
modeled carbon total. See [model card](docs/MODEL_CARD.md) and [results](docs/RESULTS.md).

## Screenshots and figures

![Future forecast and empirical validation band](images/synthetic/forecast.png)
![Held-out model comparison](images/uci/model_comparison.png)
![Interactive decision sandbox](images/screenshots/02_decision_sandbox.png)

Additional actual browser captures are in `images/screenshots/`. Each dataset has
nine generated analytical plots in `images/<dataset>/`, with CSV/JSON backing data
in `outputs/<dataset>/`. There is no fabricated GitHub repository screenshot.

## Repository layout

```text
AI-Energy-Forecasting/
├── data/                 # attributed raw hourly CSVs; regenerated processed data
├── notebooks/            # executed, beginner-friendly learning notebook
├── src/                  # cleaning, features, models, evaluation, forecast, simulation
├── models/               # trusted bundles in archive; metadata; binaries gitignored
├── outputs/              # real CSV/JSON/log evidence and offline dashboard.html
├── images/               # generated plots, architecture SVG, actual browser captures
├── docs/                 # A–M guide, full source, cards, GitHub/interview/proof playbooks
├── templates/            # responsive self-contained dashboard template
├── tests/                # isolated unit, leakage, API and end-to-end checks
├── scripts/              # export, evidence verification, optional browser capture
├── .github/workflows/    # CI configuration (runs once you publish)
├── main.py               # one-command CLI
├── app.py                # Flask API + dashboard
├── generate_data.py      # deterministic simulation input
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── DATA_LICENSE.md
├── LICENSE
└── README.md
```

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/validate_project.py
```

Bundled execution: **67 automated tests passed**, **29/29 artifact checks passed**.
Actual reports: `outputs/test-results.txt`, `outputs/test-results.xml`,
`outputs/verification.json`. Optional browser checks and instructions are in
`scripts/browser_check.py`; the included screenshots are real app captures.
GitHub Actions is configured but no remote CI success is claimed before publication.

## Learning outcomes

Time-series preprocessing; power versus energy units; leakage prevention; cyclic
features and lag windows; linear/tree/neural regression; train-only scaling;
chronological model selection; one-step versus multistep evaluation; practical
baseline comparisons; recursive forecast uncertainty; simulation assumptions;
Flask JSON serving; automated tests; reproducibility and professional proof of work.

## GitHub / placement showcase

Recommended repo name: **`ai-energy-consumption-forecasting`**.
Use the [GitHub playbook](docs/GITHUB_PLAYBOOK.md) for commands, topics and seven
honest days of commits. Use [DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) and
[INTERVIEW_QA.md](docs/INTERVIEW_QA.md) to explain what the code actually does.
The project is GitHub-ready; it has **not been published to your account**.

## Limitations / next steps

No live meter feed or automated equipment control; no representative multi-site
validation; naive dataset clocks with unresolved UCI DST semantics; no external
forecast weather/holidays; no formal probabilistic guarantee; no 48/168-hour
backtest; only sampled day-ahead origins; single-site and old/synthetic data;
Flask development server is not production infrastructure.

Useful extensions: more rolling origins, direct multihorizon models, holiday and
forecast-weather features with issuance timestamps, timezone handling, calibrated
probabilistic forecasts, drift monitoring and authenticated production deployment.
These are **future work**, not implemented features. See [SECURITY.md](SECURITY.md).

## License and attribution

Software: [MIT](LICENSE). Public UCI data and derivatives: **CC BY 4.0**, credited to
Georges Hebrail and Alice Berard; see [DATA_LICENSE.md](DATA_LICENSE.md). Synthetic
data: CC0. No endorsement by UCI or any energy company is implied. This scaffold
uses AI assistance; students should verify, understand and extend it honestly.
````

## `notebooks/01_energy_forecasting.ipynb` — code cells

An executed reading/learning path using the same modular pipeline. Open the notebook for narrative, actual outputs and figures.

### Code cell 1

````python
from pathlib import Path
import os, sys, json, subprocess

ROOT = Path.cwd().resolve()
while not (ROOT / 'main.py').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if not (ROOT / 'main.py').exists():
    raise FileNotFoundError('Open this notebook inside AI-Energy-Forecasting.')
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
from importlib.metadata import version
from src.utils import path, load_json, compute_metrics
print('Project:', ROOT.name)
print({p: version(p) for p in ['numpy','pandas','scikit-learn','Flask']})
````

### Code cell 2

````python
RUN = 'synthetic'  # change to 'uci' to explore the measured benchmark
if not path('outputs', RUN, 'metrics.json').exists() or not path('models', RUN, 'forecast_bundle.joblib').exists():
    subprocess.run([sys.executable, 'main.py', '--dataset', RUN], cwd=ROOT, check=True)
    print('Executed the full pipeline because artifacts were absent.')
else:
    print('Reading the actual bundled run; no claim of fresh retraining in this cell.')
from src.data import load_processed
frame = load_processed(RUN)
quality = load_json(path('outputs', RUN, 'data_quality.json'))
metrics = load_json(path('outputs', RUN, 'metrics.json'))
print(quality['source']['title'])
print('Source type:', quality['source']['source_type'])
print('Period:', quality['start'], 'to', quality['end'])
display(frame.head(10))
````

### Code cell 3

````python
audit_keys = ['hourly_rows','observed_target_hours','missing_target_hours_before_fill',
              'imputed_target_hours','remaining_missing_target_hours']
display(pd.Series({k:quality[k] for k in audit_keys}, name='Audit'))
assert frame.index.is_monotonic_increasing and frame.index.is_unique
assert np.allclose(frame.Energy_kW, frame.Energy_kWh, equal_nan=True)
print('Filled target hours are excluded as labels, not silently treated as measurements.')
````

### Code cell 4

````python
fig, axes = plt.subplots(1,2,figsize=(12,3.5))
frame.Energy_kW.tail(168).plot(ax=axes[0],color='#098974',lw=1.2,title='Last week of stored history')
axes[0].set_ylabel('Mean power (kW)')
obs = frame.loc[frame.target_observed,'Energy_kW']
obs.groupby(obs.index.hour).mean().plot(ax=axes[1],color='#386592',marker='o',ms=3,title='Mean observed power by hour')
axes[1].set(xlabel='Dataset hour',ylabel='kW')
plt.tight_layout(); plt.show()
````

### Code cell 5

````python
from src.features import build_features, next_feature_row
X, y = build_features(frame)
print('Eligible samples:',len(X),'| Features:',len(X.columns))
display(X.head(3).round(3))
t = X.index[100]
history = frame.loc[:t-pd.Timedelta(hours=1),'Energy_kW']
np.testing.assert_allclose(next_feature_row(t,history).values,X.loc[[t]].values,atol=1e-10)
changed = frame.copy()
changed.loc[t:,'Energy_kW'] = 99999.0
X_changed, _ = build_features(changed)
np.testing.assert_allclose(X.loc[t],X_changed.loc[t])
print('PASS: feature parity and no current/future target leakage.')
````

### Code cell 6

````python
from src.evaluation import chronological_split
splits = chronological_split(X,y)
rows=[]
for name, (xx,yy) in splits.items():
    rows.append({'split':name,'rows':len(yy),'start':str(yy.index.min()),'end':str(yy.index.max())})
display(pd.DataFrame(rows))
assert splits['train'][0].index.max() < splits['validation'][0].index.min()
assert splits['validation'][0].index.max() < splits['test'][0].index.min()
````

### Code cell 7

````python
from src.models import fit_candidate, predict_rows
X_train,y_train=splits['train']; X_val,y_val=splits['validation']
ridge_spec,seconds,warnings = fit_candidate('ridge',X_train,y_train)
ridge_val = compute_metrics(y_val,predict_rows(ridge_spec,X_val))
print('Fresh educational Ridge fit — validation one-step only:')
display(pd.Series(ridge_val).round(4))
print('Scaler fit rows:',len(X_train),'| fit seconds:',round(seconds,3))
````

### Code cell 8

````python
comparison=pd.read_csv(path('outputs',RUN,'model_comparison.csv'))
cols=['label','selected','validation_day_ahead_rmse_kw','test_day_ahead_rmse_kw','test_one_step_rmse_kw','test_day_ahead_r2']
display(comparison[cols].round(4))
print('Validation-selected:',metrics['selected_model_label'])
print('Test one-step:',metrics['test_one_step'])
print('Test day-ahead:',metrics['test_day_ahead'])
print('R² is NOT percentage accuracy. Origins:',metrics['test_backtest_origins'])
````

### Code cell 9

````python
backtest=pd.read_csv(path('outputs',RUN,'backtest_predictions.csv'))
recomputed=compute_metrics(backtest.actual_kw,backtest.predicted_kw)
assert np.isclose(recomputed['rmse_kw'],metrics['test_day_ahead']['rmse_kw'],atol=1e-5)
print('PASS: stored day-ahead RMSE matches actual prediction CSV.')
horizons=pd.read_csv(path('outputs',RUN,'horizon_metrics.csv'))
ax=horizons.plot(x='horizon',y='rmse_kw',color='#098974',marker='o',figsize=(9,3),legend=False,title='Held-out error by forecast lead time')
ax.set(xlabel='Hours ahead',ylabel='RMSE (kW)');plt.tight_layout();plt.show()
````

### Code cell 10

````python
from src.forecast import load_bundle, forecast_bundle
bundle=load_bundle(RUN)
future=forecast_bundle(bundle,48)
print('Cutoff:',bundle['cutoff'],'| first future hour:',future.index[0])
print('Next 24 h energy:',round(future.predicted_kwh.iloc[:24].sum(),2),'kWh')
assert future.index[0] > pd.Timestamp(bundle['cutoff'])
assert future.lower_kw.iloc[24:].isna().all()
display(future.head().round(3))
display(Image(filename=str(path('images',RUN,'forecast.png'))))
````

### Code cell 11

````python
from src.simulation import simulate
scenario_frame,scenario=simulate(future,{'flexible_fraction':0.10})
keys=['baseline_energy_kwh','shifted_energy_kwh','moved_energy_kwh','baseline_cost_inr',
      'shifted_cost_inr','hypothetical_savings_inr','hypothetical_savings_pct','emissions_reduction_kg']
display(pd.Series({k:scenario[k] for k in keys}).round(3))
assert np.isclose(scenario['baseline_energy_kwh'],scenario['shifted_energy_kwh'])
assert scenario['emissions_reduction_kg']==0
print('ASSUMPTIONS:');print(chr(10).join(scenario['assumptions']))
````

### Code cell 12

````python
from app import app
with app.test_client() as client:
    response=client.post('/predict',json={'dataset':RUN,'hours':24})
    assert response.status_code==200
    result=response.get_json()
    print('Real route response:',result['model'],result['cutoff'],len(result['predictions']),'hours')
    display(pd.DataFrame(result['predictions']).head())
    invalid=client.post('/predict',json={'hour':14,'day':2})
    print('Insufficient context correctly rejected:',invalid.status_code)
````

### Code cell 13

````python
summary=[]
for run_id in ['synthetic','uci']:
    metric_file=path('outputs',run_id,'metrics.json')
    if metric_file.exists():
        m=load_json(metric_file)
        summary.append({'source':run_id,'selected':m['selected_model_label'],
                        'day_ahead_rmse_kw':m['test_day_ahead']['rmse_kw'],
                        'day_ahead_wape_pct':m['test_day_ahead']['wape_pct'],
                        'one_step_r2':m['test_one_step']['r2']})
display(pd.DataFrame(summary).round(4))
````
