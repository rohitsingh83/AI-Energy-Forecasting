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
