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
    data_js = path("docs", "data.js")
    import json
    data_js.write_text("const SNAPSHOT = " + json.dumps(json_safe(snapshot)) + ";", encoding="utf-8")
    print(f"Exported GitHub Pages data bundle: {data_js} ({data_js.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
