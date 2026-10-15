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
