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
