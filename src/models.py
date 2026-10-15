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
