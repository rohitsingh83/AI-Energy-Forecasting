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
