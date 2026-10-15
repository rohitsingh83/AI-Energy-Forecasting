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
