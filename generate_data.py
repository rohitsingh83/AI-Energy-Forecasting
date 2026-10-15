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
