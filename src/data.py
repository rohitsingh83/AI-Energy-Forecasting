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
