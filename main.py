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
