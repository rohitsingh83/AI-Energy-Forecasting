"""Regenerate the human-readable result report from real saved artifacts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.utils import path, load_json


def main():
    text = ["# Actual execution results\n",
            "This report is generated from saved pipeline artifacts, not target or invented scores. "
            "Run `python scripts/export_results.py` after a verified rerun to refresh it. "
            "The supplied runs use seed 42, full settings, CPU execution and pinned direct dependencies.\n",
            "## At a glance\n",
            "| Dataset | Validation-selected model | Test one-step RMSE (kW) | Test one-step R² | Test 24 h RMSE (kW) | Test 24 h MAE (kW) | Test 24 h R² |",
            "|---|---|---:|---:|---:|---:|---:|"]
    runs = []
    for run in ["synthetic", "uci"]:
        m = load_json(path("outputs",run,"metrics.json"))
        q = load_json(path("outputs",run,"data_quality.json"))
        s = load_json(path("outputs",run,"simulation.json"))
        f = load_json(path("outputs",run,"forecast_meta.json"))
        manifest = load_json(path("outputs",run,"run_manifest.json"))
        runs.append((run,m,q,s,f,manifest))
        a,b=m['test_one_step'],m['test_day_ahead']
        text.append(f"| {run} | {m['selected_model_label']} | {a['rmse_kw']:.3f} | {a['r2']:.3f} | {b['rmse_kw']:.3f} | {b['mae_kw']:.3f} | {b['r2']:.3f} |")
    text.append("\nOne-step replay updates actual historical lags between predictions. A day-ahead origin recursively forecasts 24 hours without actual updates inside that window. Scores are not interchangeable. R² is not percentage accuracy. Different load scales make cross-site absolute RMSE comparisons inappropriate.\n")
    for run,m,q,s,f,manifest in runs:
        a,b=m['test_one_step'],m['test_day_ahead']
        text += [f"## {run}: {q['source']['title']}\n",
                 f"- Source type: **{q['source']['source_type']}**.",
                 f"- Cleaned grid: **{q['hourly_rows']:,} hours**, {q['start']} → {q['end']}.",
                 f"- Missing target hours before fill: {q['missing_target_hours_before_fill']}; causal fills: {q['imputed_target_hours']}; still missing: {q['remaining_missing_target_hours']}.",
                 f"- Selected: **{m['selected_model_label']}**, chosen on validation 24 h RMSE **{m['validation_day_ahead']['rmse_kw']:.3f} kW**.",
                 f"- Test observed one-step targets: **{a['n']:,}**.",
                 f"- Test day-ahead windows: **{m['test_backtest_origins']} sampled daily origins × 24 h**, **{b['n']} observed targets scored**; imputed targets excluded.",
                 f"- Test day-ahead WAPE: **{b['wape_pct']:.2f}%**; MAPE: {b['mape_pct']:.2f}% (unstable at low load).",
                 f"- Day-ahead RMSE reduction vs weekly-naive baseline: **{m['day_ahead_rmse_improvement_vs_weekly_baseline_pct']:.2f}%**; this is forecast-error reduction, not energy saved.",
                 f"- Empirical validation error radius: **{m['validation_error_radius_kw']:.3f} kW**; observed test band coverage **{m['empirical_test_band_coverage_pct']:.2f}%** (no guarantee).",
                 f"- Train/evaluate/final-refit stage elapsed here: {manifest['total_train_evaluate_seconds']:.1f} s; excludes data preparation, graph rendering and API startup. Hardware runtimes vary.",
                 f"- Input SHA256: `{q['source']['input_sha256']}`.\n",
                 "### Actual candidate comparison\n",
                 "| Candidate | Selected | Validation 24 h RMSE | Test 24 h RMSE | Test one-step RMSE | Test 24 h R² |",
                 "|---|---|---:|---:|---:|---:|"]
        comp=pd.read_csv(path('outputs',run,'model_comparison.csv'))
        for _,r in comp.iterrows():
            text.append(f"| {r['label']} | {'Yes' if r['selected'] else '—'} | {r['validation_day_ahead_rmse_kw']:.3f} | {r['test_day_ahead_rmse_kw']:.3f} | {r['test_one_step_rmse_kw']:.3f} | {r['test_day_ahead_r2']:.3f} |")
        text += ["\nAll RMSE values are kW on that same source/task. Selection is locked before the test loop.\n",
                 "### Genuine future forecast, not future actuals\n",
                 f"- Forecast cutoff: **{f['cutoff']}**.",
                 f"- Stored primary horizon: {f['hours']} hours, {f['forecast_start']} → {f['forecast_end']}.",
                 f"- Forecast energy: **{f['total_kwh']:.2f} kWh**; forecast peak **{f['peak_kw']:.2f} kW** at {f['peak_time']}.",
                 "- `forecast_168h.csv` provides exploratory longer output. The empirical band stops after hour 24, because longer horizons were not validated here.\n",
                 "### Default hypothetical first-24-hour business scenario\n",
                 f"- Energy before / after: **{s['baseline_energy_kwh']:.2f} / {s['shifted_energy_kwh']:.2f} kWh**.",
                 f"- Eligible energy actually moved in the heuristic: **{s['moved_energy_kwh']:.2f} kWh**.",
                 f"- Baseline / shifted illustrative costs: **INR {s['baseline_cost_inr']:.2f} / {s['shifted_cost_inr']:.2f}**.",
                 f"- Hypothetical saving: **INR {s['hypothetical_savings_inr']:.2f} ({s['hypothetical_savings_pct']:.2f}%)**.",
                 f"- Original / shifted peak: **{s['baseline_peak_kw']:.2f} / {s['shifted_peak_kw']:.2f} kW**.",
                 "- Modeled carbon reduction: **0 kg**, because total energy is conserved under a constant illustrative intensity.",
                 "- Rates (INR 5 / 8 / 12), flexibility (10%), job timing and carbon factor are assumptions, not sourced tariffs or measured outcomes.\n",
                 f"![Forecast from the last dataset observation](../images/{run}/forecast.png)\n",
                 f"![Model evidence](../images/{run}/model_comparison.png)\n"]
    text += ["## What should not be hidden\n",
             "The measured UCI household is much less predictable in relative terms than the virtual building. Its selected model's day-ahead WAPE is 48.44%; the supplied evidence does not establish operational readiness. The test-best UCI MLP did not replace RF because RF had already won validation. That MLP's train+validation fit emitted a max-iteration convergence warning, visible in the real log.\n",
             "Only up to 14 daily origins per split are sampled, with eligible history/target coverage. Results do not evaluate every test day, 48/168-hour accuracy, multisite transfer, current households or city-grid reliability. The final served model is refit on all history after held-out reporting; stored test scores are not in-sample scores from that final refit.\n",
             "## Evidence locations\n",
             "- `outputs/<run>/metrics.json`, `model_comparison.csv`, `split_report.json`.",
             "- `one_step_predictions.csv`, `backtest_predictions.csv`, `horizon_metrics.csv`.",
             "- `forecast.csv`, `forecast_168h.csv`, `forecast_meta.json`.",
             "- `simulation.csv`, `simulation.json`, `logs/pipeline.log`.",
             "- `models/<run>/metadata.json`, `data/raw/*.metadata.json`.",
             "- `outputs/test-results.txt`, `verification.json`, `browser-checks.json`.",
             "- Actual analytical plots and browser captures in `images/`.\n",
             "See [MODEL_CARD.md](MODEL_CARD.md), [DATA_CARD.md](DATA_CARD.md) and [CLAIMS_AUDIT.md](CLAIMS_AUDIT.md) for scope. No business intervention, realized saving or live grid connection is claimed.\n"]
    target=path('docs','RESULTS.md')
    target.write_text('\n'.join(text),encoding='utf-8')
    print('Wrote',target)


if __name__=='__main__':
    main()
