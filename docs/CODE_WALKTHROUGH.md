# Code walkthrough — beginner reading path

Read this with the real files or the copy-paste [FULL_SOURCE.md](FULL_SOURCE.md).
The reusable implementation is in `src`; the notebook imports it so there is one
source of truth. Start with `main.py`, follow one phase at a time and inspect its
saved artifacts before advancing.

## 1. `main.py` — your control panel

**Purpose:** make `python main.py` run the complete experiment. `argparse` lets you
choose a phase, dataset, separate run ID and forecast horizon without editing code.

**Important functions:**

- `parse_args`: validates command, source, safe run ID and 1–168-hour horizon.
- `main`: prepares data, trains/evaluates, forecasts, simulates, plots and logs.

Default source is synthetic. `--dataset uci` uses the measured-household adaptation.
`--dataset csv --csv PATH --run-id building` uses a permitted custom file. A distinct
`--run-id smoke --quick --no-mlp` preserves baseline artifacts while learning faster.

`evaluate` reads recorded test evidence. It deliberately does **not** load the final
all-history model and grade it on the same test observations it has now seen.
`train` already includes proper selection, test reporting and final refit.

**Try:** run `python main.py --help`, then only `prepare`. Inspect what changed.

## 2. `generate_data.py` — a transparent virtual building

**Purpose:** create a reproducible building demand series without corporate access.

- `make_dataset(n_days, seed)` creates an RNG **inside** the call, so repeated calls
  with the same arguments return identical data.
- A Pandas hourly `DatetimeIndex` supplies hour, weekday, month and day-of-year.
- Morning/evening activity, weekend reduction, temperature-driven HVAC, occupancy
  and noise combine into plausible positive hour-average kW.
- A cosine correctly puts the synthetic daily temperature maximum at 14:00.
- 0.5% of target readings are deliberately missing for cleaning practice.
- `generate` writes the CSV and `.metadata.json` explicitly labeled synthetic.

**Do not claim:** calibrated weather, a physics digital twin, a real client or real
smart-grid logs. Temperature/occupancy are not fed in as unknown future covariates.

**Try:** compare seed 42 and 43 in a separate file, not by silently replacing the
bundled proof dataset. Explain which patterns are imposed rather than discovered.

## 3. `src/config.py` — visible assumptions

**Purpose:** make modeling and scenario settings easy to inspect.

| Setting | Meaning |
|---|---|
| `SEED=42` | Estimator and default data reproducibility |
| `LAGS=(1,2,24,48,168)` | Short-term, daily and weekly demand memory |
| `MAX_LAG=168` | Minimum recent history for serving |
| `TRAIN_FRACTION=.70` / validation `.15` | Ordered eligible-row fractions |
| `BACKTEST_HORIZON=24` | Validated multistep task |
| `BACKTEST_ORIGINS=14` | Bounded evenly sampled eligible days per split |
| `INTERVAL_COVERAGE=.90` | Descriptive validation absolute-error quantile |
| `MAX_FORWARD_FILL_HOURS=3` | Past-only fill limit |
| `MAX_HORIZON=168` | Supported exploratory forecast length, not validated accuracy |

Illustrative rates, flexibility and carbon factor are constants here, not fetched
from a real utility. An API request can override the allowed numeric settings.

## 4. `src/utils.py` — reliable plumbing

**Purpose:** shared paths, metrics, strict JSON, hashes and real logs.

- `PROJECT_ROOT` is derived from the file location rather than whichever directory
  the terminal happened to start in.
- `path` / `ensure_dir` create predictable project-relative artifacts.
- `json_safe` recursively replaces NaN/Infinity with `None`, serializes NumPy values,
  timestamps and paths; `save_json(..., allow_nan=False)` enforces actual JSON.
- `sha256_file` hashes input bytes in chunks for provenance, not secrecy.
- `compute_metrics` masks unobserved truth, reports sample count, MAE/RMSE/R²/MAPE/WAPE.
  RMSE uses `sqrt(mean_squared_error(...))` to avoid deprecated sklearn signatures.
- `configure_logging` sends genuine phase/model messages to console and a per-run
  file; appending preserves evidence when separate phases are run later.

**Try:** recompute day-ahead RMSE from its CSV. Verify a modified raw file has a
new SHA256. Do not imply a hash verifies the truth or legal ownership of the data.

## 5. `src/data.py` — source and causal cleaning

### Public downloader

`download_uci` uses one fixed official HTTPS archive URL. It streams at most the
expected small archive size into `.cache`, reads only Date/Time/Global_active_power
in chunks and does not extract arbitrary ZIP paths.

For each chunk it computes hourly sums and valid counts. Because a chunk can split
an hour, it combines **sums and counts across chunks before dividing**. Taking the
mean of chunk means would incorrectly weight partial hours.

At least 54 valid source minutes are required for an observed hourly target.
Remaining hours are NaN. Source/coverage/license/hash metadata are written with the
adapted hourly file. Reusing this bundled file needs no dataset network access.

### Generic input cleaning

`clean_data`:

1. requires timestamp and average-power columns;
2. rejects invalid clocks or timezone-aware custom inputs requiring preprocessing;
3. sorts records, converts numerics, masks negative/infinite power;
4. resamples by hourly **mean**, not sum of power readings;
5. trims unobserved leading/trailing hours;
6. records `target_observed` before filling;
7. fills from past readings only for at most three hours;
8. preserves long gaps and computes hourly kWh as kW × 1 h;
9. returns both data and quality counts.

Duplicate/sub-hourly rows are averaged under an equal-duration observation
assumption. Irregular-duration readings require duration-weighted preparation first.
The UCI adapter handles its regular minute data separately with coverage counts.

`prepare` stores processed data in a separate folder, source/quality JSON and a
24-row preview. `load_processed` restores the timestamp index and Boolean mask.

**Try:** hand-check a four-hour missing block. The fourth missing hour remains NaN.
If a future actual value changes, earlier filled values must not change.

## 6. `src/features.py` — the forecasting contract

`FEATURE_NAMES` is an explicit ordered list shared by training and serving.

### Calendar

For hour h, use sine/cosine of `2πh/24`; equivalent cycles encode weekday and annual
phase. Hour 23 is then close to hour 0 rather than numerically far away. Weekend
is an additional known-in-advance binary flag.

### Lag/rolling features

`energy.shift(24)` at time t means demand one day earlier. All rolling statistics
start from `energy.shift(1)`, so the current target does not sneak into its own inputs.
Only rows with complete features, finite target and `target_observed=True` are kept.

### Training–serving parity

`next_feature_row(timestamp, history)` builds the same features for the next hour
using the last 168 values. Tests compare it to many historical training rows to
numerical tolerance. `validate_history` rejects insufficient/gappy recent history.

**Try:** change all targets at t and later, rebuild features, and verify features
at t remain identical. That is a powerful concrete leakage test.

## 7. `src/models.py` — candidates and train-only preprocessing

Baselines require no estimator fit:

- `previous_hour`: predict last observed value; recursive output stays constant;
- `same_hour_yesterday`: repeat the prior-day pattern;
- `same_hour_last_week`: repeat the prior-week pattern.

ML candidates:

- **Ridge:** `StandardScaler` + regularized linear regression. Clear, fast reference.
- **Random forest:** averages nonlinear decision trees, bounded depth/leaf size.
  It uses parallel fitting but one worker for one-row recursive inference to avoid
  repeated thread-setup overhead.
- **MLP:** a small two-hidden-layer feedforward neural regressor. X is standardized
  in a pipeline. `TransformedTargetRegressor` also standardizes y using fit data.
  `early_stopping=False` avoids a randomly held-out internal validation set;
  `shuffle=False` keeps the training order. The explicit time split selects models.

`fit_candidate` records convergence warnings rather than suppressing them. A
max-iteration warning means training stopped at its configured limit, not that the
best parameters are proven optimal. UCI's train+validation MLP emitted one in this
build; it was not the selected model.

`predict_rows` is for **one-step** X rows with actual historical lag observations.
Its behavior is intentionally different from an entire recursive future window.

## 8. `src/evaluation.py` — strongest proof in the project

### Ordered split

`chronological_split` slices aligned eligible X/y by position without shuffling.
The exact date ranges and counts are saved. Splits are percentages of **eligible
supervised rows**, not necessarily equal percentages of raw calendar duration.

### Origins

`select_origins` finds daily issuance times within a split whose prior week is
complete and whose next 24 hours remain in the split, with at least 18 observed
target hours. It evenly samples up to 14 valid origins to keep CPU work bounded.
An origin is the first predicted hour; history ends one hour before it.

### Recursive validation

`rolling_backtest` slices history through origin−1h, generates 24 predictions, then
joins observed future truth solely for scoring. It cannot use future truth to
construct any intermediate lag in that issued forecast.

Every candidate is fit on train, scored on validation and compared by day-ahead
RMSE. `selection.json` and a log line lock the champion before the test loop starts.

### Test and deployment refits

1. Refit on train+validation, evaluate untouched test one-step and day-ahead.
2. Save every candidate comparison, selected backtests, per-horizon metrics and
   RF feature importance (clearly a diagnostic, not causal importance).
3. Compute the selected model's validation 90th absolute-error percentile and
   descriptive empirical test coverage.
4. Save test metrics and source/split details.
5. Only now refit the selected candidate on **all historical eligible data** and
   save a future-serving bundle plus environment metadata.

The final serving fit has seen the old test period. Therefore old test scores are
explicitly the earlier train+validation evaluation, not an in-sample final-fit score.

**Try:** inspect UCI: MLP is better than RF on the sampled test day-ahead windows,
but RF remains champion because it won validation. Explain why changing it now
would make the test part of model selection.

## 9. `src/forecast.py` — genuine future sequence

`recursive_forecast` validates horizon and history, repeatedly builds the next feature
row, predicts, clips nonnegative and appends **its own prediction** to working history.
Only the last 168 values are retained because no feature needs older values.

`forecast_bundle` adds a descriptive band for h≤24 from validation errors. For h>24,
lower/upper are NaN internally and null in JSON. The model may forecast up to 168 h,
but we make no unmeasured seven-day accuracy/coverage claim.

`save_forecast` writes the selected primary horizon and a separate full exploratory
168-hour file plus explicit cutoff/energy/peak/source metadata.

`load_bundle` only loads trusted local joblib. Do not accept user-uploaded pickles.
The bundle contains estimator, feature order, source, input hash, 168-hour history,
cutoff and validation radius, so inference does not require refitting.

## 10. `src/simulation.py` — a transparent business experiment

`validate_scenario` accepts only documented finite numeric settings. Ordered rates,
nonnegative carbon factor and 0–50% flexibility are enforced.

`simulate` takes the first 24 predicted hours:

```text
eligible = sum(peak-hour energy × flexible fraction)
capacity = sum(max(original global peak - off-peak demand, 0))
moved = min(eligible, capacity)
```

Subtract moved energy from peak hours proportionally and distribute to off-peak
hours according to headroom. Compute rate×hourly-kWh costs. Assert total energy is
conserved, demand stays nonnegative and global peak never increases.

A constant carbon factor times unchanged total kWh yields unchanged modeled carbon.
The output records **zero** emissions reduction and all assumptions, including
pre-scheduling permission, absent equipment limits and hypothetical savings.
This is not an industrial optimizer or a measured causal intervention.

## 11. `src/visualization.py` — graphs from real evidence

Matplotlib uses the headless `Agg` backend so a server can generate files without
a desktop display. Seaborn adds consistent theme styling. A local helper writes
clean PNGs and closes each figure.

Nine plots per run cover history/splits, routines, missingness, model comparison,
one-step test, future forecast, residual/lead-time diagnostics, RF importance and
scenario outcomes. Every plot's title/footnote indicates scope. No generative-image
model was used to fabricate charts or screenshots.

## 12. `app.py` — validated serving, not model training per request

`create_app` allows Flask test clients and a normal running server. Its endpoint
handlers select only trained safe run IDs and validated horizons/settings.
Joblib bundles/forecasts are cached by file modification time. Small CSV/JSON
artifacts are downloadable through fixed allowlists; model pickles/private paths
are not download endpoints. Body size is bounded at 16 KiB. Debug is off.

`/predict` accepts `dataset` and `hours`, not the original arbitrary hour/day
payload. The API needs recent saved demand context to honor the feature contract.
The frontend calls same-origin relative URLs, so proxies/previews work.

No production auth/TLS/SCADA is implied. See API/security documentation.

## 13. `templates/dashboard.html` — the five-view interface

CSS, icons, SVG charts and JavaScript are inline: no CDN, remote images or fonts.
The sidebar opens Overview, Forecast explorer, Decision sandbox, Model evidence
and Project runbook. Dataset/horizon controls call relative JSON endpoints.

Charts use observed/predicted arrays, sensible unit labels, forecast cutoff, hover
readouts and an empirical band only where available. The sandbox calls `/api/simulate`.
A precomputed export switches to an embedded snapshot provider instead of failing
network requests in an offline viewer. That mode is labeled clearly and mirrors
the conserved-energy scenario logic locally; it is not a live Python inference call.

The UI is responsive; actual desktop/mobile captures are in `images/screenshots`.

## 14. `scripts/` — reproducible evidence helpers

- `export_dashboard.py`: queries the real local Flask test client/artifacts and renders
  a standalone HTML with embedded data for trained runs.
- `validate_project.py`: checks hashes, validation-selected champion, scores recomputed
  from predictions, future timestamps, band limits, conservation, API and plot files.
- `browser_check.py`: optional Playwright checks and real desktop/mobile screenshots;
  requires separate browser tooling and a running app, not part of normal runtime.
- `export_results.py`: builds the human-readable results report from actual saved
  metrics, comparisons, forecasts and scenario artifacts, without retraining.
- `export_source.py`: collects actual text implementation and README into one complete
  copy-paste document. Does not manufacture a second model implementation.
- `package_project.py`: creates the downloadable ZIP and per-file SHA256 manifest;
  includes trusted demo bundles but excludes caches, environments and credentials.

## 15. `tests/` and configuration

- `conftest.py`: creates a genuine small offline end-to-end experiment in pytest temp
  folders; it cannot overwrite bundled outputs/models.
- `test_data.py`: seeds, missingness, units, duplicates, causal prefix invariance,
  bad schemas/timestamps and source minute coverage.
- `test_forecasting.py`: feature parity/leakage, masks, ordered splits, scaling,
  baseline/recursive behavior, invalid horizons/history and backtest boundaries.
- `test_simulation.py`: energy, peak and cost invariants, settings and strict JSON.
- `test_api_and_pipeline.py`: actual pipeline artifacts, endpoints, bad payloads,
  downloads, limits and CLI validation.
- `pyproject.toml`: pytest discovers `tests`; future deprecations fail checks.
- `.github/workflows/ci.yml`: fresh CPU checks on supported Python versions after push.
- `requirements.txt`: tested direct runtime pins; `requirements-dev.txt`: tests/notebooks.
- `.gitignore`: avoids environments, credentials, cache and heavy derived binaries.

## 16. Study experiment to make the proof yours

1. Reproduce the existing run and do not change the original test report.
2. State a hypothesis, e.g. “weekly memory matters more for the building than house.”
3. Remove the weekly lag in a new branch/run and compare on validation only.
4. Record what changed, why, its validation errors and its limitations.
5. If choosing new models after seeing old test results, reserve a fresh future
   holdout before claiming a new unbiased test improvement.
6. Add one test that proves your change preserves the forecasting contract.
7. Commit your actual code/notes at the real time of work.

This is more valuable than adding a deep-learning library without evidence.
