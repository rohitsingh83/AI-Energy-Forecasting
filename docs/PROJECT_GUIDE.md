# AI-Powered Energy Consumption Forecasting System
## Complete student build guide — A through M

**Goal:** forecast hourly electricity demand for homes/buildings with datasets and
virtual simulation, then demonstrate a responsible, reproducible decision-support
workflow on GitHub. No corporate energy system, GPU or paid service is required.

This guide accompanies a **completed implementation**. Actual source files,
executed experiments, models, evidence and tests are in the project. Read
[START_HERE.md](../START_HERE.md) if you want to run it before studying the details.

### Navigation

- [A. Project explanation](#a-project-explanation)
- [B. Tech stack options](#b-tech-stack-options)
- [C. Selected approach](#c-selected-approach)
- [D. Architecture and implementation phases](#d-architecture-and-implementation-phases)
- [E. Folder structure](#e-folder-structure)
- [F. Installation and environment setup](#f-installation-and-environment-setup)
- [G. Complete code and file explanations](#g-complete-code-and-file-explanations)
- [H. Virtual simulation](#h-virtual-simulation)
- [I. Execution and expected outputs](#i-execution-and-expected-outputs)
- [J. GitHub publishing](#j-github-publishing)
- [K. Full README](#k-full-readme)
- [L. Seven-day commit and learning plan](#l-seven-day-commit-and-learning-plan)
- [M. Proof checklist](#m-proof-checklist)

---

# A. Project explanation

## A1. In simple language

Imagine that you manage a building. You can see what electricity it used yesterday,
but tomorrow's use is uncertain. People arrive at different times, appliances run,
and hot or cold weather changes heating/cooling demand. If you could estimate the
next 24 hours, you could plan flexible work and anticipate expensive peaks.

This project teaches a computer to recognize patterns in historical readings and
estimate future electricity use. It then tests a hypothetical response: move some
flexible work from expensive hours into cheaper hours without pretending the work
or energy disappeared.

**What you actually build:** a Python forecasting pipeline, evidence of how well it
predicts unseen historical periods, future prediction files, graphs, a small API,
an interactive dashboard and a transparent virtual decision sandbox.

**What you do not build:** a real power-grid controller, a live corporate connection,
a guaranteed bill-saving product or a system that independently prevents blackouts.

## A2. In technical language

This is a supervised, autoregressive time-series regression system. It converts a
univariate hourly demand series into lagged/rolling and calendar features, compares
seasonal-naive baselines with Ridge, random forest and a scaled MLP, selects a model
using chronological validation 24-hour recursive backtests, evaluates on untouched
test observations and finally refits that selected candidate for future inference.

A Flask interface serves forecasts using a trusted local bundle containing the
estimator, feature order, final 168-hour history, source metadata and cutoff.
Predictions feed an explicitly assumption-based, energy-conserving load-shifting
heuristic. Reproducible CSV/JSON/PNG outputs, tests and source hashes form proof.

For demand `P_t` measured as hour-average kW:

```text
X_t = known calendar at t
      + [P_(t-1), P_(t-2), P_(t-24), P_(t-48), P_(t-168)]
      + mean/std of P_(t-24)..P_(t-1)
      + mean of P_(t-168)..P_(t-1)

P_hat_t = model(X_t)
Energy_t (kWh) = P_hat_t (kW) × 1 hour
```

At step 2 of a 24-hour forecast, lag 1 includes step 1's **prediction**. Future
observed consumption is not available inside a forecast window.

## A3. What problem does this address?

| Problem | How forecasting can help | Important boundary |
|---|---|---|
| Uncertain demand | Provide an expected hourly demand profile | Reliability also needs capacity, reserves and network operations |
| Expensive peak-time activity | Identify candidate hours for flexible scheduling | Real savings require real tariffs and constraints |
| Inefficient manual analysis | Automate repeatable cleaning, estimates and reports | The source must remain reliable and fresh |
| Production/consumption mismatch | Supply planners can use demand estimates as one input | This project does not simulate generator dispatch |
| Renewable integration | Demand predictions support net-load planning | Solar/wind and storage need their own models |
| Sustainability planning | Make consumption patterns and assumptions visible | Forecasting alone does not establish an emissions reduction |

A forecast is a **planning input**, not the final optimization or physical action.

## A4. Industry use cases

### Smart cities

A city could aggregate representative facility/household demand, identify periods
of higher demand, and coordinate demand response. This single-household/building
lab demonstrates methods only. It cannot establish city-scale model performance.

### Electricity boards / utilities

Demand forecasts help with generation/procurement planning, reserve considerations
and operational awareness. A real utility additionally needs feeder/network data,
weather, holidays, reliability requirements and trained operators.

### Manufacturing plants

A plant can consider moving flexible charging, pumping or noncritical batch work
away from expensive periods. Production deadlines, safety, machine capacity and
labor shifts must constrain any actual schedule. They are not assumed solved here.

### Data centers

A data center can use forecasts as one input to capacity and cooling decisions.
Service-level and thermal limits are essential. The often-quoted DeepMind example
concerned **up to 40% cooling-energy reduction** at Google facilities, not a promise
that this student's demand regressor saves 40% of total electricity. See the
[claims audit](CLAIMS_AUDIT.md) for the official source and careful wording.

### Renewable energy companies

Demand forecasts help when combined with wind/solar forecasts to estimate net
load and consider storage timing. Our implementation forecasts **consumption only**;
it does not train a renewable-production model or battery optimizer.

## A5. Full workflow

1. **Collect historical observations:** use a public measured dataset and a seeded
   virtual building rather than pretending to have corporate logs.
2. **Understand schema and units:** timestamp + mean power kW; distinguish power
   from interval energy and cumulative registers.
3. **Clean causally:** sort, resample to hourly mean power, average duplicates,
   reject invalid values, fill only short gaps from past observations.
4. **Preserve observation status:** never grade the model against values you invented
   through filling. Long gaps remain missing.
5. **Engineer features:** cyclic calendar, lagged demand and past-only rolling stats.
6. **Separate time:** earlier train, middle validation, latest test. No random split.
7. **Compare candidates:** simple seasonal baselines before complex models.
8. **Select on validation:** lock the lowest day-ahead RMSE candidate before test.
9. **Evaluate properly:** separate one-step replay from fixed-origin 24-hour forecasts.
10. **Refit for genuine future use:** only after test reporting, use all history.
11. **Forecast:** recursive hourly predictions after the last observed timestamp.
12. **Visualize and simulate:** demand curves, residuals, model comparisons and a
    clearly labeled hypothetical load-shifting scenario.
13. **Build proof:** files, hashes, tests, notebook, screenshots and honest Git commits.

---

# B. Tech stack options

| | Option A — easiest | Option B — intermediate | Option C — advanced |
|---|---|---|---|
| Tools | Python, pandas, NumPy, matplotlib/seaborn, scikit-learn, joblib | A + statsmodels, Flask/Streamlit, richer experiment tracking | Python, PyTorch/TensorFlow, larger data pipeline, API/container/model registry |
| Models | Seasonal naive, linear/Ridge, random forest; optional small sklearn MLP | SARIMA/SARIMAX, gradient boosting, more systematic rolling tuning | LSTM/GRU/temporal CNN/transformer, probabilistic multihorizon methods |
| Features | Calendar + causal demand lags | Weather forecasts, holidays, exogenous lags with issuance times | Multisite sequences, embeddings, physical/schedule covariates |
| Difficulty | Beginner to early intermediate | Intermediate | Advanced |
| GPU | No | Usually no | Optional but helpful for large neural training; not inherently mandatory |
| Typical output | Repeatable forecast CSVs, error metrics and plots | Scenario dashboard, richer API and rolling evaluation | Larger scalable forecasting service and monitoring |
| Main risk | Accidental time leakage / poor units | Misaligned exogenous forecasts and test tuning | Complexity without better baseline-relative evidence |
| Best student fit | Quick execution and clear understanding | After mastering A | After acquiring good data and evaluation skills |

ARIMA is a useful classical alternative, but it introduces stationarity/seasonal
modeling choices and potentially slower tuning. LSTM is not automatically better
than a tree or seasonal baseline on a small building dataset. A model's name is
less important than a defensible comparison on the intended forecast horizon.

# C. Selected approach

## C1. Choose Option A, with a thin professional serving layer

The core is beginner-friendly CPU scikit-learn. We add Flask and an inline
HTML/SVG dashboard as packaging, not as a replacement for clear modeling.

Why this choice:

- runs on a normal student laptop;
- uses widely taught libraries;
- makes every transformation visible;
- includes strong baselines, a linear model, a tree ensemble and the requested MLP;
- avoids GPU/cloud infrastructure becoming the project;
- supports a one-command run, reproducible artifacts and interview-ready proof.

The delivered **MLP won the synthetic-building validation**. The **random forest
won the real-household validation**. We do not force a neural network to win for
marketing. Baselines are allowed to win on a new dataset.

## C2. Improvements over the original short example

| Original pattern | Correction in this project | Why |
|---|---|---|
| Random `train_test_split` | Chronological train/validation/test | Future behavior must not train a past forecast |
| Only hour + weekday | 15 cyclic, lag and rolling features | Demand has short-term and weekly memory |
| Unscaled MLP | Train-only X and y scaling | Neural optimization behaves better |
| Fill every missing value indefinitely | At most 3 past-only hours; keep observation mask | Avoid inventing long periods of truth |
| Use test to choose the best model | Choose on validation; report test later | Prevent optimistic selection bias |
| Single test scatter score | One-step and 24-hour recursive backtests | The deployed task is multistep forecasting |
| API accepts arbitrary hour/day | API requests horizon using saved lag history | A real autoregressive forecast needs recent demand |
| “Energy” without a unit | Explicit hourly mean kW and interval kWh | Avoid incorrect billing/aggregation math |
| “Real time” for old dataset | Historical replay with explicit cutoff | Do not misrepresent data freshness |
| Savings/emissions assumed | Assumption-based cost scenario; conserved energy; zero modeled CO₂ reduction | No fabricated business impact |

---

# D. Architecture and implementation phases

## D1. Text block diagram

```text
INPUTS
  UCI public minute power → coverage-aware hourly aggregation
  OR seeded virtual-building CSV
  OR your permitted hourly CSV
                         │
                         v
INGEST / CLEAN
  parse timestamps → sort → hourly mean → short causal fill
  preserve observed-target mask → audit + source SHA256
                         │
                         v
FEATURES
  known calendar + shifted lags + shifted rolling windows
                         │
                         v
ORDERED SUPERVISED ROWS
  70% train → 15% validation → 15% untouched test
                         │
            ┌────────────┴─────────────────────┐
            v                                  v
  seasonal naive baselines           Ridge / random forest / MLP
            └────────────┬─────────────────────┘
                         v
VALIDATION 24-HOUR FORECASTS → LOCK CHAMPION
                         │
                         v
TRAIN+VALIDATION REFIT → ONE-STEP + DAY-AHEAD TEST REPORT
                         │
                         v
FINAL ALL-HISTORY REFIT → TRUSTED MODEL BUNDLE + 168-HOUR HISTORY
                         │
                         v
RECURSIVE FORECAST → CSV / JSON / PNG
                         │
            ┌────────────┴───────────────────┐
            v                                v
  virtual load-shifting sandbox       Flask forecast API
            └────────────┬───────────────────┘
                         v
             dashboard / offline export / GitHub evidence
```

The vector diagram is [images/architecture.svg](../images/architecture.svg).

## D2. Module responsibilities and data flow

| Module | Receives | Produces |
|---|---|---|
| `generate_data.py` | Seed + days | Fictitious hourly CSV + metadata |
| `src/data.py` | Public archive, synthetic CSV or custom CSV | Cleaned hourly frame, observation mask, provenance report |
| `src/features.py` | Hourly history | Aligned X/y and identical one-row inference features |
| `src/models.py` | Allowed training X/y | Baseline specs or fitted sklearn pipelines |
| `src/evaluation.py` | Features + ordered dates | Validation selection, held-out reports, final bundle |
| `src/forecast.py` | Trusted bundle + horizon | Recursive forecast and descriptive first-24-hour band |
| `src/simulation.py` | First 24 forecast hours + assumptions | Energy-conserving hypothetical schedule and costs |
| `src/visualization.py` | Actual saved data/metrics | Nine evidence plots per run |
| `main.py` | CLI command/arguments | Orchestration, logs and artifacts |
| `app.py` + template | Saved models/artifacts + validated HTTP requests | JSON responses and interactive dashboard |
| `tests/`, `scripts/` | Code and saved artifacts | Quality checks, actual screenshots and offline export |

Raw input is never replaced by cleaned data. Every run has its own output and model
folders. The selected final model is not the same fit used to produce held-out test
scores: that distinction is recorded explicitly.

## D3. Phase 1 — Setup

**Do:** install supported Python, create a virtual environment, install dependencies,
open the project in VS Code or another editor, inspect `main.py --help`.

**Why:** isolate this project's packages and make execution repeatable.

**Expected output:** an active `.venv` and importable runtime packages.

**Common mistakes:** Python 3.9/3.10 with newer pinned packages; using the wrong pip;
running from an unrelated folder; committing `.venv` or credentials.

**Verify:**

```bash
python --version
python -c "import pandas, numpy, sklearn, flask; print('Runtime imports OK')"
python main.py --help
```

## D4. Phase 2 — Dataset loading

**Do:** begin with bundled synthetic data; also run the public measured UCI mode.
Inspect the first rows, source metadata, date range, units and missingness.

**Why:** the target's meaning and provenance determine what conclusions are valid.

**Expected output:** raw hourly CSVs with metadata; UCI source adaptation is attributed.

**Common mistakes:** calling synthetic rows real smart-grid logs; claiming a single
house represents a city; confusing minute kW with cumulative kWh.

**Verify:**

```bash
python main.py prepare --dataset synthetic
python main.py prepare --dataset uci
```

Read `outputs/<run>/dataset_preview.csv` and `data_quality.json`. Raw row counts are
17,520 synthetic and 34,589 UCI hourly; cleaned UCI loses two partial boundary hours.

## D5. Phase 3 — Data cleaning

**Do:** sort time, validate schema, reject negative/infinite power, average duplicates,
resample hourly mean, fill up to three hours from the past, keep observation status.

**Why:** ML needs consistent, meaningful historical readings. The model must not learn
from future interpolation or get rewarded for predicting imputed labels.

**Expected output:** `data/processed/<run>/hourly_energy.csv` + data-quality audit.

**Common mistakes:** random row ordering; `bfill`; indefinite `ffill`; treating missing
readings as zero; summing power; ignoring long gaps or invalid timestamps.

**Verify:** sorted unique hourly index; no negative target; audit's before/after gap
counts; `target_observed=False` for filled values. Run `tests/test_data.py`.

Bundled UCI: 444 internal missing hours, 33 filled, 411 still missing. Keeping those
long gaps visible is a correct behavior, not an unfinished cleaning step.

## D6. Phase 4 — Feature engineering

**Do:** derive sine/cosine calendar features, weekend, lags 1/2/24/48/168, and past-only
24/168-hour statistics. Use `shift(1)` before rolling.

**Why:** demand depends on recent load and routines; cyclic features preserve the
closeness of hour 23 and hour 0.

**Expected output:** 15 feature columns and an aligned observed target, with warmup
and invalid-gap rows dropped. Save `feature_preview.csv`.

**Common mistakes:** a rolling mean containing the current target; using actual future
weather; a lag computed after random shuffling; different feature order at inference.

**Verify:** `next_feature_row(t, history)` numerically equals the training feature row
at `t`; tests prove future target changes do not change current features.

## D7. Phase 5 — Model building

**Do:** compare three baselines, Ridge, random forest and MLP. Fit scalers on allowed
training data only. Perform chronological validation day-ahead backtests and lock
the champion before touching test results.

**Why:** complex ML should justify itself against simple routines. Validation selects;
test estimates generalization. A randomly selected internal MLP validation holdout
is disabled; early stopping uses training convergence, not shuffled future periods.

**Expected output:** a selected model, validation predictions, split report, logs and
later a reusable final joblib bundle.

**Common mistakes:** choosing model by highest test R²; no baselines; unscaled MLP;
confusing a convergence warning with a fully successful optimization guarantee.

**Verify:** `selection.json` matches the minimum **validation** day-ahead RMSE in
`model_comparison.csv`. Read real logs for warnings. `train` includes phases 5–6:

```bash
python main.py train --dataset synthetic
```

## D8. Phase 6 — Evaluation

**Do:** refit candidates on train+validation; report one-step and recursive 24-hour
scores on untouched test data. Use up to 14 evenly sampled eligible daily origins.
Exclude unobserved/imputed target hours. Save residual and horizon diagnostics.

**Why:** a good next-hour replay is not necessarily a good day-ahead forecast.

**Expected output:** MAE, RMSE, R², MAPE and WAPE in JSON/CSV; actual/predicted files;
one-step and multistep results are separately named.

**Common mistakes:** calling R² “accuracy”; comparing absolute kW errors across different
load scales; scoring the final all-data refit on its own training test period;
pretending 14 origins cover every day or all seven-day horizons.

**Verify:** rerun metrics from saved predictions with `scripts/validate_project.py`.
Read the report without retraining:

```bash
python main.py evaluate --dataset synthetic
```

The code then refits the already-selected model on all history **only for future
use**. Reported test scores stay those obtained before that final refit.

## D9. Phase 7 — Forecasting

**Do:** load a trusted bundle, require 168 complete recent hours, predict after cutoff,
append each prediction to working history and repeat. Export the chosen horizon
plus an exploratory 168-hour CSV.

**Why:** saved models should be reusable without retraining and must not peek at future
meter readings during a multistep request.

**Expected output:** `forecast.csv`, `forecast_168h.csv`, `forecast_meta.json`.

**Common mistakes:** supplying only `hour/day` to a model trained on lags; forecasting
from today's date when the last data are old; using actual test values for every
future recursive step; claiming 168-hour accuracy from 24-hour evaluation.

**Verify:** first forecast timestamp = cutoff + 1 hour; all predictions finite and
nonnegative; error band stops after hour 24; fake future truth cannot alter an
issued forecast. Example:

```bash
python main.py forecast --dataset synthetic --horizon 48
```

## D10. Phase 8 — Visualization and virtual decisions

**Do:** generate historical, routine, missingness, comparison, held-out prediction,
future forecast, residual, feature-importance and scenario figures. Open the web
lab. Try 0%, 10%, 20% flexible peak load and explain the assumptions.

**Why:** a recruiter or stakeholder should see both results and limitations, not just
an estimator's name or an unsupported “savings” number.

**Expected output:** nine PNG figures per dataset; actual dashboard captures; simulation
CSV/JSON with conserved energy and zero modeled carbon benefit under constant intensity.

**Common mistakes:** graphing random test rows as a chronological curve; hiding a weak
baseline comparison; screenshotting invented metrics; labeling simulated savings
as real savings; assuming cheaper hours automatically mean lower carbon intensity.

**Verify:** every graph has a backing data file; costs equal hourly kWh × rates; total
shifted energy equals baseline; no shifted hour exceeds baseline maximum.

```bash
python main.py simulate --dataset synthetic
python main.py visualize --dataset synthetic
python app.py
```

## D11. Phase 9 — GitHub publishing

**Do:** inspect files, initialize your own repository, commit actual code/evidence,
create a public GitHub repo, add the real remote and push. Read Section J.

**Why:** a reproducible public work sample is stronger than an unsupported résumé line.

**Expected output:** your actual repo URL, readable README images and CI status.

**Common mistakes:** committing `.venv`, the raw 126.8 MB source, private meter data or
credentials; fake commit histories; broken relative images; claiming CI passed
before any remote run; uploading joblib binaries without a trusted-source policy.

**Verify:** clone your real repo into a new folder, install requirements, run the
synthetic pipeline and tests. Check README rendering and source attribution.
Remote publication is the one account-dependent action **not performed here**.

## D12. Phase 10 — Final output and demo

**Do:** deliver code, results, screenshots, notebook and a short honest demo. Practice
explaining why the model was selected and one important limitation.

**Why:** interviewers assess reasoning and reproducibility, not just a polished page.

**Expected output:** project archive, source repo, README, saved evidence, a 90-second
walkthrough and a defensible résumé bullet.

**Common mistakes:** claiming end-to-end personal authorship without understanding an
AI-assisted scaffold; claiming grid deployment; not knowing kW versus kWh or why a
random split is inappropriate.

**Verify:** another person can run the project; you can change an assumption and
explain the result; all résumé numbers match actual saved evidence. Use Section M
and the [demo script](DEMO_SCRIPT.md).

---

# E. Folder structure

```text
AI-Energy-Forecasting/
│
├── data/
│   ├── raw/
│   │   ├── energy_data.csv
│   │   ├── energy_data.metadata.json
│   │   ├── uci_household_hourly.csv
│   │   └── uci_household_hourly.metadata.json
│   └── processed/
│       ├── synthetic/hourly_energy.csv
│       └── uci/hourly_energy.csv
├── notebooks/01_energy_forecasting.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   ├── forecast.py
│   ├── simulation.py
│   └── visualization.py
├── models/
│   ├── synthetic/{forecast_bundle.joblib,metadata.json}
│   └── uci/{forecast_bundle.joblib,metadata.json}
├── outputs/
│   ├── synthetic/             # metrics, predictions, scenarios, logs
│   ├── uci/                   # same artifact contract, real household source
│   ├── dashboard.html         # standalone offline interactive export
│   ├── test-results.{txt,xml}
│   ├── verification.json
│   └── browser-checks.json
├── images/
│   ├── architecture.svg
│   ├── synthetic/             # nine plots
│   ├── uci/                   # nine plots
│   └── screenshots/           # actual desktop/mobile captures
├── templates/dashboard.html
├── tests/
├── scripts/
├── docs/                      # this guide, cards, full code and proof playbooks
├── .github/workflows/ci.yml
├── main.py
├── app.py
├── generate_data.py
├── README.md
├── START_HERE.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
├── LICENSE
├── DATA_LICENSE.md
├── SECURITY.md
└── CONTRIBUTING.md
```

- **data/raw:** preserves provenance and input rather than overwriting with cleaned rows.
- **data/processed:** derived aligned data. Included in the download, gitignored for regeneration.
- **notebooks:** executed beginner tutorial; reusable pipeline remains in `src`, not copied
  into an incompatible alternate implementation.
- **src:** one responsibility per module; feature definitions shared by training/inference.
- **models:** final trusted local bundles and environment metadata. Binary bundles are
  included for immediate demo in the archive but intentionally not Git-tracked.
- **outputs:** small, real machine-readable proof files and logs. Separate run IDs prevent
  mixing measured and synthetic results.
- **images:** analytical evidence and actual UI screenshots, with no fake repo preview.
- **docs:** domain explanation, usage, full source, license/data/model cards and proof plan.
- **tests:** isolated temporary data/models; no network or distributed bundle required.
- **scripts:** export/verify/capture tools, not hidden manual model manipulations.
- **templates:** no CDN dependency; browser code calls relative API URLs.
- **.github:** ready-to-run CI; actual remote success can only exist after publication.

---

# F. Installation and environment setup

## F1. Requirements

- Python **3.11–3.13**, 64-bit; Python 3.11 or 3.12 is a good beginner choice.
- CPU laptop; no GPU. A few GB of free memory/disk is sufficient for this small lab.
- Git for publishing; an editor and a browser. Jupyter is optional.
- Internet once for pip packages; UCI re-download only if you request `--refresh-data`.

Build evidence was executed on Python **3.13.16**. Runtime direct dependencies:

```text
numpy==2.3.5
pandas==2.2.3
scipy==1.17.1
scikit-learn==1.6.1
matplotlib==3.10.9
seaborn==0.13.2
joblib==1.5.3
Flask==3.1.3
requests==2.33.0
```

This is the actual `requirements.txt`, not a list of unrelated advanced libraries.
Development extras add pytest, notebook tooling and JupyterLab. The direct packages
are pinned; this is not a complete hash lock of every transitive dependency.

## F2. Windows PowerShell

1. Install supported 64-bit Python; enable the launcher/PATH option if available.
2. Extract the project; open PowerShell in `AI-Energy-Forecasting`.
3. Check versions: `py --list` and `git --version`.
4. Create and activate an isolated environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --dataset synthetic
python app.py
```

Use `py -3.12` or `py -3.13` when appropriate. If script activation is blocked, the
simplest alternative is not to change system policy: run the environment directly.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe app.py
```

In CMD, activation is `.venv\Scripts\activate.bat`. To stop the server press Ctrl+C.

## F3. macOS / Linux

```bash
cd AI-Energy-Forecasting
python3 --version
# Ensure this is 3.11–3.13 before continuing.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --dataset synthetic
python app.py
```

On Linux, if `venv` support is missing, install your distribution's Python venv
package for the chosen interpreter. Do not use system-wide `sudo pip` as a workaround.

## F4. Notebook environment

```bash
python -m pip install -r requirements-dev.txt
jupyter lab
```

Open `notebooks/01_energy_forecasting.ipynb`. The notebook locates the project root
and imports the same pipeline. It explicitly says whether it is reading bundled
results or running the experiment because results are absent.

## F5. Google Colab alternative

A GPU runtime is unnecessary. Upload the provided ZIP into `/content`, then:

```python
from zipfile import ZipFile
with ZipFile('/content/AI-Energy-Forecasting.zip') as archive:
    archive.extractall('/content')
```

```python
%cd /content/AI-Energy-Forecasting
%pip install -r requirements.txt
```

Restart the runtime if Colab asks after replacing preinstalled library versions,
then change back to the project folder and run:

```python
!python main.py --dataset synthetic
from IPython.display import Image, display
import pandas as pd
print(pd.read_csv('outputs/synthetic/model_comparison.csv'))
display(Image('images/synthetic/forecast.png'))
```

Download the resulting files when finished. Colab's VM `127.0.0.1` is **not** your
laptop's loopback; run the Flask UI locally or use the standalone HTML rather than
promising a normal local browser can open Colab's loopback server.

## F6. Frequent setup problems

| Symptom | Likely cause | Fix |
|---|---|---|
| Package has no compatible wheel | Unsupported/old Python or architecture | Use supported 64-bit Python 3.11–3.13 |
| `ModuleNotFoundError` | Wrong interpreter or incomplete installation | Activate `.venv`; use `python -m pip` |
| `main.py` not found | Wrong working folder | `cd` to extracted project root |
| Models absent after Git clone | Bundles are intentionally gitignored | Run `python main.py` first |
| Port 5000 busy | Another service uses it | Choose `PORT=5001` on Unix or `$env:PORT='5001'` in PowerShell |
| UCI download unavailable | Network/site problem | Use bundled hourly CSV or default synthetic mode |
| Long recent data gap | Insufficient forecast lag history | Repair source or choose a trustworthy cutoff; do not silently fill weeks |
| MLP convergence warning | Max iterations reached | Report it; tune only with validation, not by test score |

---

# G. Complete code and file explanations

## G1. Where the full code is

**Every implementation file is present in the repository, with full copy-paste
source also collected in [FULL_SOURCE.md](FULL_SOURCE.md).** No pseudocode, missing
corporate database or unavailable `energy.csv` is required. That compendium includes
runtime Python, the dashboard, scripts, tests, configuration and CI content.

Read [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) beside the actual files. Key files:

| File | What to understand first |
|---|---|
| [`generate_data.py`](../generate_data.py) | Seeded simulation, daily/seasonal load and honest metadata |
| [`src/config.py`](../src/config.py) | 168-hour history, split fractions, horizons and assumptions |
| [`src/utils.py`](../src/utils.py) | Project-relative paths, strict JSON, hashes, metrics and logs |
| [`src/data.py`](../src/data.py) | Real downloader, source aggregation, causal cleaner and masks |
| [`src/features.py`](../src/features.py) | Vectorized training features and identical inference row |
| [`src/models.py`](../src/models.py) | Naive baselines and sklearn estimators/scalers |
| [`src/evaluation.py`](../src/evaluation.py) | Ordered splits, rolling origins, validation lock and final refit |
| [`src/forecast.py`](../src/forecast.py) | Recursive forecasts and honest error-band limits |
| [`src/simulation.py`](../src/simulation.py) | Conserved energy, rate assumptions and headroom allocation |
| [`src/visualization.py`](../src/visualization.py) | Nine graphs generated from actual saved evidence |
| [`main.py`](../main.py) | Orchestrates the phases or one full run |
| [`app.py`](../app.py) | Validated JSON API and safe artifact allowlist |
| [`templates/dashboard.html`](../templates/dashboard.html) | Responsive UI, SVG charts and offline snapshot behavior |
| [`tests/`](../tests) | Leakage/unit/API/simulation/end-to-end verification |
| [`scripts/`](../scripts) | Actual artifact checks, export and optional browser capture |

## G2. The critical anti-leakage code

```python
past = energy.shift(1)
features['rolling_mean_24'] = past.rolling(24, min_periods=24).mean()
features['rolling_std_24'] = past.rolling(24, min_periods=24).std(ddof=1)
features['rolling_mean_168'] = past.rolling(168, min_periods=168).mean()
```

Without `shift(1)`, a rolling statistic would contain the value being predicted.
Building past-only features before a chronological split is legitimate here: a
first test-hour forecast is allowed to use historical train/validation readings.
Learned scalers, estimators and validation choices are still fit only on their
permitted time segments.

## G3. Why two evaluation modes exist

```text
One-step replay:
  predict t → receive observed t → predict t+1 → receive observed t+1

One fixed-origin 24-hour forecast:
  know history through cutoff
  predict +1 → reuse prediction → predict +2 → ... → predict +24
  only afterwards compare against observed future hours
```

A random 80/20 split with a single generic score cannot prove the second task.
The stored prediction files and unit tests make the distinction inspectable.

## G4. Metric formulas

```text
MAE  = mean(abs(actual - predicted))                         [kW]
RMSE = sqrt(mean((actual - predicted)^2))                    [kW]
R²   = 1 - sum((actual-predicted)^2)/sum((actual-mean)^2)      [dimensionless]
MAPE = 100 * mean(abs(error/actual))                         [%; unstable near 0]
WAPE = 100 * sum(abs(error))/sum(abs(actual))                [%]
```

Use observed targets only. R² may be negative and is not classification accuracy.
The implementation uses sklearn metrics with explicit finite-value masks.

## G5. Tests as executable explanation

Run:

```bash
python -m pytest -q
```

The delivered run passed **67 tests**, covering deterministic generation, units,
causal filling, schema failures, UCI coverage, no future target leakage, feature
parity, chronological splits, train-only scaling, recursive predictions, valid
horizons, long gaps, energy/cost/peak constraints, strict JSON, API validation and
an isolated offline end-to-end pipeline.

The tests create temporary fixtures. Those fixtures are not passed off as measured
portfolio results. Optional browser checks are a separate tool, not hidden inside
the 67-test count. See [PROOF_CHECKLIST.md](PROOF_CHECKLIST.md).

---

# H. Virtual simulation

## H1. How the dataset represents real-world use

A row represents average power during one hour. Repeated morning/evening routines,
weekends, HVAC effects and noise give the synthetic building plausible structure.
The real UCI mode demonstrates how less predictable measured household data can
reduce performance. Both are useful, provided their provenance stays clear.

The synthetic generator is not a certified digital twin. The real dataset is not
a live feed. Neither establishes the effectiveness of a city-wide system.

## H2. Step-by-step virtual business scenario

### Step 1 — Choose a site and a cutoff

Select `synthetic` or `uci`. Record the source title and last observation. A
forecast made from 2010 data belongs to 2010's next hours, not today's energy use.

**Save:** `data_quality.json`, metadata and input SHA256.

### Step 2 — Reproduce and inspect the held-out forecast evidence

Run the full pipeline and inspect `model_comparison.csv`, MAE/RMSE/R², the one-step
curve and the day-ahead errors. Do not assume model readiness from one attractive plot.

**Proof:** actual logs, split report, test predictions, metric JSON and comparison PNG.

### Step 3 — Generate the next 24 forecast hours

The model recursively predicts future mean power, then converts each one-hour
interval into kWh. Plot recent history and the future curve, marking the cutoff.

**Save:** `forecast.csv`, `forecast_meta.json`, `forecast.png`.

### Step 4 — Make assumptions explicit

The default hypothetical tariff is:

| Hour in dataset clock | Illustrative rate |
|---|---:|
| 00:00–05:59 | INR 5 / kWh |
| 17:00–21:59 | INR 12 / kWh |
| All other hours | INR 8 / kWh |

Assume 10% of peak-hour forecast energy is flexible. Allow that work to be moved
anywhere into off-peak hours inside the same 24-hour planning window, including
pre-scheduling. Do not claim these rates are a real Hyderabad or French tariff.

### Step 5 — Calculate the baseline

```text
baseline_energy = sum(hourly forecast kWh)
baseline_cost   = sum(hourly forecast kWh × illustrative rate)
```

This is an estimated cost for the model's demand curve, not an actual invoice.
Taxes, demand charges, fixed fees and billing minimums are absent.

### Step 6 — Shift only eligible flexible energy

1. Sum the flexible fraction of peak-hour energy.
2. Measure each off-peak hour's headroom below the original forecast maximum.
3. Move no more than the lesser of eligible energy and total available headroom.
4. Subtract moved energy proportionally from eligible peak hours.
5. Add it to off-peak hours in proportion to available headroom.
6. Verify nonnegative loads, equal total energy, and no hour above the original peak.

This is a transparent heuristic, not an industrial mixed-integer scheduler.
Comfort, device power, job deadlines, safety and rebound still need real constraints.

### Step 7 — Compare the scenario

```text
hypothetical_saving = baseline_cost - shifted_cost
saving_pct = 100 × hypothetical_saving / baseline_cost
```

Default synthetic example from this build:

- forecast energy: **2,543.53 kWh**;
- baseline cost: **₹20,991.89**;
- shifted cost: **₹20,553.34**;
- flexible energy moved: **62.65 kWh**;
- hypothetical saving: **₹438.55 (2.09%)**;
- peak: **131.96 → 130.67 kW**;
- energy before/after: exactly conserved apart from displayed rounding.

These are scenario results under assumptions, not achieved savings at any facility.

### Step 8 — Handle carbon honestly

The illustrative constant factor is 0.70 kg CO₂e/kWh, **not a verified current
regional emissions factor**. Because total energy stays constant, modeled carbon
stays constant too: **zero modeled reduction**. Real carbon-aware shifting would
need time-varying marginal intensity and a defensible counterfactual. Do not infer
that a cheaper tariff means cleaner power.

### Step 9 — Change a decision parameter

Open Decision sandbox; compare 0%, 10%, 20% flexible share or different valid rates.
Observe that zero flexibility means no cost benefit. Change off-peak rate up to
peak rate and explain why the economic incentive disappears. Keep the original
baseline results so you can distinguish your new scenario from the bundled one.

### Step 10 — Capture proof

Save the scenario table, assumptions and graph. Record which dataset, cutoff,
model, horizon and settings were used. Include an explicit sentence: “This is a
forecast-based virtual decision experiment; savings are not measured.”

## H3. Graphs and backing files

| Figure | Purpose | Backing data |
|---|---|---|
| `energy_history.png` | Seasonal history and split boundaries | Processed hourly data + split report |
| `usage_patterns.png` | Mean hourly/weekly routines | Observed historical targets |
| `data_quality.png` | Missing, filled and still-missing counts | `data_quality.json` |
| `model_comparison.png` | Validation vs test day-ahead RMSE | `model_comparison.csv` |
| `actual_vs_predicted.png` | Last eligible one-step test rows | `one_step_predictions.csv` |
| `forecast.png` | Last 48 h context + next 48 h forecast | Hourly data + `forecast_168h.csv` |
| `residual_diagnostics.png` | Day-ahead residuals and lead-time error | Backtest CSV + `horizon_metrics.csv` |
| `feature_importance.png` | RF diagnostic, not causal importance | `feature_importance.csv` |
| `simulation.png` | Baseline vs shifted load and costs | `simulation.csv` / `.json` |

Every dataset gets its own nine files in `images/<run>/`. RF importance is labeled
a diagnostic even when the champion is MLP; it is not attributed to the wrong model.

---

# I. Execution and expected outputs

## I1. Fastest complete run

From the project root after installing requirements:

```bash
python main.py
python main.py --dataset uci
python app.py
```

Open **http://127.0.0.1:5000**. The delivered archive already contains trusted model
bundles and processed data, so you can start with `python app.py` immediately.
A Git clone intentionally does not carry those derived artifacts; rebuild first.

## I2. Individual actions

| Goal | Command |
|---|---|
| Inspect CLI | `python main.py --help` |
| Recreate synthetic raw input | `python generate_data.py --days 730 --seed 42` |
| Load + clean | `python main.py prepare --dataset synthetic` |
| Train/select/test/final refit | `python main.py train --dataset synthetic` |
| Print recorded test metrics | `python main.py evaluate --dataset synthetic` |
| Forecast next 48 hours | `python main.py forecast --dataset synthetic --horizon 48` |
| Generate default scenario | `python main.py simulate --dataset synthetic` |
| Recreate PNG graphs | `python main.py visualize --dataset synthetic` |
| Export offline interactive HTML | `python scripts/export_dashboard.py` |
| Verify units/leakage/API code | `python -m pytest -q` |
| Verify stored evidence consistency | `python scripts/validate_project.py` |

The scenario always uses the **first 24 forecast hours**, including when an
exploratory forecast request is longer. A saved plot/export is a snapshot; recreate
it after changing model or input data rather than assuming it auto-updates.

## I3. Example actual terminal result

Synthetic experiment from this build:

```text
Selected: MLP neural network
Test day-ahead RMSE: 6.888 kW
Test one-step R²: 0.831
Open the dashboard: python app.py
```

First synthetic future predictions:

```text
2024-12-31 00:00   104.116 kW
2024-12-31 01:00   101.943 kW
2024-12-31 02:00   102.833 kW
2024-12-31 03:00   102.738 kW
2024-12-31 04:00   102.762 kW
```

UCI experiment:

```text
Selected: Random forest
Test day-ahead RMSE: 0.598 kW
Test one-step R²: 0.567
```

The UCI MLP train+validation fit emitted a genuine max-iteration convergence
warning. It was not suppressed or converted into a claimed optimization success.
The selected RF completed. Read `outputs/uci/logs/pipeline.log`.

Runtimes in this sandbox were roughly 37 s synthetic and 69 s UCI including its
first archive download. Your runtime depends on CPU, disk and package versions.
These are observations, not minimum hardware or performance guarantees.

## I4. API and browser testing

A normal browser address bar makes a GET request. It cannot directly submit the
original POST `/predict` JSON example. Use a browser-friendly GET:

```text
http://127.0.0.1:5000/api/forecast?dataset=synthetic&hours=24
```

Or POST with curl:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"dataset":"synthetic","hours":24}'
```

PowerShell:

```powershell
$body = @{dataset='synthetic'; hours=24} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:5000/predict' `
  -Method Post -ContentType 'application/json' -Body $body
```

Model API calls return historical cutoff, model identity, horizon and a prediction
list. Unsupported hours, dataset IDs or arbitrary `hour/day` payloads are rejected.
See [API.md](API.md) for all endpoints and error cases.

## I5. Dashboard views

1. **Overview:** forecast energy, peak, held-out day-ahead error, hypothetical saving,
   context/forecast curve, provenance and data-quality summary.
2. **Forecast explorer:** 24/48/168-hour controls, first-24-hour empirical band, hourly
   table and full CSV export.
3. **Decision sandbox:** flexible share and tariff controls with conserved energy.
4. **Model evidence:** full candidate comparison, one-step and day-ahead curves,
   lead-time error and evaluation audit.
5. **Project runbook:** commands, actual artifact downloads and source attribution.

`outputs/dashboard.html` embeds the real evidence and can be viewed offline without
Python, external scripts/fonts or network. It recalculates the scenario locally;
it does **not** pretend that an offline snapshot is making a live Python request.

## I6. Verification interpretation

Bundled verification is local, not remote GitHub evidence:

- 67 pytest tests passed.
- 29 artifact/provenance checks passed.
- Optional browser check report and actual captures are included.
- CI configuration is supplied; no remote run can be claimed until you publish.

An automated pass does not establish production readiness. It establishes that the
specified behaviors were checked within this project's scope.

---

# J. GitHub publishing

## J1. Professional repository identity

**Recommended name:** `AI-Energy-Forecasting`

**Description:**

> CPU-friendly energy demand forecasting with real UCI data, virtual-building
> simulation, leakage-safe backtests, a Flask dashboard and reproducible ML evidence.

**Suggested topics:**

```text
python machine-learning time-series energy-forecasting smart-buildings
climate-tech scikit-learn flask data-science mlp-regressor random-forest portfolio
```

These are topical labels, not claims that any named company uses your repository.

## J2. Create the repository

1. Sign in to your own GitHub account.
2. Choose New repository, use the name/description above, and select visibility.
3. For a first push of this already-complete folder, do not auto-add another README,
   license or `.gitignore`; they already exist here.
4. Copy **your actual remote URL**. Replace `YOUR_USERNAME` below; it is an example,
   not an already-created repository.

## J3. First honest import

```bash
cd AI-Energy-Forecasting
git init
git branch -M main
git config user.name 'Rohit Singh'
git config user.email 'rohitsingh8389788@gmail.com'
git status --short
git add .
git diff --cached --stat
git commit -m 'feat: complete AI energy forecasting platform and GitHub Pages showcase'
git remote add origin https://github.com/rohitsingh83/AI-Energy-Forecasting.git
git push -u origin main
```

Use GitHub's supported authentication flow; do not put a personal access token in
code, command examples, README or a committed remote URL. HTTPS authentication may
use a credential manager or a fine-grained token entered securely when prompted.

The folder arrived complete. One truthful import commit is better than faking seven
days of authored history. Use the learning plan below for actual future work.

## J4. What should be tracked?

**Track:** modular source, requirements, tests, CI YAML, documentation, attributed
small hourly input data, model metadata, real small output CSV/JSON/logs, graphs,
actual screenshots and the executed learning notebook.

**Regenerate / ignore:** virtual environments, cache ZIP, 126.8 MB raw source text,
derived processed CSVs and binary joblib files. The downloadable bundle can include
trusted binaries for convenience without bloating Git commit history.

**Never publish:** private meter/address/bill data, `.env`, keys or tokens. Add any
private custom input path to `.gitignore` or store it outside the repository.

## J5. Verify the published proof

- README image links should work relative to repository root.
- Dataset attribution and limitations should remain visible.
- Actions should run; claim a green build only after it actually succeeds remotely.
- Clone into a fresh folder; run requirements, `python main.py`, and pytest.
- Add your actual repo URL and actual demo/video URL to your résumé/LinkedIn.
- Pin the repo. Use a concise About description and correct topics.
- If using GitHub Pages for the offline HTML, explain that it is a **snapshot**,
  not a hosted Flask inference service. GitHub Pages cannot run Python/Flask.

See [GITHUB_PLAYBOOK.md](GITHUB_PLAYBOOK.md) for branch/commit/release strategies.
No repository has been created in your account by this build.

---

# K. Full README

The complete, copy-paste-ready GitHub README is already written at
**[README.md](../README.md)**. It includes:

- overview and problem statement;
- industry use cases with honest scope;
- implemented features and tech stack;
- dataset source, license, transformations and units;
- architecture diagram and repository layout;
- Windows/macOS/Linux installation;
- full and phase-by-phase usage;
- API example;
- actual model results and virtual-scenario outcomes;
- screenshots and graphs using relative paths;
- testing/reproducibility;
- learning outcomes, limitations, future work and licenses.

Use that file as the root README when publishing. Add your own name, actual remote
URL and demo link after they exist. Do not replace actual metrics with aspirational
numbers, remove dataset labels, invent CI badges or list employers as clients.

For the literal README content alongside all code in one document, see
[FULL_SOURCE.md](FULL_SOURCE.md). It is generated from the actual project files,
not a second implementation that could silently disagree with them.

---

# L. Seven-day commit and learning plan

This is a **forward plan**, not a claim that the supplied scaffold was built by the
student over seven historical days. If rebuilding from scratch, follow the phases.
If importing the completed project, commit it honestly once, then make real daily
learning/experiment improvements.

| Day | Focus and meaningful work | Suggested actual commit | Evidence to capture |
|---|---|---|---|
| 1 — Setup | Create environment, run smoke example, explain problem/units; write your learning notes | `docs: record setup and forecasting scope` | Python/import output, setup note, actual commit |
| 2 — Dataset | Inspect both sources and licenses; audit hourly UCI coverage; write a source summary in your words | `data: document source coverage and power units` | Dataset preview and attribution metadata |
| 3 — Preprocessing | Trace causal fill and lag features; add one edge-case test you understand | `test: cover a documented missing-data edge case` | Data-quality audit and passing new test |
| 4 — Model | Reproduce baselines + ML; perform a train/validation-only ablation, e.g. remove weekly lag | `experiment: compare validation lag ablation` | Config, validation comparison, untouched-test policy |
| 5 — Evaluation | Explain one-step vs recursive, inspect residuals; add more origins or new future holdout if changing selection | `analysis: document horizon errors and baseline comparison` | Saved predictions, metric recomputation, limitations |
| 6 — Visualization | Improve a chart/table or add a useful unit label; rehearse scenario explanation | `ui: improve forecast evidence and scenario labels` | Real before/after screenshot and settings |
| 7 — Upload | Final README, clean clone test, publish actual repo, record actual demo | `docs: add reproducibility evidence and demo links` | Real repo screenshot, URL, CI result if green |

Do not make empty or backdated commits to manufacture proof. Commit messages should
match the actual change. If you have only one day, publish a coherent verified
project in one day and be transparent about AI assistance and what you learned.

A strong student-specific extension is one documented controlled experiment, not
seven cosmetic README edits. For each experiment save a hypothesis, changed config,
validation result, baseline and conclusion. Do not repeatedly optimize on the same
test set; reserve a new future holdout if iterating model selection after seeing it.

---

# M. Proof checklist

## M1. Included evidence and where to find it

| Required proof | Included file / location | What it proves |
|---|---|---|
| Dataset preview | `outputs/<run>/dataset_preview.csv` | Schema and real stored sample values |
| Source and license | `data/raw/*.metadata.json`, `DATA_LICENSE.md` | Synthetic vs measured provenance and permissions |
| Preprocessing output | `data_quality.json`, processed hourly CSV | Explicit gap handling and observation masks |
| Training logs | `outputs/<run>/logs/pipeline.log` | Actual model fits, selection, warnings and cutoff |
| Feature preview | `outputs/<run>/feature_preview.csv` | Actual feature order and target alignment |
| Test error metrics | `metrics.json`, `model_comparison.csv` | Quantified held-out performance with baselines |
| Actual vs predicted | `one_step_predictions.csv`, `actual_vs_predicted.png` | Chronological one-step replay, clearly labeled |
| Day-ahead proof | `backtest_predictions.csv`, `horizon_metrics.csv` | Fixed-origin multistep evaluation |
| Future predictions | `forecast.csv`, `forecast_168h.csv`, `forecast.png` | Future timestamps after explicit cutoff |
| Business simulation | `simulation.csv`, `simulation.json`, `simulation.png` | Hypothetical decisions and conserved energy |
| Working web UI | `images/screenshots/*.png`, `outputs/dashboard.html` | Real desktop/mobile captures and interactive snapshot |
| Executed notebook | `notebooks/01_energy_forecasting.ipynb` | Inspectable learning path with actual outputs |
| Tests | `outputs/test-results.txt`, `.xml` | 67 actual local checks passed |
| Artifact checks | `outputs/verification.json` | Consistent hashes, metrics, forecast and scenario artifacts |
| Browser checks | `outputs/browser-checks.json` | Actual interaction/CSV/responsiveness checks |
| GitHub repo preview | **You capture after publishing** | Your real public URL, README and remote CI |

## M2. Screenshot file naming and storage

Included actual browser captures:

```text
images/screenshots/01_overview_synthetic.png
images/screenshots/02_decision_sandbox.png
images/screenshots/03_model_evidence.png
images/screenshots/04_overview_uci.png
images/screenshots/05_mobile.png
```

Suggested additional **real captures you can make**, not fabricated files:

```text
images/screenshots/06_dataset_preview.png
images/screenshots/07_preprocessing_terminal.png
images/screenshots/08_training_terminal.png
images/screenshots/09_tests_terminal.png
images/screenshots/10_github_repository.png
```

Use meaningful names, not `Screenshot (14).png`. Include the source/run/setting in
the caption when relevant. Never add an image of a repo, terminal success or graph
that has not actually occurred. CSV/JSON/logs are stronger backing evidence than a
cropped success screenshot alone.

README usage from repository root:

```markdown
![Day-ahead forecasting evidence](images/synthetic/model_comparison.png)
![Actual application dashboard](images/screenshots/01_overview_synthetic.png)
```

Optional repeatable browser capture:

```bash
python -m pip install playwright
python -m playwright install chromium
# Linux may additionally need: python -m playwright install-deps chromium
# Keep python app.py running in a separate terminal.
python scripts/browser_check.py --url http://127.0.0.1:5000
```

Playwright is optional screenshot tooling; it is not a runtime requirement for the
ML project or part of the 67 pytest test count.

## M3. Before an internship/placement application

- [ ] I reproduced a full run on my own machine.
- [ ] I can explain power vs energy, past-only features and chronological splits.
- [ ] I know how one-step and day-ahead scores differ.
- [ ] I can name a naive baseline and explain validation selection.
- [ ] I can explain the real household's relatively weak day-ahead performance.
- [ ] I distinguish generated data from public measurements and from corporate logs.
- [ ] My scenario savings are explicitly hypothetical, not achieved business savings.
- [ ] I do not claim carbon reduction from energy-conserving shifting at constant intensity.
- [ ] I understand the code and acknowledge AI assistance where appropriate.
- [ ] I added one meaningful, documented experiment or improvement.
- [ ] My actual repo is public/readable and cloneable; its README images render.
- [ ] All résumé metrics can be traced to saved evidence; no fake CI/commit/client claims.

## M4. Honest résumé bullet template

> Built a CPU-based hourly energy forecasting lab using public UCI household data
> and a seeded virtual building, comparing seasonal baselines with Ridge, random
> forest and MLP using chronological validation and 24-hour recursive backtests;
> packaged forecasts, scenario analysis, a Flask dashboard and automated tests.

After personally reproducing, optionally add the **correct task-specific** numbers:

> The validation-selected RF achieved 0.598 kW day-ahead RMSE on 14 sampled UCI test
> origins, 29.59% lower RMSE than the weekly seasonal baseline; scenario savings were
> explicitly simulated, not measured.

Do not translate that 29.59% **forecast error** improvement into 29.59% energy saved.

---

## Reference documents

[Data card](DATA_CARD.md) · [Model card](MODEL_CARD.md) · [Results](RESULTS.md) ·
[API](API.md) · [GitHub playbook](GITHUB_PLAYBOOK.md) · [Code walkthrough](CODE_WALKTHROUGH.md) ·
[Full source](FULL_SOURCE.md) · [Interview Q&A](INTERVIEW_QA.md) ·
[Demo script](DEMO_SCRIPT.md) · [Claims audit](CLAIMS_AUDIT.md) ·
[Requirements completion map](REQUIREMENTS_TRACEABILITY.md)
