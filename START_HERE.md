# Start here — your completed Energy Forecasting project

This is a **working, tested project**, not just a plan. Full source code, two
executed dataset experiments, trained models, plots, an interactive dashboard,
an API, tests and GitHub documentation are included.

## 1. See it immediately

Open **[`outputs/dashboard.html`](outputs/dashboard.html)** in a browser.
It is a standalone interactive snapshot: no installation, CDN or server needed.
Switch between synthetic-building and real UCI-household data, explore horizons,
compare models and change the load-shifting scenario.

For a Python-backed live demo:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000**. The downloaded archive already contains trusted
local model bundles and processed data. If you clone the later GitHub repository,
run the next section first: large derived model files are intentionally gitignored.
Use Python **3.11, 3.12 or 3.13**, not an older Python environment.

## 2. Rebuild the evidence

```bash
python main.py --dataset synthetic
python main.py --dataset uci
python scripts/export_dashboard.py
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/validate_project.py
```

Both hourly datasets are bundled, so rebuilding needs no dataset download.
`--refresh-data` deliberately regenerates synthetic data or downloads UCI anew.
The synthetic generator is seeded. Exact runtimes and tiny floating-point values
can vary by machine; do not promise bit-for-bit results on arbitrary environments.

## 3. Read in this order

| Need | File |
|---|---|
| Professional GitHub landing page | [README.md](README.md) |
| All requested A–M sections and 10 implementation phases | [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) |
| Every source file, copy-paste ready | [docs/FULL_SOURCE.md](docs/FULL_SOURCE.md) |
| Beginner explanation of each module | [docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md) |
| Hands-on executed notebook | [notebooks/01_energy_forecasting.ipynb](notebooks/01_energy_forecasting.ipynb) |
| Data source, units and missingness | [docs/DATA_CARD.md](docs/DATA_CARD.md) |
| Actual results and model limitations | [docs/RESULTS.md](docs/RESULTS.md), [docs/MODEL_CARD.md](docs/MODEL_CARD.md) |
| API / curl / Windows tests | [docs/API.md](docs/API.md) |
| Publish the repo and plan seven honest days of commits | [docs/GITHUB_PLAYBOOK.md](docs/GITHUB_PLAYBOOK.md) |
| Screenshots and proof checklist | [docs/PROOF_CHECKLIST.md](docs/PROOF_CHECKLIST.md) |
| Placement / internship demo and interview answers | [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md), [docs/INTERVIEW_QA.md](docs/INTERVIEW_QA.md) |
| Corrections to unsupported claims in the original brief | [docs/CLAIMS_AUDIT.md](docs/CLAIMS_AUDIT.md) |
| Requirement-by-requirement completion map | [docs/REQUIREMENTS_TRACEABILITY.md](docs/REQUIREMENTS_TRACEABILITY.md) |

## 4. Real measured results from this build

| Dataset | Validation-selected model | Test 24 h RMSE | Test one-step R² |
|---|---|---:|---:|
| Synthetic building, 17,520 hours | MLP | 6.888 kW | 0.831 |
| Public UCI household, 34,587 cleaned hours | Random forest | 0.598 kW | 0.567 |

These are **different forecasting tasks**. R² is not percentage accuracy. The
24-hour backtest uses 14 sampled origins, not every test day. Detailed metrics,
forecasts, hashes, test reports and actual browser captures are included.

## 5. What remains yours to do

The project has **not been published to your GitHub account**. Follow the supplied
Git commands, use your own account, and capture the real repository URL/CI result.
Do not invent a seven-day commit history. Reproduce the code, understand it and add
one documented experiment before using it in an interview.

This is a historical-data / virtual-decision lab, not a live corporate energy
system. Bill savings are hypothetical; energy-conserving load shifting alone
produces **zero modeled CO₂ reduction** under the constant-intensity assumption.
