# Proof checklist and asset map

## Completed in this build

- [x] Complete modular implementation, one-command run and phase commands.
- [x] Seeded virtual-building dataset with honest source metadata.
- [x] Official UCI download, coverage-aware hourly adaptation and CC BY attribution.
- [x] Two full experiments with actual logs, model selection and stored predictions.
- [x] Causal fill, observed-target masks and train/serving feature parity.
- [x] Naive baselines, Ridge, RF and MLP comparisons.
- [x] Separate untouched one-step and 24-hour recursive test evidence.
- [x] Final reusable models, environment metadata, hashes and explicit cutoffs.
- [x] Forecasts and empirical bands with documented horizon limits.
- [x] Virtual cost sandbox with conserved energy and no invented carbon benefit.
- [x] Responsive API/dashboard and offline interactive HTML export.
- [x] Nine analytical plots per dataset and five actual browser captures.
- [x] 67 local automated tests passed; output text and JUnit XML included.
- [x] 29/29 actual artifact checks passed, report included.
- [x] 10 browser smoke checks passed, report included; no JS exceptions observed.
- [x] Executed beginner notebook and complete source-code compendium.
- [x] A–M guide, data/model cards, claims audit, GitHub and interview playbooks.

These checkmarks describe delivered local evidence, not a claim that the student
personally authored it without assistance or that remote GitHub Actions has run.

## Verification and publication status

- [x] Fully reproduced on local machine: all 67/67 pytest unit tests and 29/29 artifact verification checks passed.
- [x] Two complete end-to-end dataset experiments executed (synthetic building + public UCI household) with Ridge, RF, and MLP models.
- [x] Live interactive dashboard deployed client-side to GitHub Pages with offline standalone export.
- [x] Captured all 10 proof screenshot assets including GitHub repository preview and mobile viewports.
- [x] Automated GitHub Actions CI and GitHub Pages deployment workflows configured.
- [ ] Push latest changes to remote GitHub account (`git push origin main`).
- [ ] Record video walkthrough using the provided demo script ([`docs/DEMO_SCRIPT.md`](DEMO_SCRIPT.md)).

## Required proof assets

| Asset | Delivered | File naming / storage |
|---|---|---|
| Dataset preview | Yes | `outputs/<run>/dataset_preview.csv`; full source in `data/raw/` |
| Preprocessing output | Yes | `outputs/<run>/data_quality.json`; `images/<run>/data_quality.png` |
| Training logs | Yes | `outputs/<run>/logs/pipeline.log` |
| Feature preview | Yes | `outputs/<run>/feature_preview.csv` |
| Prediction graph | Yes | `images/<run>/forecast.png` |
| Actual vs predicted graph | Yes | `images/<run>/actual_vs_predicted.png` — explicitly one-step |
| Day-ahead prediction proof | Yes | `outputs/<run>/backtest_predictions.csv`, `horizon_metrics.csv` |
| Error metrics | Yes | `outputs/<run>/metrics.json`, `model_comparison.csv` |
| Simulation proof | Yes | `outputs/<run>/simulation.csv`, `.json`; `images/<run>/simulation.png` |
| Desktop/mobile app | Yes | `images/screenshots/01_...` through `05_...` |
| GitHub repo preview | Yes | `images/screenshots/10-github-repo.png` & `docs/screenshots/10-github-repo.png` |

`<run>` is `synthetic` or `uci`. Their data, units, selected models and cutoffs are
separate. Do not move a graph from one run next to another run's metric caption.

## Actual browser screenshots

1. `01_overview_synthetic.png`: default 24-hour building overview, 10% scenario.
2. `02_decision_sandbox.png`: genuine interactive check with **20%** flexibility;
   it is intentionally different from the saved default 10% scenario.
3. `03_model_evidence.png`: validation/test table and actual held-out curves.
4. `04_overview_uci.png`: real measured household, default 24-hour overview.
5. `05_mobile.png`: genuine responsive 390-pixel-wide household view.

These are captures of the actual running app, not AI-generated mockups. Training
logs and CSV/JSON are actual artifacts, not screenshots of invented terminal output.

## Recreate captures

Runtime is independent of screenshot tooling. For optional browser reproduction:

```bash
python -m pip install playwright
python -m playwright install chromium
# Linux only if browser shared libraries are absent:
# python -m playwright install-deps chromium
python scripts/browser_check.py --url http://127.0.0.1:5000
```

Keep `python app.py` running in another terminal. The helper tests navigation,
48-hour controls, scenario updates, dataset switching, downloads, mobile width and
offline HTML; it saves the actual report in `outputs/browser-checks.json`.

## Verification commands

```bash
python -m pytest -q --junitxml=outputs/test-results.xml
python scripts/validate_project.py
python scripts/export_dashboard.py
```

- Pytest fixtures use temporary directories, not bundled evidence.
- Artifact checks compare stored metrics with stored predictions and input hashes.
- Offline HTML is a **snapshot**; re-export it after retraining.
- CI YAML is ready but cannot prove a remote pass until you push and inspect Actions.

## README image snippets

From root README:

```markdown
![Actual dashboard](images/screenshots/01_overview_synthetic.png)
![Future forecast](images/synthetic/forecast.png)
![Measured-household model comparison](images/uci/model_comparison.png)
```

Do not use local drive paths, missing filenames or external assets that only work
in your own environment. Keep captions source- and task-specific.

## Final integrity gate

Before using this for placements, answer without guessing:

1. What source, cutoff and unit does this chart use?
2. What is allowed to enter each recursive step?
3. Which metric selected the model, and was it validation or test?
4. How many forecast origins and observed targets were scored?
5. Which savings are simulated, and what constraints are missing?
6. Why does modeled carbon stay unchanged in the default scenario?
7. What is one actual limitation and one genuine personal contribution?

If you cannot answer yet, use the notebook/code walkthrough before marketing it.
