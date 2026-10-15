# GitHub publishing, commit strategy and proof plan

The deliverable is GitHub-ready, but no repository has been created or pushed in
your account. A real account/remote URL and your authorization are needed. Do not
invent a repo screenshot, CI badge, hosted demo or historical commit trail.

## 1. Repository identity — copy-paste ready

**Name:** `ai-energy-consumption-forecasting`

**Description:**

```text
CPU-friendly energy demand forecasting with real UCI data, virtual-building simulation, leakage-safe backtests, a Flask dashboard and reproducible ML evidence.
```

**Topics:**

```text
python
machine-learning
time-series
energy-forecasting
smart-buildings
climate-tech
scikit-learn
flask
data-science
random-forest
mlp-regressor
portfolio
```

**About/demo link:** add your real demo/video URL only after it exists. The standalone
`outputs/dashboard.html` can be hosted as a labeled snapshot on a static host. It
is not the same thing as a deployed Flask inference API; GitHub Pages runs no Python.

## 2. Before uploading

```bash
python -m pytest -q
python scripts/validate_project.py
python scripts/export_dashboard.py
```

Read `git status` before committing. Inspect custom data and notebook outputs for
secrets/private records. Keep UCI attribution. Remove caches and virtual environments.
Do not upload the 126.8 MB original text, its cached archive or private meter files.

The provided `.gitignore` deliberately excludes binary models and processed CSVs.
The downloadable package includes them for convenience, but a Git clone should
recreate them with `python main.py --dataset synthetic` and `--dataset uci`.
This makes source control smaller and safer. Do not claim a model binary is a safe
format for arbitrary user uploads.

## 3. Create an empty GitHub repository

1. Sign in to your account and select **New repository**.
2. Choose the recommended name/description or a clear alternative.
3. Choose public for placement proof, provided all published data are permitted.
4. Do **not** initialize another README, license or `.gitignore` when pushing this
   existing folder. They are already included.
5. Copy the actual HTTPS remote for your repository.

## 4. First push

From the extracted root folder (not `src`):

```bash
git init
git branch -M main
git config user.name 'Your Real Name'
git config user.email 'YOUR_VERIFIED_OR_GITHUB_NOREPLY_EMAIL'
git status --short
git add .
git diff --cached --stat
# Inspect before committing. Do not commit private data or credentials.
git commit -m 'feat: add reproducible energy forecasting lab and evidence'
git remote add origin https://github.com/YOUR_USERNAME/ai-energy-consumption-forecasting.git
git push -u origin main
```

Replace placeholder username/email with your real values. They are instructional
examples, not a created repo. Authenticate using GitHub's normal credential manager,
SSH or secure token prompt. Never paste a token into source, docs or a remote URL.

If `origin` exists, inspect it with `git remote -v` before using
`git remote set-url origin <YOUR_REAL_URL>`. Do not overwrite someone else's remote
or force-push an unrelated repository. If GitHub already contains commits, clone
it first or reconcile history deliberately rather than blindly using `--force`.

## 5. Professional commits going forward

Use small real changes and clear messages:

```text
feat: support a new permitted hourly CSV schema
fix: reject empty run IDs before resolving output paths
test: verify recursive predictions ignore future actuals
experiment: compare validation performance without weekly lag
docs: clarify kW versus interval kWh
ui: add a visible historical cutoff to the forecast chart
```

Example branch workflow:

```bash
git switch -c experiment/weekly-lag-ablation
# Make and verify a meaningful change.
git add src tests docs
git commit -m 'experiment: document validation weekly-lag ablation'
git push -u origin experiment/weekly-lag-ablation
```

Open a pull request explaining hypothesis, code change, tests, validation evidence
and limitations. Do not repeatedly optimize against the already-viewed test set.

## 6. Seven-day proof plan

This is a future learning schedule. The supplied code already exists; one honest
import commit is preferable to staging artificial historical “days.” If you rebuild
it as a learning exercise, commit actual completed work at the time it happens.

| Day | Focus | Meaningful output | Commit example | Proof |
|---|---|---|---|---|
| 1 | Setup | Environment check + problem/units note in your own words | `docs: record environment and project scope` | Real imports/version output |
| 2 | Dataset | Source/license/coverage inspection | `data: document household source coverage` | Actual preview + metadata |
| 3 | Preprocessing | Trace bounded causal fill; add an edge-case test | `test: cover a missing-data boundary` | Quality audit + new passing test |
| 4 | Model | Reproduce baselines, train-only scaling and validation lock | `experiment: compare validation feature ablation` | Config + validation table |
| 5 | Evaluation | Explain recursive windows; inspect horizon residuals | `analysis: document day-ahead forecast errors` | Backtest CSV + honest limitations |
| 6 | Visualization | Improve one useful unit/control/graph | `ui: improve demand evidence readability` | Actual before/after capture |
| 7 | Publish | README, fresh-clone check, real URL/demo | `docs: publish reproducibility and demo links` | Real repo/CI screenshot |

No empty/backdated commits, fake contribution graph or fabricated client outcomes.
AI assistance is acceptable when you understand, verify and contribute genuine work.

## 7. README visual strategy

Keep the top concise: project goal, one actual dashboard screenshot, quickstart,
results, data/source honesty and limitations. Deeper explanation is linked.
The supplied root README already follows this structure.

Use relative paths from repository root:

```markdown
![Actual dashboard](images/screenshots/01_overview_synthetic.png)
![Forecast curve](images/synthetic/forecast.png)
![Real-household comparison](images/uci/model_comparison.png)
```

Do not use local Windows paths or `/home/user/...` in README image links. Ensure
file capitalization matches on GitHub. Avoid dozens of redundant images at the top.

## 8. Verify the remote repository

1. Open the real URL while signed out or in a private window.
2. Confirm README images and documentation links render.
3. Confirm the license credits UCI separately from MIT code.
4. Open Actions. Report a pass only after it actually runs green on your remote.
5. Clone into a fresh location and test without bundled ignored models:

```bash
git clone https://github.com/YOUR_USERNAME/ai-energy-consumption-forecasting.git
cd ai-energy-consumption-forecasting
python -m venv .venv
# Activate using your OS-specific command.
python -m pip install -r requirements.txt pytest==9.0.3
python main.py --dataset synthetic
python -m pytest -q
python app.py
```

6. Add a short demo video if useful. Blur personal windows; use a clean desktop.
7. Pin the repository. Add factual topics and a link to your learning notes.

## 9. Release strategy

After you personally reproduce and verify:

```bash
git tag -a v1.0.0 -m 'Verified dataset forecasting and virtual decision lab'
git push origin v1.0.0
```

Use a GitHub release for optional trusted model/download bundles, with environment,
input hashes and a warning about pickle trust. Keep the source tree lightweight.
A release should state that the demo uses historical datasets and scenario savings
are hypothetical, not business outcomes.

## 10. Placement proof and honest résumé framing

Strong evidence chain:

```text
source → causal transforms → ordered validation → held-out predictions
       → reproducible metrics → runnable UI/API → tests → public documentation
```

A polished page alone is not enough. Be able to explain at least one failed model,
the real household's high relative error, why RF stays selected despite MLP's better
sampled test RMSE, and why constant-intensity shifting does not save modeled carbon.
Use actual project metrics only after reproducing them. Do not list named energy
companies as clients or assert live vacancies/salary premiums without verification.
