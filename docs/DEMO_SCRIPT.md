# Demo and interview presentation kit

## 1. 90-second demo — speaking notes, not fabricated experience

**0–15 s — problem and scope**

> This is my dataset-driven energy forecasting lab for homes and buildings. It
> uses a public measured household dataset and a separate virtual building, so
> it does not require or claim access to corporate energy systems.

**15–30 s — sources and cutoff**

Show the dataset dropdown and source banner.

> The synthetic series is seeded and clearly labeled. The measured benchmark is
> UCI's French household active-power dataset, adapted hourly with coverage checks
> and attribution. Predictions start after each dataset's last observation, not today.

**30–50 s — trustworthy evaluation**

Open Model evidence.

> I compare three seasonal/persistence baselines with Ridge, random forest and a
> scaled MLP. Features use only past demand and known calendar. I select on ordered
> validation day-ahead error and report untouched test scores. One-step replay
> updates actual lags; a day-ahead window recursively uses its own predictions.

If asked for a number, name its task:

> The validation-selected RF achieved 0.598 kW day-ahead RMSE on 14 sampled UCI test
> origins. Relative household error remains high; this is not an operational-readiness claim.

**50–70 s — future and a decision**

Open Forecast explorer, then Decision sandbox. Move flexibility from 10% to 20%.

> The next 24 hours have a descriptive validation error band. Longer horizons are
> exploratory. The sandbox shifts flexible forecast energy to illustrative cheaper
> hours while conserving total kWh and respecting original peak headroom. Costs
> are hypothetical, not achieved savings. Constant-intensity modeled carbon is unchanged.

**70–90 s — engineering proof**

Open Project runbook or the actual repo after publishing.

> Modular code, actual CSV/JSON predictions, source hashes, real plots, an executed
> notebook and 67 passing automated tests make the work reproducible. My next step
> is a documented experiment with more forecast origins and properly issued weather
> forecasts, not adding a larger neural network without baseline evidence.

If you have not personally reproduced or extended the scaffold, do not recite “I
built it from scratch.” Explain that you used AI assistance and identify what you
understood, verified and changed yourself.

## 2. Five-minute deep-dive

1. **30 s:** domain problem, kW vs kWh, source honesty.
2. **45 s:** cleaning and gap mask; demonstrate a filled value not scored as truth.
3. **45 s:** cyclic calendar + lags; explain `shift(1).rolling(...)`.
4. **60 s:** baselines, ordered validation selection, one-step versus recursive.
5. **45 s:** real measured limitation and why test-best MLP does not replace RF.
6. **45 s:** API horizon contract and virtual cost scenario conservation.
7. **30 s:** tests, fresh-clone reproducibility, own contribution and future work.

## 3. Screen-recording checklist

- Reproduce the code before recording; use a clean browser and readable zoom.
- Show source labels and cutoff, not only attractive KPIs.
- Start with the default 24-hour horizon; explain 48/168 h are exploratory.
- Show the selected row and at least one seasonal baseline.
- Change a sandbox setting and wait for a real recalculation.
- Say savings are hypothetical and no live meter is connected.
- Show actual test output or a verified repo/CI run, not a fabricated terminal.
- End with real repo/demo links after you have published them.
- Blur unrelated personal accounts/windows; never show credentials.

## 4. Copy-paste résumé wording

**Safe project summary:**

> Built a CPU-based hourly energy forecasting lab with public UCI household data
> and a seeded virtual building; compared seasonal baselines, Ridge, random forest
> and MLP using chronological validation and recursive 24-hour backtests, then
> packaged results, scenario analysis, a Flask dashboard and automated tests.

**Optional evidence bullet after personal reproduction:**

> Validation-selected RF reached 0.598 kW day-ahead RMSE on 14 sampled UCI test origins,
> 29.59% lower RMSE than a weekly seasonal baseline; published reproducible prediction
> artifacts, data/model cards and 67 passing local checks.

Do not call RMSE reduction “energy saved.” Do not call R² “accuracy.” Do not imply
GitHub/CI was published until you have the actual URL and result.

## 5. LinkedIn/project post template

> I reproduced and studied an AI-assisted energy forecasting project using two
> clearly separated sources: public UCI household measurements and a seeded virtual
> building. The focus was not just model fitting: I worked through causal lag features,
> chronological evaluation, baseline comparisons and the difference between next-hour
> replay and fixed-origin day-ahead forecasting.
>
> The lab includes a Flask dashboard, downloadable forecasts, a hypothetical
> load-shifting sandbox and automated checks. Scenario savings are simulated, not
> measured, and the model is not a live grid controller.
>
> My contribution/experiment: **[describe actual change and result]**.
> Code/demo: **[insert real links after publishing]**.

Do not leave placeholders in a published post. Describe your actual work honestly.
