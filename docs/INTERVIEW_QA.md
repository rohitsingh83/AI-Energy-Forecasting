# Interview Q&A — answers that match the implemented project

Use these as learning notes, not a script for claiming experience you have not had.
Know the actual code and your own changes; explain AI assistance honestly if asked.

## 1. What is this project and what did it actually do?

It is an hourly electricity-demand forecasting lab using a public measured household
series and a separate seeded virtual building. The implementation loads/cleans data,
creates past-only features, compares baselines and ML, selects on chronological
validation, reports held-out one-step/day-ahead results, refits for future forecasts,
and serves those through Flask. A sandbox explores hypothetical flexible-load costs.
It was not connected to corporate meters or used to control a real grid.

## 2. Why is forecasting useful in energy?

It supplies an estimate for planning procurement, capacity and flexible scheduling.
It can help anticipate peaks, but it cannot independently prevent blackouts or
reduce consumption. Actual results depend on supply capacity, constraints, tariffs,
controls and the actions taken in response to forecasts.

## 3. What data did you use?

Two clearly separated datasets: a seeded 730-day virtual mixed-use building and
UCI Individual Household Electric Power Consumption by Hebrail/Berard, covering one
French household from 2006–2010. The real minute active-power series is averaged
hourly using at least 54 valid minutes per observed hour. Attribution is CC BY 4.0.
The synthetic building is not described as real corporate smart-grid logs.

## 4. What is the difference between kW and kWh?

kW is instantaneous or interval-average power; kWh is energy over a duration. At
2 kW mean for one hour the energy is 2 kWh. Here each row is a one-hour interval,
so forecast interval kWh equals mean kW numerically. A daily energy total is the
sum of hourly kWh. Cumulative meter-register kWh must be differenced first.

## 5. Why not use a random train/test split?

A forecast must predict later behavior using information available earlier. A
random split can put future or highly neighboring seasonal behavior into training
and make results optimistic. This project slices eligible rows chronologically
70/15/15 and keeps validation selection separate from final test reporting.

## 6. How did you handle missing data without leakage?

Sort and resample hourly mean power, mask invalid readings, forward-fill at most
three hours from the past, and keep an original observation mask. Long gaps stay
missing and invalidate affected feature windows. Imputed targets are not fit labels
or evaluation truth. No backward fill or future-endpoint interpolation is used.

## 7. Which features did you engineer?

Cyclic sine/cosine hour, weekday and day-of-year, weekend flag, lags 1/2/24/48/168,
and past-only 24-hour mean/std and 168-hour mean: 15 features. The rolling windows
start from `shift(1)`. Training and inference feature rows are compared in tests.

## 8. Why use sine/cosine rather than raw hour alone?

Hours are circular. Hour 23 and 0 are neighbors, despite being far apart as raw
numbers. The sine/cosine pair preserves circular proximity and lets linear/MLP
models represent routine cycles more naturally. Trees can also use these features.

## 9. Did you use weather or occupancy?

They are present only as generated explanatory variables in the synthetic dataset.
The forecasting features deliberately do not use unknown future realized weather
or occupancy. A future extension should use weather forecasts and planned schedules
that were issued before the demand-forecast cutoff, not realized future values.

## 10. Which algorithms and why?

Three naive baselines establish whether ML is worthwhile. Ridge gives an explainable
regularized linear reference. Random forest learns nonlinear interactions with
little preprocessing. A small scaled MLP tests the neural-network approach from
the brief without requiring a GPU. Model choice is determined by validation error,
not by the most impressive algorithm name.

## 11. How did you scale the MLP?

X uses a StandardScaler pipeline; y uses StandardScaler inside
TransformedTargetRegressor. Each fit learns transformations only from its allowed
training segment. Internal random validation early stopping is disabled; the explicit
chronological validation split selects models. Seed and training settings are recorded.

## 12. What is recursive forecasting?

At the cutoff I know the last 168 readings. I predict the next hour, append that
prediction to working history, and use it as a lag for the next step. I repeat for
the desired horizon. No future measured demand enters a fixed-origin window.
Errors can compound; 168-hour output is exploratory because only 24 h was backtested.

## 13. What is one-step replay versus day-ahead evaluation?

One-step replay predicts each next hour using actual demand that arrived before
that hour. Day-ahead issues all 24 hours using only history before the origin and
its own recursive predictions. The former can look better and must not be advertised
as fixed-origin day-ahead accuracy. Both are saved separately.

## 14. How was the model selected?

Fit all candidates on train, generate validation 24-hour forecasts on up to 14
evenly sampled eligible daily origins, choose the lowest validation RMSE and lock
it. Refit on train+validation for untouched test reporting. After evaluation, refit
only the locked champion on all historical data for genuine future serving.

## 15. What results did the completed build produce?

Synthetic: validation-selected MLP, test day-ahead RMSE 6.888 kW, one-step R² 0.831.
Real UCI: validation-selected RF, test day-ahead RMSE 0.598 kW, day-ahead R² 0.253,
one-step R² 0.567. Day-ahead tests use 14 sampled origins, not every test day.
Absolute kW errors across differently sized sites are not directly comparable.

## 16. Is 0.831 R² “83.1% accuracy”?

No. R² compares squared error to predicting the evaluation target mean. It is not
classification accuracy or a fraction of correctly predicted hours. It can be
negative. I report error units, forecast task, sample count, baseline and limitations.

## 17. Why is the real household harder?

A single home's appliance behavior is irregular, data have long gaps, and the model
has no observed schedules/weather/holidays. The synthetic building imposes recurring
patterns by design. UCI day-ahead WAPE of 48.44% is a substantial limitation; I do not
claim operational readiness from a relatively small absolute kW error alone.

## 18. MLP scores better than RF on UCI test day-ahead windows. Why serve RF?

RF won validation and was locked before the test loop. Changing the winner now
would use test data for selection. The difference is reported transparently. A
new research round should use training/validation and reserve a fresh future holdout.

## 19. What is the forecast error band?

Prediction ± the pooled 90th percentile of absolute validation day-ahead errors,
with lower power clipped at zero. It is descriptive, not a guaranteed conformal
or normal-confidence interval. It is shown only for the validated first 24 hours.
Temporal dependence, horizon differences and the final refit can change coverage.

## 20. How did you save and serve the model?

Joblib stores the trusted local estimator, feature names, recent 168-hour history,
cutoff, provenance and validation radius. Flask accepts a run ID and horizon, loads
or caches the matching bundle and returns timestamped JSON. It does not train per
request. Untrusted pickle uploads are unsafe and not accepted.

## 21. Why not accept only hour and weekday in the API?

The trained model depends on recent demand lags, not just a generic calendar. An
arbitrary hour/day lacks a date, cutoff and history. The API instead forecasts the
next contiguous hours from its saved historical context and validates 1–168 h.

## 22. How is the business simulation implemented?

Take the first 24 predicted hours. Price their kWh with illustrative off-peak,
standard and peak rates. Move a configurable peak-hour flexible share into
available off-peak headroom. Verify equal total energy, nonnegative loads and no
increase beyond the original peak. Return hypothetical costs and explicit assumptions.
It is a scheduling heuristic, not a real equipment optimizer or billing quote.

## 23. Does it reduce carbon emissions or actually save money?

No measured outcome is claimed. The default synthetic scenario gives ₹438.55
hypothetical cost saving under invented rates and flexibility. Total energy is
unchanged. With a constant illustrative carbon factor, modeled emissions stay
unchanged: zero reduction. Real impact needs interventions, real constraints,
actual tariffs/carbon data and a credible counterfactual.

## 24. What tests and proof did you provide?

67 automated checks cover deterministic generation, schema/coverage, causal fills,
feature leakage/parity, ordered splits, scaling, recursive behavior, scenario
invariants, JSON/API errors and an isolated offline end-to-end run. Separate
artifact checks recompute scores/hashes; browser checks capture real interactions.
Executed notebook, CSV/JSON predictions, actual logs and figures back every metric.
Remote GitHub/CI proof is only claimed once actually published.

## 25. How would you make this production-ready?

Use representative multisite data and explicit timezone/DST logic, forecast-weather
issuance times, holidays/schedules, richer rolling backtests, calibrated probabilistic
forecasts, drift/data freshness monitoring and retraining policy. Add a model registry,
authenticated production WSGI service, observability, safe controls, physical job/
comfort constraints and human approval. None of that is implied by a Flask demo.

## 26. What did you personally learn or change?

Answer with a real action: reproduce, debug a test, explain the empty-run-ID fix,
perform a validation ablation, add a constraint or improve a chart. Do not claim
unperformed corporate deployments or solo authorship of an AI-assisted scaffold.
A specific experiment plus evidence is more convincing than memorized terminology.
