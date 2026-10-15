# Model card

## Intended use

Educational forecasting and portfolio proof: hourly demand predictions for a
single historical site, model comparison and a transparent virtual load-shifting
scenario. Not intended for billing, electricity trading, safety-critical control,
blackout prevention, automated dispatch or performance guarantees.

## Implementation

- Runtime: Python / pandas / NumPy / scikit-learn / joblib; CPU only.
- Inputs at issuance: last 168 hourly mean-power readings and the known calendar.
- Features: cyclic hour/day-of-week/day-of-year, weekend flag, lags 1/2/24/48/168,
  past-only 24-hour mean/std and 168-hour mean. 15 features in total.
- Targets: observed hour-average active power in kW, clipped nonnegative at inference.
- No realized future weather, occupancy or future demand enters a forecast.
- Forecasting: recursive hourly updates. For step two and later, newest lag values
  include the model's own previous predictions. Maximum supported horizon: 168 h;
  only 24-hour forecasting is validated by the supplied multistep protocol.

## Candidates

| Candidate | Details |
|---|---|
| Previous hour | Persistence baseline; recursively constant |
| Same hour yesterday | Seasonal naive with lag 24 |
| Same hour last week | Seasonal naive with lag 168 |
| Ridge | StandardScaler + L2-regularized linear regression, alpha=1 |
| Random forest | 90 trees, max_depth=12, min_samples_leaf=6, max_features=0.9 |
| MLP | X and y standardized on fit data; hidden layers (48,24), ReLU/Adam, max_iter=220, shuffle=False, early_stopping=False |

The MLP's internal random validation holdout is disabled. This keeps time-based
model selection in the explicit validation split. Scaling is fit only on each
allowed training segment. Seed 42 is used for generated data and estimators.
A `--quick` smoke run uses smaller RF/MLP settings and is not benchmark-equivalent.

## Selection and evaluation protocol

1. Build causal lag features and retain observed target rows with complete inputs.
2. Split ordered eligible samples approximately 70% train / 15% validation / 15% test.
3. Fit each candidate on train only.
4. Measure validation one-step performance and 24-hour recursive performance on
   up to 14 evenly sampled eligible daily origins within validation.
5. **Lock the champion with the lowest validation day-ahead RMSE.** Baselines may win.
6. Refit each candidate on train+validation and report untouched test results.
   Up to 14 eligible daily test origins are sampled; they are not every test day.
7. Exclude imputed/unobserved target hours when computing every score.
8. After all test reporting, refit the selected candidate on all eligible historical
   observations for genuine future forecasting. Do not score this final refit on
   the same test data it has now seen; the API serves it only after the data cutoff.

In **one-step replay**, actual lag readings arrive between successive predictions.
In **day-ahead backtesting**, all 24 predictions at one origin are recursive with no
future actual updates inside that window. Next origin can use history available
at that later time. These tasks must not be described as the same accuracy measure.

## Bundled results

| Dataset | Selected model | One-step RMSE | One-step R² | 24 h RMSE | 24 h R² | 24 h WAPE |
|---|---|---:|---:|---:|---:|---:|
| Synthetic building | MLP | 6.711 kW | 0.831 | 6.888 kW | 0.829 | 5.86% |
| UCI household | Random forest | 0.466 kW | 0.567 | 0.598 kW | 0.253 | 48.44% |

See [RESULTS.md](RESULTS.md) and raw artifacts for all candidates and exact values.
The UCI MLP happens to score better than RF on the sampled test day-ahead windows.
We **do not switch** to it, because validation had already selected RF. This honest
difference is evidence that holdout protocol matters, not a reason to cherry-pick.
The real household's day-ahead WAPE is relatively high; this model is not ready for
operational household scheduling solely on the basis of these results.

## Error band

Take the 90th percentile (`method='higher'`) of absolute errors pooled across the
selected model's validation 24-hour backtests. Show prediction ± this radius,
clipping the lower bound at zero, **only for the first 24 forecast hours**.

| Source | Radius | Empirical test coverage |
|---|---:|---:|
| Synthetic | 13.662 kW | 92.84% |
| UCI | 0.951 kW | 90.18% |

This is a descriptive error band, not a normal-distribution confidence interval
or formally guaranteed conformal interval. Errors are time-dependent, pooled
across lead times, and calibration comes from a pre-refit validation model.
The final all-history refit can have a different error distribution. Recalibrate
on rolling origins for an operational system. No uncertainty estimate is offered
beyond 24 hours, because that horizon was not backtested here.

## Metrics interpretation

- MAE: average absolute miss, in kW.
- RMSE: square root of average squared miss, in kW; penalizes large misses more.
- R²: improvement in squared error relative to predicting the evaluation target
  mean; can be negative, **is not classification accuracy**, and 0.83 is not a
  promise that 83% of future hours are correct.
- MAPE: mean percentage error; unstable when true load is close to zero.
- WAPE: total absolute miss / total observed absolute load; a useful aggregate
  percentage but still affected by the evaluation sample and load scale.
- Compare RMSE only between models on the **same site, split and forecast task**.
  Smaller absolute UCI kW error does not imply a better model than the larger
  building forecast; the target magnitudes are completely different.

## Limitations and responsible use

Synthetic patterns are designed by us and easier than unknown real operations.
Public household data are old, site-specific and incomplete. No true operational
experiment, cost saving, carbon saving, net-zero outcome or grid reliability
benefit has been demonstrated. Forecasts can inform planning but cannot create
supply capacity or eliminate outages on their own.

Production would need representative multisite data, explicit UTC/site clocks,
forecast-weather availability timestamps, holiday schedules, all-test-day and
seasonal backtests, probabilistic calibration, model monitoring/drift detection,
data freshness alerts, retraining policy, model registry, authenticated APIs,
production WSGI, operator approval and safe equipment constraints.

## Artifact/reproducibility information

Each `models/<run>/metadata.json` includes Python/package versions, selected model,
feature order, input SHA256 and final refit cutoff. `outputs/<run>/run_manifest.json`
mirrors that evidence. Raw files and transformations are attributed in the data
card. Exact direct dependencies are pinned in `requirements.txt`; numerical
rounding and training time can vary with hardware and transitive dependencies.
Only load trusted local `.joblib` files. New versions should be retrained, not
assumed compatible with arbitrary pickle files or library versions.
