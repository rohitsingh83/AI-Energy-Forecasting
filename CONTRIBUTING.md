# Contributing and learning honestly

1. Create an issue describing a specific problem, hypothesis or improvement.
2. Work on a branch, e.g. `git switch -c feature/weather-forecast-input`.
3. Install the runtime and development requirements in a virtual environment.
4. Make one focused change, document assumptions and add/update tests.
5. Run `python -m pytest -q` and `python scripts/validate_project.py` where bundles exist.
6. If changing model behavior, run both dataset experiments, report baselines
   and update the run evidence. Do not select hyperparameters on the test set.
7. Open a pull request explaining what changed, why and how it was verified.

## Portfolio integrity

This scaffold was assembled with AI assistance. Understand and verify each module
before presenting it as your own proof of learning. Add your own documented
experiments, explain assistance if asked, and never invent a corporate client,
production deployment, seven-day historical commit trail, measured saving or
improved model score. Commit actual work when you do it; do not backdate commits.

Useful first contributions: a new chronological experiment, holiday features,
external forecast-weather covariates with issuance timestamps, more rolling
origins, site-aware timezone handling, residual monitoring, or a model ablation.
Each improvement should have evidence rather than only a README claim.
