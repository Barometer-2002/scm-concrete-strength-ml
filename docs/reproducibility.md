# Reproducibility and research boundaries

## Reported study configuration

The underlying study used 1,456 cylinder-strength records, eight engineered
features, an 80/20 split with random seed 66, 300 Optuna trials per model, and
5 repeats of 10-fold cross-validation. The compared model families were RF,
XGBoost, LightGBM, SVR, and MLP.

The selected LightGBM model reported test-set `R2 = 0.8620`,
`RMSE = 6.6175 MPa`, and `MAE = 4.2178 MPa`. At 90% target coverage, the
KNN-RSCP experiment reported empirical coverage `0.9007` and mean interval
width `20.2764 MPa`.

These values are historical study results. They are **not** reproduced by the
synthetic example in this repository. Exact numerical reproduction requires
the audited research dataset, frozen split, package versions, and best-model
parameters.

## What this repository verifies

- deterministic feature construction;
- common model interfaces and leakage-aware scaling pipelines;
- repeated cross-validation and regression metrics;
- fixed-width, residual-scaled, and KNN-enhanced conformal intervals;
- constraint, Pareto, and TOPSIS decision-support utilities;
- an end-to-end workflow on synthetic data.

The public package is a cleaned implementation of the reusable methods, not a
byte-for-byte archive of the paper workspace.
