# Methodology

## Engineering features

The raw mixture dosages are converted into the eight inputs used by the study:

- `B = Cement + FA + SF + GGBFS`
- `W/B = Water / B`
- `A/B = (Coarse aggregate + Fine aggregate) / B`
- `SR = Fine aggregate / (Coarse aggregate + Fine aggregate)`
- `SP/B = SP / B`
- `FA/B = FA / B`
- `SF/B = SF / B`
- `GGBFS/B = GGBFS / B`

This representation separates total binder content, water demand, aggregate
structure, admixture dosage, and individual SCM replacement ratios. The code
checks arithmetic validity but does not certify mixture feasibility.

## Point prediction

The model registry exposes Random Forest, XGBoost, LightGBM, support vector
regression, and a multilayer perceptron. SVR and MLP are wrapped in a
`StandardScaler` pipeline so that scaling is learned inside every validation
fold. `repeated_cross_validate` evaluates all model families under the same
shuffled repeated K-fold protocol. Optional Optuna tuning minimizes mean CV
RMSE.

## KNN-RSCP intervals

`KNNResidualScaleConformalRegressor` implements a reusable split-conformal
variant:

1. Split the available training records into a fitting subset and a calibration
   subset.
2. Generate out-of-fold predictions within the fitting subset and use absolute
   residuals as scale targets.
3. Add two local KNN diagnostics to the scale-model inputs: mean standardized
   neighbor distance and neighbor-target standard deviation.
4. Fit a residual-scale model and normalize calibration residuals by their
   predicted scale.
5. Apply the finite-sample corrected conformal quantile and return adaptive
   intervals for new records.

The implementation targets marginal coverage under exchangeability. It does
not guarantee conditional coverage for every mixture family, material source,
or strength range. External validation remains necessary.

## Multi-objective extension

The decision-support module is an engineering extension rather than part of the
reported paper experiment. It provides constraint filtering, non-dominated
solution detection, and TOPSIS ranking. Cost and environmental scores are
calculated only from factors supplied by the caller; the package does not embed
universal prices or emission factors.
