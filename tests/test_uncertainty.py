import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from scm_concrete_ml.data import generate_synthetic_mixtures
from scm_concrete_ml.features import TARGET_COLUMN, engineer_mix_features
from scm_concrete_ml.uncertainty import (
    KNNResidualScaleConformalRegressor,
    conformal_quantile,
    interval_metrics,
)


def test_conformal_quantile_uses_finite_sample_correction():
    scores = np.arange(1.0, 11.0)
    assert conformal_quantile(scores, alpha=0.2) == 10.0


def test_knn_rscp_returns_ordered_adaptive_intervals():
    frame = generate_synthetic_mixtures(200, random_state=23)
    X = engineer_mix_features(frame)
    y = frame[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=19
    )
    regressor = KNNResidualScaleConformalRegressor(
        estimator=RandomForestRegressor(n_estimators=80, random_state=19, n_jobs=1),
        scale_estimator=RandomForestRegressor(
            n_estimators=60,
            min_samples_leaf=3,
            random_state=19,
            n_jobs=1,
        ),
        alpha=0.1,
        scale_cv=3,
        k_neighbors=10,
        random_state=19,
    ).fit(X_train, y_train)

    prediction, lower, upper = regressor.predict_interval(X_test)
    metrics = interval_metrics(y_test, lower, upper)

    assert prediction.shape == lower.shape == upper.shape == (len(X_test),)
    assert np.all(lower <= prediction)
    assert np.all(prediction <= upper)
    assert np.std(upper - lower) > 0
    assert metrics["coverage"] >= 0.65
