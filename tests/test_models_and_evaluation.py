from scm_concrete_ml.data import generate_synthetic_mixtures
from scm_concrete_ml.evaluation import regression_metrics, repeated_cross_validate
from scm_concrete_ml.features import TARGET_COLUMN, engineer_mix_features
from scm_concrete_ml.models import get_model


def test_regression_metrics_for_exact_prediction():
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert metrics == {"r2": 1.0, "rmse": 0.0, "mae": 0.0}


def test_repeated_cross_validation_returns_every_fold():
    frame = generate_synthetic_mixtures(80, random_state=11)
    X = engineer_mix_features(frame)
    y = frame[TARGET_COLUMN]
    model = get_model("rf", params={"n_estimators": 30}, n_jobs=1)

    result = repeated_cross_validate(model, X, y, n_splits=4, n_repeats=2)

    assert len(result.folds) == 8
    assert set(result.summary) == {
        "r2_mean",
        "r2_std",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
    }
    assert result.summary["rmse_mean"] > 0
