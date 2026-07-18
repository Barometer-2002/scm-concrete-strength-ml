"""Model registry, benchmarking, and optional Optuna tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .evaluation import repeated_cross_validate

MODEL_NAMES = ("rf", "xgb", "lgbm", "svr", "mlp")


@dataclass(frozen=True)
class TuningResult:
    """Best parameters and CV RMSE from an Optuna study."""

    model_name: str
    best_params: dict[str, Any]
    best_rmse: float
    study: Any


def get_model(
    name: str,
    *,
    random_state: int = 42,
    n_jobs: int = -1,
    params: dict[str, Any] | None = None,
):
    """Create one of the five model families used in the study."""

    name = name.lower().strip()
    override = dict(params or {})

    if name == "rf":
        defaults = {
            "n_estimators": 300,
            "min_samples_leaf": 1,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        return RandomForestRegressor(**(defaults | override))
    if name == "svr":
        defaults = {"kernel": "rbf", "C": 80.0, "epsilon": 0.1, "gamma": "scale"}
        return make_pipeline(StandardScaler(), SVR(**(defaults | override)))
    if name == "mlp":
        defaults = {
            "hidden_layer_sizes": (96, 64),
            "early_stopping": True,
            "max_iter": 600,
            "random_state": random_state,
        }
        return make_pipeline(StandardScaler(), MLPRegressor(**(defaults | override)))
    if name == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("Install the 'boosters' extra to use XGBoost") from exc
        defaults = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 5,
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        return XGBRegressor(**(defaults | override))
    if name == "lgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("Install the 'boosters' extra to use LightGBM") from exc
        defaults = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 48,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbose": -1,
        }
        return LGBMRegressor(**(defaults | override))
    raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_NAMES)}")


def benchmark_models(
    X,
    y,
    *,
    model_names: tuple[str, ...] = ("rf", "svr", "mlp"),
    n_splits: int = 5,
    n_repeats: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare model families under one repeated-CV protocol."""

    rows = []
    for name in model_names:
        result = repeated_cross_validate(
            get_model(name, random_state=random_state),
            X,
            y,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        rows.append({"model": name, **result.summary})
    return pd.DataFrame(rows).sort_values("rmse_mean", ignore_index=True)


def _suggest_params(trial, name: str) -> dict[str, Any]:
    if name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 16, 20]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "max_features": trial.suggest_categorical("max_features", [1.0, "sqrt", "log2"]),
        }
    if name == "svr":
        return {
            "C": trial.suggest_float("C", 10.0, 300.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.02, 0.5, log=True),
            "gamma": trial.suggest_float("gamma", 0.01, 1.0, log=True),
        }
    if name == "mlp":
        width = trial.suggest_int("width", 48, 160, step=16)
        layers = trial.suggest_int("layers", 1, 3)
        return {
            "hidden_layer_sizes": tuple([width] * layers),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 3e-4, 5e-3, log=True
            ),
        }
    if name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }
    if name == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 80),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        }
    raise ValueError(f"Unsupported model for tuning: {name}")


def tune_model(
    name: str,
    X,
    y,
    *,
    n_trials: int = 50,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
) -> TuningResult:
    """Tune a model with Optuna by minimizing repeated-CV RMSE."""

    name = name.lower().strip()
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Install the 'tuning' extra to use Optuna") from exc

    X_array = np.asarray(X)
    y_array = np.asarray(y, dtype=float)
    cv = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    def objective(trial) -> float:
        model = get_model(
            name,
            random_state=random_state,
            n_jobs=1,
            params=_suggest_params(trial, name),
        )
        scores = cross_val_score(
            model,
            X_array,
            y_array,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
        )
        return float(-scores.mean())

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)
    params = _suggest_params(optuna.trial.FixedTrial(study.best_params), name)
    return TuningResult(
        model_name=name,
        best_params=params,
        best_rmse=float(study.best_value),
        study=study,
    )
