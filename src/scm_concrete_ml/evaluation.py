"""Regression evaluation with repeatable cross-validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold


@dataclass(frozen=True)
class CrossValidationResult:
    """Fold-level scores and aggregate statistics from repeated CV."""

    folds: pd.DataFrame
    summary: dict[str, float]


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return R2, RMSE, and MAE for point predictions."""

    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "mae": float(mean_absolute_error(truth, prediction)),
    }


def repeated_cross_validate(
    estimator: RegressorMixin,
    X,
    y,
    *,
    n_splits: int = 5,
    n_repeats: int = 2,
    random_state: int = 42,
) -> CrossValidationResult:
    """Evaluate an estimator under shuffled repeated K-fold validation."""

    if n_splits < 2 or n_repeats < 1:
        raise ValueError("n_splits must be >= 2 and n_repeats must be >= 1")

    X_array = np.asarray(X)
    y_array = np.asarray(y, dtype=float)
    splitter = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    rows: list[dict[str, float | int]] = []
    for fold_number, (train_index, valid_index) in enumerate(
        splitter.split(X_array), start=1
    ):
        model = clone(estimator)
        model.fit(X_array[train_index], y_array[train_index])
        prediction = model.predict(X_array[valid_index])
        rows.append({"fold": fold_number, **regression_metrics(y_array[valid_index], prediction)})

    folds = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    for metric in ("r2", "rmse", "mae"):
        summary[f"{metric}_mean"] = float(folds[metric].mean())
        summary[f"{metric}_std"] = float(folds[metric].std(ddof=1))
    return CrossValidationResult(folds=folds, summary=summary)
