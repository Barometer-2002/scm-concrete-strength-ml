"""Split conformal prediction with optional residual and KNN scale models."""

from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def conformal_quantile(scores, *, alpha: float) -> float:
    """Return the finite-sample corrected conformal quantile."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite one-dimensional array")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    level = min(1.0, math.ceil((values.size + 1) * (1.0 - alpha)) / values.size)
    return float(np.quantile(values, level, method="higher"))


def interval_metrics(y_true, lower, upper) -> dict[str, float]:
    """Measure empirical coverage and interval width."""

    truth = np.asarray(y_true, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if truth.shape != low.shape or truth.shape != high.shape:
        raise ValueError("y_true, lower, and upper must have the same shape")
    if np.any(high < low):
        raise ValueError("upper interval bounds must not be below lower bounds")

    covered = (truth >= low) & (truth <= high)
    widths = high - low
    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
    }


def _default_point_estimator(random_state: int):
    return RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )


def _default_scale_estimator(random_state: int):
    return RandomForestRegressor(
        n_estimators=180,
        min_samples_leaf=4,
        random_state=random_state,
        n_jobs=-1,
    )


class SplitConformalRegressor(BaseEstimator, RegressorMixin):
    """Fixed-width split conformal regressor."""

    def __init__(
        self,
        estimator=None,
        *,
        alpha: float = 0.1,
        calibration_size: float = 0.2,
        random_state: int = 42,
    ):
        self.estimator = estimator
        self.alpha = alpha
        self.calibration_size = calibration_size
        self.random_state = random_state

    def fit(self, X, y):
        X_array, y_array = check_X_y(X, y, dtype=float)
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_array,
            y_array,
            test_size=self.calibration_size,
            random_state=self.random_state,
        )
        template = (
            self.estimator
            if self.estimator is not None
            else _default_point_estimator(self.random_state)
        )
        self.estimator_ = clone(template).fit(X_fit, y_fit)
        residuals = np.abs(y_cal - self.estimator_.predict(X_cal))
        self.quantile_ = conformal_quantile(residuals, alpha=self.alpha)
        self.n_features_in_ = X_array.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, ("estimator_", "quantile_"))
        return self.estimator_.predict(check_array(X, dtype=float))

    def predict_interval(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prediction = self.predict(X)
        return prediction, prediction - self.quantile_, prediction + self.quantile_


class ResidualScaleConformalRegressor(BaseEstimator, RegressorMixin):
    """Residual-scaled split conformal regressor.

    Out-of-fold point-model residuals train a secondary scale model. Calibration
    residuals are normalized by that estimated local scale before the conformal
    quantile is calculated.
    """

    def __init__(
        self,
        estimator=None,
        scale_estimator=None,
        *,
        alpha: float = 0.1,
        calibration_size: float = 0.2,
        scale_cv: int = 5,
        min_scale: float = 1e-3,
        random_state: int = 42,
    ):
        self.estimator = estimator
        self.scale_estimator = scale_estimator
        self.alpha = alpha
        self.calibration_size = calibration_size
        self.scale_cv = scale_cv
        self.min_scale = min_scale
        self.random_state = random_state

    def _fit_scale_context(self, X_fit: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
        self.X_scale_reference_ = X_fit
        self.y_scale_reference_ = y_fit
        return X_fit

    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        return X

    def fit(self, X, y):
        X_array, y_array = check_X_y(X, y, dtype=float)
        if self.scale_cv < 2:
            raise ValueError("scale_cv must be at least 2")
        if self.min_scale <= 0:
            raise ValueError("min_scale must be greater than zero")

        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_array,
            y_array,
            test_size=self.calibration_size,
            random_state=self.random_state,
        )
        point_template = (
            self.estimator
            if self.estimator is not None
            else _default_point_estimator(self.random_state)
        )
        cv = KFold(n_splits=self.scale_cv, shuffle=True, random_state=self.random_state)
        oof_prediction = cross_val_predict(clone(point_template), X_fit, y_fit, cv=cv)
        scale_target = np.abs(y_fit - oof_prediction)

        scale_fit_features = self._fit_scale_context(X_fit, y_fit)
        scale_template = (
            self.scale_estimator
            if self.scale_estimator is not None
            else _default_scale_estimator(self.random_state)
        )
        self.scale_estimator_ = clone(scale_template).fit(scale_fit_features, scale_target)
        self.estimator_ = clone(point_template).fit(X_fit, y_fit)

        scale_cal = np.maximum(
            self.scale_estimator_.predict(self._scale_features(X_cal)),
            self.min_scale,
        )
        residual_cal = np.abs(y_cal - self.estimator_.predict(X_cal))
        self.quantile_ = conformal_quantile(residual_cal / scale_cal, alpha=self.alpha)
        self.n_features_in_ = X_array.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, ("estimator_", "scale_estimator_", "quantile_"))
        return self.estimator_.predict(check_array(X, dtype=float))

    def predict_scale(self, X) -> np.ndarray:
        check_is_fitted(self, ("scale_estimator_", "quantile_"))
        X_array = check_array(X, dtype=float)
        return np.maximum(
            self.scale_estimator_.predict(self._scale_features(X_array)),
            self.min_scale,
        )

    def predict_interval(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prediction = self.predict(X)
        width = self.quantile_ * self.predict_scale(X)
        return prediction, prediction - width, prediction + width


class KNNResidualScaleConformalRegressor(ResidualScaleConformalRegressor):
    """KNN-enhanced residual-scaled conformal regressor (KNN-RSCP).

    The residual scale model receives the original engineered inputs plus local
    mean neighbor distance and neighbor-target standard deviation.
    """

    def __init__(
        self,
        estimator=None,
        scale_estimator=None,
        *,
        alpha: float = 0.1,
        calibration_size: float = 0.2,
        scale_cv: int = 5,
        min_scale: float = 1e-3,
        k_neighbors: int = 20,
        random_state: int = 42,
    ):
        super().__init__(
            estimator=estimator,
            scale_estimator=scale_estimator,
            alpha=alpha,
            calibration_size=calibration_size,
            scale_cv=scale_cv,
            min_scale=min_scale,
            random_state=random_state,
        )
        self.k_neighbors = k_neighbors

    def _fit_scale_context(self, X_fit: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
        if self.k_neighbors < 2:
            raise ValueError("k_neighbors must be at least 2")
        if self.k_neighbors >= len(X_fit):
            raise ValueError("k_neighbors must be smaller than the fitting subset")

        self.X_scale_reference_ = X_fit
        self.y_scale_reference_ = y_fit
        self.knn_scaler_ = StandardScaler().fit(X_fit)
        scaled_fit = self.knn_scaler_.transform(X_fit)
        self.neighbors_ = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(scaled_fit)

        distances, indices = self.neighbors_.kneighbors(scaled_fit)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        return self._append_local_features(X_fit, distances, indices)

    def _append_local_features(
        self,
        X: np.ndarray,
        distances: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        local_distance = distances.mean(axis=1)
        local_target_std = self.y_scale_reference_[indices].std(axis=1, ddof=0)
        return np.column_stack([X, local_distance, local_target_std])

    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        scaled = self.knn_scaler_.transform(X)
        distances, indices = self.neighbors_.kneighbors(
            scaled,
            n_neighbors=self.k_neighbors,
        )
        return self._append_local_features(X, distances, indices)
