"""Reusable tools for SCM concrete strength modelling and decision support."""

from .data import generate_synthetic_mixtures, load_dataset
from .evaluation import CrossValidationResult, regression_metrics, repeated_cross_validate
from .features import ENGINEERED_FEATURES, RAW_FEATURES, TARGET_COLUMN, engineer_mix_features
from .models import get_model
from .optimization import Objective, pareto_mask, topsis_score
from .uncertainty import (
    KNNResidualScaleConformalRegressor,
    ResidualScaleConformalRegressor,
    SplitConformalRegressor,
    interval_metrics,
)

__all__ = [
    "CrossValidationResult",
    "ENGINEERED_FEATURES",
    "KNNResidualScaleConformalRegressor",
    "Objective",
    "RAW_FEATURES",
    "ResidualScaleConformalRegressor",
    "SplitConformalRegressor",
    "TARGET_COLUMN",
    "engineer_mix_features",
    "generate_synthetic_mixtures",
    "get_model",
    "interval_metrics",
    "load_dataset",
    "pareto_mask",
    "regression_metrics",
    "repeated_cross_validate",
    "topsis_score",
]

__version__ = "0.1.0"
