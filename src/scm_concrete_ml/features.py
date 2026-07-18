"""Feature engineering for SCM concrete mixture records."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

RAW_FEATURES = (
    "Cement",
    "Water",
    "Coarse aggregate",
    "Fine aggregate",
    "FA",
    "SF",
    "GGBFS",
    "SP",
)

ENGINEERED_FEATURES = (
    "B",
    "W/B",
    "A/B",
    "SR",
    "SP/B",
    "FA/B",
    "SF/B",
    "GGBFS/B",
)

TARGET_COLUMN = "Cylinder compressive strength"


class FeatureValidationError(ValueError):
    """Raised when a mixture cannot be converted into valid ratio features."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FeatureValidationError(f"Missing required columns: {', '.join(missing)}")


def engineer_mix_features(frame: pd.DataFrame, *, keep_raw: bool = False) -> pd.DataFrame:
    """Convert raw component dosages into eight engineering features.

    Component dosages are expected in consistent mass-per-volume units. The
    function validates only arithmetic consistency; it does not certify that a
    mixture is physically feasible or compliant with a design standard.
    """

    _require_columns(frame, RAW_FEATURES)
    raw = frame.loc[:, RAW_FEATURES].apply(pd.to_numeric, errors="coerce")

    if raw.isna().any().any():
        bad_columns = raw.columns[raw.isna().any()].tolist()
        raise FeatureValidationError(
            f"Raw mixture columns contain missing or non-numeric values: {', '.join(bad_columns)}"
        )
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise FeatureValidationError("Raw mixture columns must contain finite values")
    if (raw < 0).any().any():
        raise FeatureValidationError("Component dosages cannot be negative")

    binder = raw[["Cement", "FA", "SF", "GGBFS"]].sum(axis=1)
    aggregate = raw[["Coarse aggregate", "Fine aggregate"]].sum(axis=1)
    if (binder <= 0).any():
        raise FeatureValidationError("Total binder content must be greater than zero")
    if (aggregate <= 0).any():
        raise FeatureValidationError("Total aggregate content must be greater than zero")

    engineered = pd.DataFrame(
        {
            "B": binder,
            "W/B": raw["Water"] / binder,
            "A/B": aggregate / binder,
            "SR": raw["Fine aggregate"] / aggregate,
            "SP/B": raw["SP"] / binder,
            "FA/B": raw["FA"] / binder,
            "SF/B": raw["SF"] / binder,
            "GGBFS/B": raw["GGBFS"] / binder,
        },
        index=frame.index,
    )

    if keep_raw:
        passthrough = frame.drop(columns=list(ENGINEERED_FEATURES), errors="ignore").copy()
        return pd.concat([passthrough, engineered], axis=1)
    return engineered.loc[:, ENGINEERED_FEATURES]
