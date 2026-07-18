"""Constraint filtering, Pareto screening, and TOPSIS ranking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Objective:
    """One optimization objective and its preferred direction."""

    column: str
    sense: Literal["min", "max"]


def _objective_matrix(frame: pd.DataFrame, objectives: list[Objective]) -> np.ndarray:
    if not objectives:
        raise ValueError("At least one objective is required")
    missing = [objective.column for objective in objectives if objective.column not in frame]
    if missing:
        raise ValueError(f"Objective columns not found: {', '.join(missing)}")

    values = frame[[objective.column for objective in objectives]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Objective columns must contain finite numeric values")
    signs = np.array([1.0 if objective.sense == "min" else -1.0 for objective in objectives])
    return values * signs


def pareto_mask(frame: pd.DataFrame, objectives: list[Objective]) -> np.ndarray:
    """Return a Boolean mask selecting non-dominated rows."""

    values = _objective_matrix(frame, objectives)
    efficient = np.ones(len(values), dtype=bool)
    for index, candidate in enumerate(values):
        if not efficient[index]:
            continue
        dominated_by_other = np.all(values <= candidate, axis=1) & np.any(
            values < candidate, axis=1
        )
        dominated_by_other[index] = False
        if dominated_by_other.any():
            efficient[index] = False
    return efficient


def topsis_score(
    frame: pd.DataFrame,
    objectives: list[Objective],
    *,
    weights: Mapping[str, float] | None = None,
) -> pd.Series:
    """Rank candidates by distance to ideal and anti-ideal objective points."""

    columns = [objective.column for objective in objectives]
    values = frame[columns].to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Candidate objective values must be non-empty and finite")

    denominators = np.linalg.norm(values, axis=0)
    denominators[denominators == 0.0] = 1.0
    normalized = values / denominators

    if weights is None:
        weight_array = np.full(len(objectives), 1.0 / len(objectives))
    else:
        weight_array = np.array(
            [weights[objective.column] for objective in objectives],
            dtype=float,
        )
        if np.any(weight_array < 0) or weight_array.sum() <= 0:
            raise ValueError("TOPSIS weights must be non-negative and sum to more than zero")
        weight_array = weight_array / weight_array.sum()
    weighted = normalized * weight_array

    ideal_best = np.empty(len(objectives))
    ideal_worst = np.empty(len(objectives))
    for index, objective in enumerate(objectives):
        if objective.sense == "max":
            ideal_best[index] = weighted[:, index].max()
            ideal_worst[index] = weighted[:, index].min()
        else:
            ideal_best[index] = weighted[:, index].min()
            ideal_worst[index] = weighted[:, index].max()

    distance_best = np.linalg.norm(weighted - ideal_best, axis=1)
    distance_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
    denominator = distance_best + distance_worst
    score = np.divide(
        distance_worst,
        denominator,
        out=np.full_like(distance_worst, 0.5),
        where=denominator > 0,
    )
    return pd.Series(score, index=frame.index, name="topsis_score")


def apply_constraints(
    frame: pd.DataFrame,
    *,
    minimum: Mapping[str, float] | None = None,
    maximum: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Filter candidate rows by inclusive lower and upper bounds."""

    mask = pd.Series(True, index=frame.index)
    for column, value in (minimum or {}).items():
        mask &= frame[column] >= value
    for column, value in (maximum or {}).items():
        mask &= frame[column] <= value
    return frame.loc[mask].copy()


def linear_impact(frame: pd.DataFrame, factors: Mapping[str, float]) -> pd.Series:
    """Calculate a linear cost or impact score from user-supplied factors."""

    missing = [column for column in factors if column not in frame]
    if missing:
        raise ValueError(f"Factor columns not found: {', '.join(missing)}")
    values = sum(frame[column].astype(float) * factor for column, factor in factors.items())
    return pd.Series(values, index=frame.index, dtype=float)
