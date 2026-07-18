"""Dataset loading and license-safe synthetic example generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import TARGET_COLUMN, engineer_mix_features


def load_dataset(
    path: str | Path,
    *,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a raw mixture CSV and return engineered inputs and numeric target."""

    frame = pd.read_csv(path)
    if target_column not in frame.columns:
        raise ValueError(f"Target column not found: {target_column}")

    target = pd.to_numeric(frame[target_column], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError(f"Target column contains missing or non-numeric values: {target_column}")

    features = engineer_mix_features(frame)
    return features, target.rename(target_column)


def generate_synthetic_mixtures(
    n_samples: int = 240,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate plausible-looking mixtures for software demos and tests.

    The generated values are artificial. They must not be used to draw material
    conclusions, calibrate a production model, or replace laboratory evidence.
    """

    if n_samples < 20:
        raise ValueError("n_samples must be at least 20")

    rng = np.random.default_rng(random_state)
    binder = rng.uniform(320.0, 600.0, n_samples)
    replacement = rng.uniform(0.04, 0.65, n_samples)
    shares = rng.dirichlet([2.4, 0.8, 2.0], n_samples)

    fa_ratio = replacement * shares[:, 0]
    sf_ratio = np.minimum(replacement * shares[:, 1], 0.14)
    ggbfs_ratio = np.maximum(replacement - fa_ratio - sf_ratio, 0.0)
    cement_ratio = 1.0 - fa_ratio - sf_ratio - ggbfs_ratio

    water_binder = rng.uniform(0.26, 0.62, n_samples)
    aggregate_binder = rng.uniform(2.5, 5.3, n_samples)
    sand_ratio = rng.uniform(0.32, 0.49, n_samples)
    sp_binder = rng.uniform(0.002, 0.021, n_samples)

    aggregate = binder * aggregate_binder
    fine_aggregate = aggregate * sand_ratio
    coarse_aggregate = aggregate - fine_aggregate

    strength = (
        106.0
        - 118.0 * water_binder
        + 0.052 * (binder - 350.0)
        - 18.0 * fa_ratio
        + 22.0 * sf_ratio
        + 9.0 * ggbfs_ratio
        + 16.0 * sf_ratio * (1.0 - water_binder)
        - 3.5 * np.maximum(aggregate_binder - 4.2, 0.0)
        + rng.normal(0.0, 5.0, n_samples)
    )
    strength = np.clip(strength, 8.0, 105.0)

    frame = pd.DataFrame(
        {
            "Cement": binder * cement_ratio,
            "Water": binder * water_binder,
            "Coarse aggregate": coarse_aggregate,
            "Fine aggregate": fine_aggregate,
            "FA": binder * fa_ratio,
            "SF": binder * sf_ratio,
            "GGBFS": binder * ggbfs_ratio,
            "SP": binder * sp_binder,
            TARGET_COLUMN: strength,
        }
    )
    return frame.round(6)
