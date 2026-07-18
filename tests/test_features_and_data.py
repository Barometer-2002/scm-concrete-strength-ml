import pandas as pd
import pytest

from scm_concrete_ml.data import generate_synthetic_mixtures
from scm_concrete_ml.features import (
    ENGINEERED_FEATURES,
    TARGET_COLUMN,
    FeatureValidationError,
    engineer_mix_features,
)


def test_engineer_mix_features_matches_expected_ratios():
    frame = pd.DataFrame(
        [
            {
                "Cement": 300.0,
                "Water": 150.0,
                "Coarse aggregate": 1_000.0,
                "Fine aggregate": 700.0,
                "FA": 100.0,
                "SF": 20.0,
                "GGBFS": 80.0,
                "SP": 5.0,
            }
        ]
    )
    result = engineer_mix_features(frame)

    assert tuple(result.columns) == ENGINEERED_FEATURES
    assert result.loc[0, "B"] == pytest.approx(500.0)
    assert result.loc[0, "W/B"] == pytest.approx(0.3)
    assert result.loc[0, "A/B"] == pytest.approx(3.4)
    assert result.loc[0, "SR"] == pytest.approx(700.0 / 1_700.0)
    assert result.loc[0, "FA/B"] == pytest.approx(0.2)


def test_engineer_mix_features_rejects_zero_binder():
    frame = pd.DataFrame(
        [
            {
                "Cement": 0,
                "Water": 150,
                "Coarse aggregate": 1_000,
                "Fine aggregate": 700,
                "FA": 0,
                "SF": 0,
                "GGBFS": 0,
                "SP": 0,
            }
        ]
    )
    with pytest.raises(FeatureValidationError, match="binder"):
        engineer_mix_features(frame)


def test_synthetic_data_has_required_schema_and_is_deterministic():
    first = generate_synthetic_mixtures(30, random_state=7)
    second = generate_synthetic_mixtures(30, random_state=7)

    pd.testing.assert_frame_equal(first, second)
    assert TARGET_COLUMN in first
    assert len(first) == 30
    assert first[TARGET_COLUMN].between(8.0, 105.0).all()
