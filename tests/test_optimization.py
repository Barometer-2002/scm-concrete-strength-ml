import pandas as pd

from scm_concrete_ml.optimization import Objective, apply_constraints, pareto_mask, topsis_score


def test_pareto_mask_excludes_dominated_candidate():
    frame = pd.DataFrame(
        {
            "strength": [50.0, 60.0, 55.0],
            "carbon": [300.0, 250.0, 280.0],
        }
    )
    mask = pareto_mask(
        frame,
        [Objective("strength", "max"), Objective("carbon", "min")],
    )
    assert mask.tolist() == [False, True, False]


def test_topsis_score_and_constraints_preserve_index():
    frame = pd.DataFrame(
        {
            "strength": [45.0, 55.0, 65.0],
            "cost": [200.0, 230.0, 300.0],
        },
        index=[10, 20, 30],
    )
    feasible = apply_constraints(frame, minimum={"strength": 50.0}, maximum={"cost": 280.0})
    scores = topsis_score(
        feasible,
        [Objective("strength", "max"), Objective("cost", "min")],
    )

    assert feasible.index.tolist() == [20]
    assert scores.index.tolist() == [20]
    assert scores.iloc[0] == 0.5
