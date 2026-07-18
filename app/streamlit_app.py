"""Interactive decision-support demo using license-safe synthetic data."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from scm_concrete_ml.data import generate_synthetic_mixtures, load_dataset
from scm_concrete_ml.features import TARGET_COLUMN, engineer_mix_features
from scm_concrete_ml.models import get_model
from scm_concrete_ml.optimization import (
    Objective,
    apply_constraints,
    linear_impact,
    pareto_mask,
    topsis_score,
)
from scm_concrete_ml.uncertainty import KNNResidualScaleConformalRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_example.csv"

st.set_page_config(page_title="SCM Concrete ML", page_icon=":material/analytics:", layout="wide")


@st.cache_data
def load_public_example() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def fit_public_model() -> KNNResidualScaleConformalRegressor:
    X, y = load_dataset(DATA_PATH)
    return KNNResidualScaleConformalRegressor(
        estimator=get_model("rf", random_state=42),
        alpha=0.1,
        k_neighbors=18,
        random_state=42,
    ).fit(X, y)


def dosage_input(label: str, minimum: float, maximum: float, default: float) -> float:
    return float(
        st.number_input(label, min_value=minimum, max_value=maximum, value=default, step=1.0)
    )


frame = load_public_example()
model = fit_public_model()

st.title("SCM concrete strength decision support")
st.caption("Public demo dataset: synthetic values for software validation only")

overview_tab, prediction_tab, screening_tab = st.tabs(
    ["Model overview", "Single mixture", "Candidate screening"]
)

with overview_tab:
    metric_columns = st.columns(4)
    metric_columns[0].metric("Example records", f"{len(frame):,}")
    metric_columns[1].metric("Input features", "8")
    metric_columns[2].metric("Interval target", "90%")
    metric_columns[3].metric("Model", "Random Forest")

    engineered = engineer_mix_features(frame)
    chart_frame = pd.concat([engineered, frame[[TARGET_COLUMN]]], axis=1)
    x_column = st.selectbox("Horizontal variable", list(engineered.columns), index=1)
    figure = px.scatter(
        chart_frame,
        x=x_column,
        y=TARGET_COLUMN,
        color="FA/B",
        color_continuous_scale="Viridis",
        opacity=0.72,
    )
    figure.update_layout(height=470, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(figure, use_container_width=True)

with prediction_tab:
    first, second, third = st.columns(3)
    with first:
        cement = dosage_input("Cement", 100.0, 700.0, 300.0)
        water = dosage_input("Water", 80.0, 350.0, 165.0)
        sp = dosage_input("Superplasticizer", 0.0, 30.0, 5.0)
    with second:
        fa = dosage_input("Fly ash", 0.0, 350.0, 80.0)
        sf = dosage_input("Silica fume", 0.0, 100.0, 20.0)
        ggbfs = dosage_input("GGBFS", 0.0, 400.0, 100.0)
    with third:
        coarse = dosage_input("Coarse aggregate", 400.0, 1_500.0, 980.0)
        fine = dosage_input("Fine aggregate", 300.0, 1_200.0, 720.0)

    mixture = pd.DataFrame(
        [
            {
                "Cement": cement,
                "Water": water,
                "Coarse aggregate": coarse,
                "Fine aggregate": fine,
                "FA": fa,
                "SF": sf,
                "GGBFS": ggbfs,
                "SP": sp,
            }
        ]
    )
    mixture_features = engineer_mix_features(mixture)
    prediction, lower, upper = model.predict_interval(mixture_features)
    output_columns = st.columns(3)
    output_columns[0].metric("Predicted strength", f"{prediction[0]:.1f} MPa")
    output_columns[1].metric("90% lower bound", f"{lower[0]:.1f} MPa")
    output_columns[2].metric("90% upper bound", f"{upper[0]:.1f} MPa")
    st.dataframe(mixture_features.round(4), use_container_width=True, hide_index=True)

with screening_tab:
    factor_column, filter_column = st.columns([2, 1])
    with factor_column:
        st.subheader("Scenario factors")
        carbon_inputs = st.columns(4)
        carbon_factors = {
            "Cement": carbon_inputs[0].number_input("Cement CO2 factor", value=0.82),
            "FA": carbon_inputs[1].number_input("FA CO2 factor", value=0.02),
            "SF": carbon_inputs[2].number_input("SF CO2 factor", value=0.03),
            "GGBFS": carbon_inputs[3].number_input("GGBFS CO2 factor", value=0.07),
        }
        cost_inputs = st.columns(4)
        cost_factors = {
            "Cement": cost_inputs[0].number_input("Cement cost factor", value=0.42),
            "FA": cost_inputs[1].number_input("FA cost factor", value=0.18),
            "SF": cost_inputs[2].number_input("SF cost factor", value=1.20),
            "GGBFS": cost_inputs[3].number_input("GGBFS cost factor", value=0.28),
        }
    with filter_column:
        st.subheader("Constraint")
        minimum_strength = st.slider("Minimum 90% lower bound", 10.0, 80.0, 35.0, 1.0)

    candidates = generate_synthetic_mixtures(400, random_state=87).drop(columns=[TARGET_COLUMN])
    candidate_features = engineer_mix_features(candidates)
    point, lower, upper = model.predict_interval(candidate_features)
    candidates["Predicted strength"] = point
    candidates["Lower 90"] = lower
    candidates["Upper 90"] = upper
    candidates["SCM replacement"] = (
        candidates[["FA", "SF", "GGBFS"]].sum(axis=1)
        / candidates[["Cement", "FA", "SF", "GGBFS"]].sum(axis=1)
    )
    candidates["Carbon score"] = linear_impact(candidates, carbon_factors)
    candidates["Cost score"] = linear_impact(candidates, cost_factors)
    feasible = apply_constraints(candidates, minimum={"Lower 90": minimum_strength})

    objectives = [
        Objective("Lower 90", "max"),
        Objective("Carbon score", "min"),
        Objective("Cost score", "min"),
        Objective("SCM replacement", "max"),
    ]
    feasible["Pareto"] = pareto_mask(feasible, objectives)
    pareto = feasible.loc[feasible["Pareto"]].copy()
    if not pareto.empty:
        pareto["TOPSIS"] = topsis_score(pareto, objectives)
        pareto = pareto.sort_values("TOPSIS", ascending=False)

    status_columns = st.columns(3)
    status_columns[0].metric("Generated", f"{len(candidates):,}")
    status_columns[1].metric("Feasible", f"{len(feasible):,}")
    status_columns[2].metric("Pareto candidates", f"{len(pareto):,}")

    if pareto.empty:
        st.warning("No candidate satisfies the current constraint.")
    else:
        figure = px.scatter(
            feasible,
            x="Carbon score",
            y="Lower 90",
            size="Cost score",
            color="SCM replacement",
            symbol="Pareto",
            color_continuous_scale="Cividis",
            hover_data=["Predicted strength", "Upper 90"],
        )
        figure.update_layout(height=520, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(figure, use_container_width=True)
        display_columns = [
            "Predicted strength",
            "Lower 90",
            "Upper 90",
            "Carbon score",
            "Cost score",
            "SCM replacement",
            "TOPSIS",
        ]
        st.dataframe(
            pareto[display_columns].head(20).round(4),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Scenario factors are editable demonstration inputs, not certified prices or EPD data."
    )
