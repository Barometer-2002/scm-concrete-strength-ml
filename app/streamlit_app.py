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
FEATURE_LABELS = {
    "B": "胶凝材料总量 B（kg/m3）",
    "W/B": "水胶比 W/B",
    "A/B": "骨胶比 A/B",
    "SR": "砂率 SR",
    "SP/B": "减水剂胶凝材料比 SP/B",
    "FA/B": "粉煤灰替代率 FA/B",
    "SF/B": "硅灰替代率 SF/B",
    "GGBFS/B": "矿渣粉替代率 GGBFS/B",
}

st.set_page_config(
    page_title="SCM 混凝土强度决策支持",
    page_icon=":material/analytics:",
    layout="wide",
)


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

st.title("SCM 混凝土强度决策支持")
st.caption("当前使用公开合成数据，仅用于软件功能验证")

overview_tab, prediction_tab, screening_tab = st.tabs(
    ["模型概览", "单组配比预测", "候选方案筛选"]
)

with overview_tab:
    metric_columns = st.columns(4)
    metric_columns[0].metric("示例数据量", f"{len(frame):,}")
    metric_columns[1].metric("输入特征数", "8")
    metric_columns[2].metric("区间目标覆盖率", "90%")
    metric_columns[3].metric("当前模型", "随机森林")

    engineered = engineer_mix_features(frame)
    chart_frame = pd.concat([engineered, frame[[TARGET_COLUMN]]], axis=1)
    x_column = st.selectbox(
        "横轴变量",
        list(engineered.columns),
        index=1,
        format_func=FEATURE_LABELS.get,
    )
    figure = px.scatter(
        chart_frame,
        x=x_column,
        y=TARGET_COLUMN,
        color="FA/B",
        color_continuous_scale="Viridis",
        opacity=0.72,
        labels={
            **FEATURE_LABELS,
            TARGET_COLUMN: "圆柱体抗压强度（MPa）",
        },
    )
    figure.update_layout(height=470, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(figure, use_container_width=True)

with prediction_tab:
    first, second, third = st.columns(3)
    with first:
        cement = dosage_input("水泥用量（kg/m3）", 100.0, 700.0, 300.0)
        water = dosage_input("用水量（kg/m3）", 80.0, 350.0, 165.0)
        sp = dosage_input("减水剂用量（kg/m3）", 0.0, 30.0, 5.0)
    with second:
        fa = dosage_input("粉煤灰用量（kg/m3）", 0.0, 350.0, 80.0)
        sf = dosage_input("硅灰用量（kg/m3）", 0.0, 100.0, 20.0)
        ggbfs = dosage_input("矿渣粉用量（kg/m3）", 0.0, 400.0, 100.0)
    with third:
        coarse = dosage_input("粗骨料用量（kg/m3）", 400.0, 1_500.0, 980.0)
        fine = dosage_input("细骨料用量（kg/m3）", 300.0, 1_200.0, 720.0)

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
    output_columns[0].metric("预测抗压强度", f"{prediction[0]:.1f} MPa")
    output_columns[1].metric("90%预测区间下界", f"{lower[0]:.1f} MPa")
    output_columns[2].metric("90%预测区间上界", f"{upper[0]:.1f} MPa")
    st.dataframe(
        mixture_features.rename(columns=FEATURE_LABELS).round(4),
        use_container_width=True,
        hide_index=True,
    )

with screening_tab:
    factor_column, filter_column = st.columns([2, 1])
    with factor_column:
        st.subheader("场景参数")
        carbon_inputs = st.columns(4)
        carbon_factors = {
            "Cement": carbon_inputs[0].number_input("水泥碳排系数", value=0.82),
            "FA": carbon_inputs[1].number_input("粉煤灰碳排系数", value=0.02),
            "SF": carbon_inputs[2].number_input("硅灰碳排系数", value=0.03),
            "GGBFS": carbon_inputs[3].number_input("矿渣粉碳排系数", value=0.07),
        }
        cost_inputs = st.columns(4)
        cost_factors = {
            "Cement": cost_inputs[0].number_input("水泥成本系数", value=0.42),
            "FA": cost_inputs[1].number_input("粉煤灰成本系数", value=0.18),
            "SF": cost_inputs[2].number_input("硅灰成本系数", value=1.20),
            "GGBFS": cost_inputs[3].number_input("矿渣粉成本系数", value=0.28),
        }
    with filter_column:
        st.subheader("筛选约束")
        minimum_strength = st.slider("90%预测区间最低下界（MPa）", 10.0, 80.0, 35.0, 1.0)

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
    feasible["方案类型"] = feasible["Pareto"].map({True: "帕累托解", False: "可行解"})
    pareto = feasible.loc[feasible["Pareto"]].copy()
    if not pareto.empty:
        pareto["TOPSIS"] = topsis_score(pareto, objectives)
        pareto = pareto.sort_values("TOPSIS", ascending=False)

    status_columns = st.columns(3)
    status_columns[0].metric("生成方案数", f"{len(candidates):,}")
    status_columns[1].metric("满足约束方案数", f"{len(feasible):,}")
    status_columns[2].metric("帕累托方案数", f"{len(pareto):,}")

    if pareto.empty:
        st.warning("没有候选方案满足当前约束。")
    else:
        figure = px.scatter(
            feasible,
            x="Carbon score",
            y="Lower 90",
            size="Cost score",
            color="SCM replacement",
            symbol="方案类型",
            color_continuous_scale="Cividis",
            hover_data=["Predicted strength", "Upper 90"],
            labels={
                "Carbon score": "碳排放估算值",
                "Lower 90": "90%预测区间下界（MPa）",
                "Cost score": "成本估算值",
                "SCM replacement": "SCM总替代率",
                "Predicted strength": "预测抗压强度（MPa）",
                "Upper 90": "90%预测区间上界（MPa）",
            },
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
            pareto[display_columns]
            .head(20)
            .rename(
                columns={
                    "Predicted strength": "预测抗压强度（MPa）",
                    "Lower 90": "90%区间下界（MPa）",
                    "Upper 90": "90%区间上界（MPa）",
                    "Carbon score": "碳排放估算值",
                    "Cost score": "成本估算值",
                    "SCM replacement": "SCM总替代率",
                    "TOPSIS": "综合排序得分",
                }
            )
            .round(4),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "场景系数为可编辑的演示参数，不代表经核验的市场价格或环境产品声明数据。"
    )
