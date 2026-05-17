"""Streamlit portfolio dashboard for the Olist customer-experience risk project."""

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

ARTIFACT_SCRIPTS = {
    "outputs/tables/time_validation_metrics.csv": "python scripts/run_time_validation.py",
    "outputs/tables/time_validation_with_history_metrics.csv": "python scripts/run_time_validation_with_history.py",
    "outputs/tables/intervention_simulation.csv": "python scripts/run_intervention_simulation.py",
    "outputs/tables/threshold_analysis.csv": "python scripts/run_intervention_simulation.py",
}

TIME_VALIDATION_PATH = TABLES_DIR / "time_validation_metrics.csv"
HISTORY_VALIDATION_PATH = TABLES_DIR / "time_validation_with_history_metrics.csv"
INTERVENTION_PATH = TABLES_DIR / "intervention_simulation.csv"
THRESHOLD_PATH = TABLES_DIR / "threshold_analysis.csv"


st.set_page_config(
    page_title="Olist CX Risk Dashboard",
    page_icon="O",
    layout="wide",
)


@st.cache_data
def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def show_missing_artifacts(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if not missing:
        return

    st.warning("Some artifacts are missing. Generate them from the project root before using this page.")
    for path in missing:
        relative_path = path.relative_to(PROJECT_ROOT)
        command = ARTIFACT_SCRIPTS.get(str(relative_path), "See README for the generating script.")
        st.code(command, language="bash")


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def format_lift(value: float) -> str:
    return f"{value:.2f}x"


def best_by_metric(df: pd.DataFrame, metric: str) -> pd.Series:
    return df.sort_values(metric, ascending=False).iloc[0]


def page_overview() -> None:
    st.title("Olist Customer-Experience Risk Prediction")
    st.markdown(
        """
        This project predicts whether an e-commerce order is at risk of receiving a low
        customer review before the review is submitted. The business use case is proactive
        intervention: identify orders most likely to become poor customer experiences while
        support or operations teams still have time to act.
        """
    )

    st.subheader("Target")
    st.markdown("`low_review = 1` when `review_score <= 2`; otherwise `low_review = 0` for scores 3-5.")

    st.subheader("Leakage-Safe Design")
    st.markdown(
        """
        The primary feature set excludes review text, review timestamps, current-order review
        outcomes, raw identifiers, and actual-delivery-derived fields. Historical risk features
        are computed only from strictly prior orders.
        """
    )

    intervention_df = load_csv(INTERVENTION_PATH)
    if intervention_df is not None and not intervention_df.empty:
        top_10 = intervention_df.loc[intervention_df["top_frac"].round(2) == 0.10].copy()
        if not top_10.empty:
            best = best_by_metric(top_10, "lift_over_random")
            st.subheader("Key Intervention Result")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Best model", best["model"])
            col2.metric("Orders flagged", f"{int(best['n_flagged']):,}")
            col3.metric("Low reviews caught", f"{int(best['true_positives']):,}")
            col4.metric("Lift", format_lift(best["lift_over_random"]))
            st.caption(
                f"At the top 10% intervention queue, precision was {format_pct(best['precision'])} "
                f"and recall was {format_pct(best['recall'])}."
            )
    else:
        show_missing_artifacts([INTERVENTION_PATH])


def page_validation_results() -> None:
    st.title("Validation Results")
    st.markdown(
        """
        Random train/test splits can be optimistic because historical and future orders are mixed.
        Time-aware validation is more realistic: train on earlier purchases and test on later purchases.
        """
    )

    time_df = load_csv(TIME_VALIDATION_PATH)
    history_df = load_csv(HISTORY_VALIDATION_PATH)
    show_missing_artifacts([TIME_VALIDATION_PATH, HISTORY_VALIDATION_PATH])

    if time_df is not None:
        st.subheader("Time-Aware Validation Without Historical Features")
        st.dataframe(time_df.round(4), use_container_width=True)
        best = best_by_metric(time_df, "roc_auc")
        st.info(f"Best ROC-AUC without history: {best['model']} ({best['roc_auc']:.4f}).")

    if history_df is not None:
        st.subheader("Time-Aware Validation With Historical Features")
        st.dataframe(history_df.round(4), use_container_width=True)
        best = best_by_metric(history_df, "roc_auc")
        st.info(f"Best ROC-AUC with history: {best['model']} ({best['roc_auc']:.4f}).")

    if time_df is not None and history_df is not None:
        metrics = ["roc_auc", "pr_auc_average_precision", "f1_class_1", "recall_class_1"]
        comparison = time_df[["model"] + metrics].merge(
            history_df[["model"] + metrics],
            on="model",
            suffixes=("_without_history", "_with_history"),
        )
        for metric in metrics:
            comparison[f"{metric}_delta"] = (
                comparison[f"{metric}_with_history"] - comparison[f"{metric}_without_history"]
            )
        st.subheader("Historical Feature Lift")
        st.dataframe(comparison.round(4), use_container_width=True)


def page_intervention_simulation() -> None:
    st.title("Intervention Simulation")
    st.markdown(
        """
        A support team usually needs a prioritized queue, not a default 0.5 classification threshold.
        This view evaluates what happens when the business flags the riskiest future orders.
        """
    )

    intervention_df = load_csv(INTERVENTION_PATH)
    show_missing_artifacts([INTERVENTION_PATH])
    if intervention_df is None:
        return

    model_names = sorted(intervention_df["model"].unique())
    selected_model = st.selectbox("Model", model_names)
    model_df = intervention_df.loc[intervention_df["model"] == selected_model].copy()

    top_10 = model_df.loc[model_df["top_frac"].round(2) == 0.10]
    if not top_10.empty:
        row = top_10.iloc[0]
        st.subheader("Top 10% Queue Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Orders flagged", f"{int(row['n_flagged']):,}")
        col2.metric("Low reviews caught", f"{int(row['true_positives']):,}")
        col3.metric("Precision", format_pct(row["precision"]))
        col4.metric("Recall", format_pct(row["recall"]))
        col5.metric("Lift", format_lift(row["lift_over_random"]))
        st.caption(f"Base low-review rate in test set: {format_pct(row['base_positive_rate'])}.")

    chart_df = model_df.sort_values("pct_orders_flagged").set_index("pct_orders_flagged")
    chart_df.index = chart_df.index * 100
    st.subheader("Queue Operating Curves")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("Precision")
        st.line_chart(chart_df[["precision"]])
    with col2:
        st.markdown("Recall")
        st.line_chart(chart_df[["recall"]])
    with col3:
        st.markdown("Lift")
        st.line_chart(chart_df[["lift_over_random"]])

    st.subheader("Intervention Table")
    st.dataframe(model_df.round(4), use_container_width=True)


def page_threshold_analysis() -> None:
    st.title("Threshold Analysis")
    st.markdown(
        """
        Lower probability thresholds catch more at-risk orders, but they also create more false
        positives and larger intervention queues. Higher thresholds focus capacity on fewer,
        higher-risk orders.
        """
    )

    threshold_df = load_csv(THRESHOLD_PATH)
    show_missing_artifacts([THRESHOLD_PATH])
    if threshold_df is None:
        return

    model_names = sorted(threshold_df["model"].unique())
    selected_model = st.selectbox("Model", model_names)
    model_df = threshold_df.loc[threshold_df["model"] == selected_model].copy()
    st.dataframe(model_df.round(4), use_container_width=True)

    chart_df = model_df.sort_values("probability_threshold").set_index("probability_threshold")
    st.subheader("Threshold Tradeoffs")
    st.line_chart(chart_df[["precision", "recall", "lift_over_random"]])


def page_reproduce() -> None:
    st.title("How to Reproduce")
    st.markdown("Place the raw Olist CSV files in `olist_data/`, then run the commands below from the project root.")
    commands = [
        "make baseline",
        "make time-validation",
        "make time-validation-history",
        "make intervention",
        "make test",
    ]
    for command in commands:
        st.code(command, language="bash")

    st.subheader("Artifact Scripts")
    st.markdown(
        """
        If a dashboard page reports missing artifacts, generate them with:

        - `python scripts/run_time_validation.py`
        - `python scripts/run_time_validation_with_history.py`
        - `python scripts/run_intervention_simulation.py`
        """
    )


PAGES = {
    "Project Overview": page_overview,
    "Validation Results": page_validation_results,
    "Intervention Simulation": page_intervention_simulation,
    "Threshold Analysis": page_threshold_analysis,
    "How to Reproduce": page_reproduce,
}


def main() -> None:
    st.sidebar.title("Olist CX Risk")
    page_name = st.sidebar.radio("Navigate", list(PAGES.keys()))
    PAGES[page_name]()


if __name__ == "__main__":
    main()
