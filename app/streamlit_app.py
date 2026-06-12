"""Streamlit portfolio dashboard for the Olist customer-experience risk project."""

import json
from pathlib import Path
import subprocess
import sys
from typing import Optional

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cx_risk.business_value import calculate_business_value

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

ARTIFACT_SCRIPTS = {
    "outputs/tables/time_validation_metrics.csv": "python scripts/run_time_validation.py",
    "outputs/tables/time_validation_with_history_metrics.csv": "python scripts/run_time_validation_with_history.py",
    "outputs/tables/intervention_simulation.csv": "python scripts/run_intervention_simulation.py",
    "outputs/tables/threshold_analysis.csv": "python scripts/run_intervention_simulation.py",
    "outputs/models/model_registry.json": "python scripts/train_scoring_models.py",
    "outputs/tables/risk_queue.csv": "python scripts/score_holdout_queue.py",
    "outputs/tables/global_feature_importance.csv": "python scripts/score_holdout_queue.py",
    "outputs/tables/business_value_simulation.csv": "python scripts/run_business_value_simulation.py",
    "outputs/tables/business_value_sensitivity.csv": "python scripts/run_business_value_simulation.py",
    "outputs/tables/rolling_backtest_metrics.csv": "python scripts/run_monitoring.py",
    "outputs/tables/calibration_metrics.csv": "python scripts/run_monitoring.py",
    "outputs/tables/calibration_bins.csv": "python scripts/run_monitoring.py",
    "outputs/tables/segment_performance.csv": "python scripts/run_monitoring.py",
    "outputs/tables/feature_drift_report.csv": "python scripts/run_monitoring.py",
    "outputs/tables/model_lift_experiments.csv": "python scripts/run_model_lift_experiments.py",
    "outputs/tables/model_lift_topk.csv": "python scripts/run_model_lift_experiments.py",
    "outputs/tables/model_lift_feature_sets.csv": "python scripts/run_model_lift_experiments.py",
}

TIME_VALIDATION_PATH = TABLES_DIR / "time_validation_metrics.csv"
HISTORY_VALIDATION_PATH = TABLES_DIR / "time_validation_with_history_metrics.csv"
INTERVENTION_PATH = TABLES_DIR / "intervention_simulation.csv"
THRESHOLD_PATH = TABLES_DIR / "threshold_analysis.csv"
MODEL_REGISTRY_PATH = MODEL_DIR / "model_registry.json"
RISK_QUEUE_PATH = TABLES_DIR / "risk_queue.csv"
GLOBAL_IMPORTANCE_PATH = TABLES_DIR / "global_feature_importance.csv"
BUSINESS_VALUE_PATH = TABLES_DIR / "business_value_simulation.csv"
BUSINESS_VALUE_SENSITIVITY_PATH = TABLES_DIR / "business_value_sensitivity.csv"
ROLLING_BACKTEST_PATH = TABLES_DIR / "rolling_backtest_metrics.csv"
CALIBRATION_METRICS_PATH = TABLES_DIR / "calibration_metrics.csv"
CALIBRATION_BINS_PATH = TABLES_DIR / "calibration_bins.csv"
SEGMENT_PERFORMANCE_PATH = TABLES_DIR / "segment_performance.csv"
FEATURE_DRIFT_PATH = TABLES_DIR / "feature_drift_report.csv"
MODEL_LIFT_EXPERIMENTS_PATH = TABLES_DIR / "model_lift_experiments.csv"
MODEL_LIFT_TOPK_PATH = TABLES_DIR / "model_lift_topk.csv"
MODEL_LIFT_FEATURE_SETS_PATH = TABLES_DIR / "model_lift_feature_sets.csv"
DEFAULT_QUEUE_MODEL = "HistGradientBoostingClassifier"


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


@st.cache_data
def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


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


def format_money(value: float) -> str:
    return f"${value:,.0f}"


def best_by_metric(df: pd.DataFrame, metric: str) -> pd.Series:
    return df.sort_values(metric, ascending=False).iloc[0]


def run_project_script(script_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def show_command_result(result: subprocess.CompletedProcess) -> None:
    if result.returncode == 0:
        st.success("Command completed successfully.")
    else:
        st.error(f"Command failed with exit code {result.returncode}.")
    if result.stdout:
        st.code(result.stdout, language="text")
    if result.stderr:
        st.code(result.stderr, language="text")


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


def page_risk_queue() -> None:
    st.title("Risk Queue")
    st.markdown(
        """
        Train the leakage-safe historical models, score the future holdout window, and review
        the ranked operational queue. Holdout labels are shown only for evaluation.
        """
    )

    registry_exists = MODEL_REGISTRY_PATH.exists()
    queue_exists = RISK_QUEUE_PATH.exists()
    col1, col2 = st.columns(2)
    col1.metric("Model registry", "Ready" if registry_exists else "Missing")
    col2.metric("Risk queue", "Ready" if queue_exists else "Missing")

    run_col1, run_col2 = st.columns(2)
    with run_col1:
        if st.button("Train models", use_container_width=True):
            result = run_project_script("scripts/train_scoring_models.py")
            st.cache_data.clear()
            show_command_result(result)
    with run_col2:
        if st.button("Score holdout queue", use_container_width=True):
            result = run_project_script("scripts/score_holdout_queue.py")
            st.cache_data.clear()
            show_command_result(result)

    show_missing_artifacts([MODEL_REGISTRY_PATH, RISK_QUEUE_PATH])
    queue_df = load_csv(RISK_QUEUE_PATH)
    if queue_df is None:
        return

    registry = load_json(MODEL_REGISTRY_PATH) or {}
    model_names = sorted(queue_df["model"].unique())
    default_model = registry.get("default_model", DEFAULT_QUEUE_MODEL)
    default_index = model_names.index(default_model) if default_model in model_names else 0

    st.subheader("Queue Controls")
    control_col1, control_col2, control_col3 = st.columns(3)
    selected_model = control_col1.selectbox("Model", model_names, index=default_index)
    risk_bands = ["critical", "high", "medium", "low"]
    present_bands = [band for band in risk_bands if band in set(queue_df["risk_band"])]
    selected_bands = control_col2.multiselect("Risk bands", present_bands, default=present_bands)
    actions = sorted(queue_df["recommended_action"].dropna().unique())
    selected_actions = control_col3.multiselect("Actions", actions, default=actions)

    model_df = queue_df.loc[queue_df["model"] == selected_model].copy()
    filtered_df = model_df.loc[
        model_df["risk_band"].isin(selected_bands)
        & model_df["recommended_action"].isin(selected_actions)
    ].copy()
    max_rows = max(1, len(filtered_df))
    top_n = st.slider("Rows to show", min_value=1, max_value=max_rows, value=min(100, max_rows))
    display_df = filtered_df.sort_values("queue_rank").head(top_n)

    high_priority_mask = model_df["risk_band"].isin(["critical", "high"])
    high_priority_df = model_df.loc[high_priority_mask]
    total_positives = model_df["actual_low_review"].sum() if "actual_low_review" in model_df else 0
    true_positives = high_priority_df["actual_low_review"].sum() if "actual_low_review" in model_df else 0
    precision = true_positives / len(high_priority_df) if len(high_priority_df) else 0.0
    recall = true_positives / total_positives if total_positives else 0.0
    base_rate = model_df["actual_low_review"].mean() if "actual_low_review" in model_df else 0.0
    lift = precision / base_rate if base_rate else 0.0

    st.subheader("Selected Model Summary")
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric("Scored orders", f"{len(model_df):,}")
    metric_col2.metric("High priority", f"{len(high_priority_df):,}")
    metric_col3.metric("Precision", format_pct(precision))
    metric_col4.metric("Recall", format_pct(recall))
    metric_col5.metric("Lift", format_lift(lift))

    st.subheader("Ranked Queue")
    preferred_columns = [
        "queue_rank",
        "order_id",
        "risk_score",
        "risk_band",
        "recommended_action",
        "reason_1",
        "reason_2",
        "reason_3",
        "actual_low_review",
        "is_true_positive",
    ]
    visible_columns = [column for column in preferred_columns if column in display_df.columns]
    st.dataframe(display_df[visible_columns].round(4), use_container_width=True)
    st.download_button(
        "Download filtered queue",
        data=filtered_df.to_csv(index=False),
        file_name="risk_queue_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("Order Detail")
    if display_df.empty:
        st.info("No orders match the selected filters.")
    else:
        order_options = display_df["order_id"].astype(str).tolist()
        selected_order = st.selectbox("Order", order_options)
        detail = display_df.loc[display_df["order_id"].astype(str) == selected_order].iloc[0]

        detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
        detail_col1.metric("Risk score", f"{detail['risk_score']:.4f}")
        detail_col2.metric("Risk band", detail["risk_band"])
        detail_col3.metric("Action", detail["recommended_action"])
        if "actual_low_review" in detail:
            detail_col4.metric("Actual low review", int(detail["actual_low_review"]))

        reason_columns = [column for column in ["reason_1", "reason_2", "reason_3"] if column in display_df.columns]
        if reason_columns:
            st.markdown("Reasons")
            st.table(
                pd.DataFrame(
                    {
                        "Reason": [
                            detail[column]
                            for column in reason_columns
                            if pd.notna(detail[column]) and str(detail[column]).strip()
                        ]
                    }
                )
            )

        context_columns = [
            "order_purchase_timestamp",
            "customer_state",
            "primary_category",
            "estimated_delivery_days",
            "freight_share",
            "multi_seller_flag",
            "high_installment_flag",
            "expensive_order_flag",
            "seller_prior_order_count",
            "seller_prior_low_review_rate",
            "category_prior_low_review_rate",
        ]
        context = {
            column: detail[column]
            for column in context_columns
            if column in detail.index
        }
        st.markdown("Context")
        st.dataframe(pd.DataFrame([context]).round(4), use_container_width=True)


def page_explainability() -> None:
    st.title("Explainability")
    st.markdown(
        """
        Model drivers use permutation importance on the future holdout set. Queue reasons are
        rule-based operational explanations from leakage-safe order attributes.
        """
    )

    show_missing_artifacts([GLOBAL_IMPORTANCE_PATH, RISK_QUEUE_PATH])
    importance_df = load_csv(GLOBAL_IMPORTANCE_PATH)
    if importance_df is None:
        return

    model_names = sorted(importance_df["model"].unique())
    default_index = model_names.index(DEFAULT_QUEUE_MODEL) if DEFAULT_QUEUE_MODEL in model_names else 0
    selected_model = st.selectbox("Model", model_names, index=default_index)
    top_n = st.slider("Top features", min_value=5, max_value=30, value=15)

    model_importance = (
        importance_df.loc[importance_df["model"] == selected_model]
        .sort_values("rank")
        .head(top_n)
        .copy()
    )
    chart_df = model_importance.set_index("feature")[["importance_mean"]].sort_values("importance_mean")
    st.subheader("Permutation Importance")
    st.bar_chart(chart_df)
    st.dataframe(model_importance.round(6), use_container_width=True)

    queue_df = load_csv(RISK_QUEUE_PATH)
    if queue_df is not None and {"model", "reason_1"}.issubset(queue_df.columns):
        st.subheader("Most Common Queue Reasons")
        reason_values = []
        model_queue = queue_df.loc[queue_df["model"] == selected_model]
        for column in ["reason_1", "reason_2", "reason_3"]:
            if column in model_queue.columns:
                reason_values.extend(
                    reason
                    for reason in model_queue[column].dropna().astype(str)
                    if reason.strip()
                )
        if reason_values:
            reason_counts = (
                pd.Series(reason_values)
                .value_counts()
                .rename_axis("reason")
                .reset_index(name="count")
            )
            st.dataframe(reason_counts, use_container_width=True)


def page_business_value() -> None:
    st.title("Business Value")
    st.markdown(
        """
        Compare intervention strategies with explicit assumptions for cost, expected save
        rate, and value per saved low review. These are scenario estimates, not measured
        causal effects.
        """
    )

    show_missing_artifacts([INTERVENTION_PATH, BUSINESS_VALUE_PATH, BUSINESS_VALUE_SENSITIVITY_PATH])
    business_df = load_csv(BUSINESS_VALUE_PATH)
    sensitivity_df = load_csv(BUSINESS_VALUE_SENSITIVITY_PATH)
    if business_df is None:
        return

    model_names = sorted(business_df["model"].unique())
    default_index = model_names.index(DEFAULT_QUEUE_MODEL) if DEFAULT_QUEUE_MODEL in model_names else 0
    selected_model = st.selectbox("Model", model_names, index=default_index)
    model_df = business_df.loc[business_df["model"] == selected_model].copy()

    top_fracs = sorted(model_df["top_frac"].dropna().unique())
    selected_top_frac = st.selectbox(
        "Queue size",
        top_fracs,
        index=top_fracs.index(0.10) if 0.10 in top_fracs else 0,
        format_func=lambda value: f"Top {value:.0%}",
    )
    queue_df = model_df.loc[model_df["top_frac"] == selected_top_frac].copy()
    intervention_types = sorted(queue_df["intervention_type"].unique())
    selected_intervention = st.selectbox("Intervention", intervention_types)
    selected_row = queue_df.loc[queue_df["intervention_type"] == selected_intervention].iloc[0]

    st.subheader("Assumptions")
    assumption_col1, assumption_col2, assumption_col3 = st.columns(3)
    cost_per_order = assumption_col1.number_input(
        "Cost per order",
        min_value=0.0,
        value=float(selected_row["cost_per_order"]),
        step=0.50,
    )
    expected_save_rate = assumption_col2.slider(
        "Expected save rate",
        min_value=0.0,
        max_value=1.0,
        value=float(selected_row["expected_save_rate"]),
        step=0.01,
    )
    value_per_saved = assumption_col3.number_input(
        "Value per saved low review",
        min_value=1.0,
        value=float(selected_row["value_per_saved_low_review"]),
        step=5.0,
    )

    recalculated = calculate_business_value(
        n_flagged=float(selected_row["n_flagged"]),
        true_positives=float(selected_row["true_positives"]),
        cost_per_order=cost_per_order,
        expected_save_rate=expected_save_rate,
        value_per_saved_low_review=value_per_saved,
    )

    st.subheader("Scenario Summary")
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric("Orders flagged", f"{int(selected_row['n_flagged']):,}")
    metric_col2.metric("True positives", f"{int(selected_row['true_positives']):,}")
    metric_col3.metric("Expected saves", f"{recalculated['expected_saved_low_reviews']:.1f}")
    metric_col4.metric("Net value", format_money(recalculated["net_value"]))
    metric_col5.metric("ROI", format_lift(recalculated["roi"]))

    detail_col1, detail_col2, detail_col3 = st.columns(3)
    detail_col1.metric("Intervention cost", format_money(recalculated["intervention_cost"]))
    detail_col2.metric("Gross value", format_money(recalculated["gross_value"]))
    detail_col3.metric("Break-even save rate", format_pct(recalculated["break_even_save_rate"]))

    st.subheader("Default Scenario Comparison")
    comparison_columns = [
        "intervention_type",
        "cost_per_order",
        "expected_save_rate",
        "value_per_saved_low_review",
        "expected_saved_low_reviews",
        "intervention_cost",
        "gross_value",
        "net_value",
        "roi",
        "break_even_save_rate",
    ]
    st.dataframe(
        queue_df[comparison_columns]
        .sort_values("net_value", ascending=False)
        .round(4),
        use_container_width=True,
    )

    if sensitivity_df is not None:
        st.subheader("Save-Rate Sensitivity")
        sensitivity_view = sensitivity_df.loc[
            (sensitivity_df["model"] == selected_model)
            & (sensitivity_df["top_frac"] == selected_top_frac)
            & (sensitivity_df["intervention_type"] == selected_intervention)
        ].copy()
        chart_df = sensitivity_view.set_index("save_rate_multiplier")[["net_value", "gross_value", "intervention_cost"]]
        st.line_chart(chart_df)
        st.dataframe(sensitivity_view.round(4), use_container_width=True)


def page_monitoring() -> None:
    st.title("Monitoring")
    st.markdown(
        """
        Rolling backtests, calibration checks, segment diagnostics, and feature drift reports
        provide a more deployment-like view of model stability.
        """
    )

    monitoring_paths = [
        ROLLING_BACKTEST_PATH,
        CALIBRATION_METRICS_PATH,
        CALIBRATION_BINS_PATH,
        SEGMENT_PERFORMANCE_PATH,
        FEATURE_DRIFT_PATH,
    ]
    show_missing_artifacts(monitoring_paths)
    rolling_df = load_csv(ROLLING_BACKTEST_PATH)
    if rolling_df is None:
        return

    model_names = sorted(rolling_df["model"].unique())
    default_index = model_names.index(DEFAULT_QUEUE_MODEL) if DEFAULT_QUEUE_MODEL in model_names else 0
    selected_model = st.selectbox("Model", model_names, index=default_index)
    model_rolling = rolling_df.loc[rolling_df["model"] == selected_model].copy()

    st.subheader("Rolling Backtest")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Mean ROC-AUC", f"{model_rolling['roc_auc'].mean():.4f}")
    metric_col2.metric("Worst ROC-AUC", f"{model_rolling['roc_auc'].min():.4f}")
    metric_col3.metric("Mean Avg Precision", f"{model_rolling['pr_auc_average_precision'].mean():.4f}")

    trend = model_rolling.sort_values("window").set_index("window")[
        ["roc_auc", "pr_auc_average_precision", "recall_class_1", "precision_class_1"]
    ]
    st.line_chart(trend)
    st.dataframe(model_rolling.round(4), use_container_width=True)

    calibration_df = load_csv(CALIBRATION_METRICS_PATH)
    calibration_bins = load_csv(CALIBRATION_BINS_PATH)
    if calibration_df is not None and calibration_bins is not None:
        st.subheader("Calibration")
        model_calibration = calibration_df.loc[calibration_df["model"] == selected_model].copy()
        st.dataframe(model_calibration.round(4), use_container_width=True)
        selected_window = st.selectbox("Calibration window", sorted(model_calibration["window"].unique()))
        bin_view = calibration_bins.loc[
            (calibration_bins["model"] == selected_model)
            & (calibration_bins["window"] == selected_window)
        ].copy()
        chart_df = bin_view.set_index("bin")[["mean_predicted_probability", "observed_low_review_rate"]]
        st.line_chart(chart_df)

    segment_df = load_csv(SEGMENT_PERFORMANCE_PATH)
    if segment_df is not None and not segment_df.empty:
        st.subheader("Segment Performance")
        model_segments = segment_df.loc[segment_df["model"] == selected_model].copy()
        segment_columns = sorted(model_segments["segment_column"].dropna().unique())
        selected_segment = st.selectbox("Segment", segment_columns)
        segment_view = model_segments.loc[model_segments["segment_column"] == selected_segment].copy()
        st.dataframe(
            segment_view.sort_values(["window", "n_orders"], ascending=[True, False]).round(4),
            use_container_width=True,
        )

    drift_df = load_csv(FEATURE_DRIFT_PATH)
    if drift_df is not None and not drift_df.empty:
        st.subheader("Feature Drift")
        selected_drift_window = st.selectbox("Drift window", sorted(drift_df["window"].unique()))
        drift_view = drift_df.loc[drift_df["window"] == selected_drift_window].copy()
        st.dataframe(
            drift_view.sort_values(["standardized_delta", "absolute_delta"], ascending=False, na_position="last")
            .head(30)
            .round(4),
            use_container_width=True,
        )

    st.subheader("Model Card")
    st.markdown("See `docs/model_card.md` for intended use, validation design, limitations, and monitoring notes.")


def page_model_lift() -> None:
    st.title("Model Lift")
    st.markdown(
        """
        Compare baseline historical features with enhanced leakage-safe historical features,
        calibrated model variants, and ranking-first top-k metrics.
        """
    )

    show_missing_artifacts([MODEL_LIFT_EXPERIMENTS_PATH, MODEL_LIFT_TOPK_PATH, MODEL_LIFT_FEATURE_SETS_PATH])
    experiments_df = load_csv(MODEL_LIFT_EXPERIMENTS_PATH)
    topk_df = load_csv(MODEL_LIFT_TOPK_PATH)
    feature_sets_df = load_csv(MODEL_LIFT_FEATURE_SETS_PATH)
    if experiments_df is None or topk_df is None:
        return

    top10 = topk_df.loc[topk_df["top_frac"].round(2) == 0.10].copy()
    best = top10.sort_values(["lift_at_k", "precision_at_k", "recall_at_k"], ascending=False).iloc[0]
    st.subheader("Best Top 10% Queue")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Feature set", best["feature_set"])
    col2.metric("Model", best["model"])
    col3.metric("Precision", format_pct(best["precision_at_k"]))
    col4.metric("Lift", format_lift(best["lift_at_k"]))

    st.subheader("Experiment Controls")
    feature_sets = sorted(topk_df["feature_set"].unique())
    selected_feature_set = st.selectbox("Feature set", feature_sets)
    model_names = sorted(topk_df.loc[topk_df["feature_set"] == selected_feature_set, "model"].unique())
    selected_model = st.selectbox("Model", model_names)

    selected_topk = topk_df.loc[
        (topk_df["feature_set"] == selected_feature_set)
        & (topk_df["model"] == selected_model)
    ].copy()
    chart_df = selected_topk.sort_values("top_frac").set_index("top_frac")[["precision_at_k", "recall_at_k", "lift_at_k"]]
    st.subheader("Top-k Ranking Metrics")
    st.line_chart(chart_df)
    st.dataframe(selected_topk.round(4), use_container_width=True)

    st.subheader("Feature Set Comparison")
    comparison = (
        top10.groupby(["feature_set", "model"], as_index=False)
        .agg(
            lift_at_10=("lift_at_k", "max"),
            precision_at_10=("precision_at_k", "max"),
            recall_at_10=("recall_at_k", "max"),
        )
        .sort_values("lift_at_10", ascending=False)
    )
    st.dataframe(comparison.round(4), use_container_width=True)

    st.subheader("Classification Metrics")
    st.dataframe(
        experiments_df.sort_values(["feature_set", "roc_auc"], ascending=[True, False]).round(4),
        use_container_width=True,
    )

    if feature_sets_df is not None:
        st.subheader("Feature Set Definitions")
        st.dataframe(feature_sets_df[["feature_set", "n_features", "train_rows", "test_rows"]], use_container_width=True)


def page_reproduce() -> None:
    st.title("How to Reproduce")
    st.markdown("Place the raw Olist CSV files in `olist_data/`, then run the commands below from the project root.")
    commands = [
        "make baseline",
        "make time-validation",
        "make time-validation-history",
        "make intervention",
        "make business-value",
        "make monitoring",
        "make model-lift",
        "make risk-queue",
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
        - `python scripts/run_business_value_simulation.py`
        - `python scripts/run_monitoring.py`
        - `python scripts/run_model_lift_experiments.py`
        - `python scripts/train_scoring_models.py`
        - `python scripts/score_holdout_queue.py`
        """
    )


PAGES = {
    "Project Overview": page_overview,
    "Validation Results": page_validation_results,
    "Intervention Simulation": page_intervention_simulation,
    "Threshold Analysis": page_threshold_analysis,
    "Risk Queue": page_risk_queue,
    "Explainability": page_explainability,
    "Business Value": page_business_value,
    "Monitoring": page_monitoring,
    "Model Lift": page_model_lift,
    "How to Reproduce": page_reproduce,
}


def main() -> None:
    st.sidebar.title("Olist CX Risk")
    page_name = st.sidebar.radio("Navigate", list(PAGES.keys()))
    PAGES[page_name]()


if __name__ == "__main__":
    main()
