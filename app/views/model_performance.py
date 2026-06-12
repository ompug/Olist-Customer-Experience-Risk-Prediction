"""Model Performance: validation, operating curves, thresholds, calibration, experiments."""

from __future__ import annotations

import streamlit as st

import charts
from components import artifacts
from components.layout import honesty_note, page_header

_METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "pr_auc_average_precision": "PR-AUC (avg precision)",
    "f1_class_1": "F1 (low review)",
    "recall_class_1": "Recall (low review)",
}


def _validation_tab() -> None:
    time_df = artifacts.load_csv(artifacts.TIME_VALIDATION)
    history_df = artifacts.load_csv(artifacts.HISTORY_VALIDATION)
    if not artifacts.require_artifacts([artifacts.TIME_VALIDATION, artifacts.HISTORY_VALIDATION]):
        return

    st.markdown(
        "Time-aware validation trains on earlier purchases and tests on later purchases. "
        "It is intentionally harder - and more honest - than a random split."
    )

    merged = time_df[["model", "roc_auc", "pr_auc_average_precision"]].merge(
        history_df[["model", "roc_auc", "pr_auc_average_precision"]],
        on="model",
        suffixes=("_base", "_history"),
    )
    fig = charts.grouped_bar_chart(
        merged,
        x="model",
        series=[
            ("roc_auc_base", "ROC-AUC, base features"),
            ("roc_auc_history", "ROC-AUC, + historical features"),
            ("pr_auc_average_precision_base", "PR-AUC, base features"),
            ("pr_auc_average_precision_history", "PR-AUC, + historical features"),
        ],
        y_format="score",
        height=400,
    )
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)

    metrics = list(_METRIC_LABELS)
    comparison = time_df[["model"] + metrics].merge(
        history_df[["model"] + metrics],
        on="model",
        suffixes=("_base", "_history"),
    )
    for metric in metrics:
        comparison[f"{metric}_delta"] = comparison[f"{metric}_history"] - comparison[f"{metric}_base"]
    st.markdown("**Historical-feature lift per model**")
    delta_columns = ["model"] + [f"{metric}_delta" for metric in metrics]
    st.dataframe(
        comparison[delta_columns]
        .rename(columns={f"{metric}_delta": f"{label} delta" for metric, label in _METRIC_LABELS.items()})
        .round(4),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Full validation tables"):
        st.markdown("Without historical features")
        st.dataframe(time_df.round(4), use_container_width=True, hide_index=True)
        st.markdown("With historical features")
        st.dataframe(history_df.round(4), use_container_width=True, hide_index=True)


def _operating_curves_tab() -> None:
    intervention = artifacts.load_csv(artifacts.INTERVENTION)
    if not artifacts.require_artifacts([artifacts.INTERVENTION]):
        return

    st.markdown(
        "A support team works a prioritized queue, not a 0.5 threshold. These curves show "
        "precision, recall, and lift as the queue grows."
    )
    model_names = sorted(intervention["model"].unique())
    selected_model = st.selectbox("Model", model_names, key="curves_model")
    curve = (
        intervention.loc[
            (intervention["model"] == selected_model)
            & (intervention["selection_strategy"] == "top_fraction")
        ]
        .sort_values("pct_orders_flagged")
        .copy()
    )

    column_1, column_2 = st.columns(2)
    with column_1:
        fig = charts.line_chart(
            curve,
            x="pct_orders_flagged",
            series=[("precision", "Precision"), ("recall", "Recall")],
            x_format="percent",
            y_format="percent",
        )
        fig.update_xaxes(title="Share of orders flagged")
        charts.add_operating_point(fig, 0.10, "10%")
        st.plotly_chart(fig, use_container_width=True)
    with column_2:
        fig = charts.line_chart(
            curve,
            x="pct_orders_flagged",
            series=[("lift_over_random", "Lift over random")],
            x_format="percent",
            y_format="lift",
        )
        fig.update_xaxes(title="Share of orders flagged")
        charts.add_operating_point(fig, 0.10, "10%")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Operating points table"):
        st.dataframe(curve.round(4), use_container_width=True, hide_index=True)


def _thresholds_tab() -> None:
    threshold = artifacts.load_csv(artifacts.THRESHOLD)
    if not artifacts.require_artifacts([artifacts.THRESHOLD]):
        return

    st.markdown(
        "Probability thresholds are the alternative queue policy: lower thresholds catch more "
        "risk but inflate the queue."
    )
    model_names = sorted(threshold["model"].unique())
    selected_model = st.selectbox("Model", model_names, key="threshold_model")
    view = (
        threshold.loc[threshold["model"] == selected_model]
        .sort_values("probability_threshold")
        .copy()
    )
    fig = charts.line_chart(
        view,
        x="probability_threshold",
        series=[
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("pct_orders_flagged", "Share flagged"),
        ],
        x_format="score",
        y_format="percent",
        height=400,
    )
    fig.update_xaxes(title="Probability threshold")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Threshold table"):
        st.dataframe(view.round(4), use_container_width=True, hide_index=True)


def _calibration_tab() -> None:
    if not artifacts.require_artifacts(
        [artifacts.CALIBRATION_METRICS, artifacts.CALIBRATION_BINS],
        context="Calibration artifacts come from the monitoring backtests.",
    ):
        return

    metrics = artifacts.load_csv(artifacts.CALIBRATION_METRICS)
    bins = artifacts.load_csv(artifacts.CALIBRATION_BINS)

    st.markdown(
        "A calibrated model means a 0.30 risk score behaves like a 30% low-review rate. "
        "The closer the curve hugs the diagonal, the more trustworthy the scores."
    )
    column_1, column_2 = st.columns(2)
    model_names = sorted(bins["model"].unique())
    selected_model = column_1.selectbox("Model", model_names, key="calibration_model")
    windows = sorted(bins.loc[bins["model"] == selected_model, "window"].unique())
    selected_window = column_2.selectbox("Backtest window", windows, key="calibration_window")

    view = bins.loc[(bins["model"] == selected_model) & (bins["window"] == selected_window)]
    st.plotly_chart(charts.reliability_chart(view), use_container_width=True)
    st.dataframe(
        metrics.loc[metrics["model"] == selected_model].round(4),
        use_container_width=True,
        hide_index=True,
    )


def _experiments_tab() -> None:
    if not artifacts.require_artifacts(
        [artifacts.MODEL_LIFT_EXPERIMENTS, artifacts.MODEL_LIFT_TOPK],
        context="These are candidate experiments; they replace the production default only "
        "if they win on queue metrics.",
    ):
        return

    experiments = artifacts.load_csv(artifacts.MODEL_LIFT_EXPERIMENTS)
    topk = artifacts.load_csv(artifacts.MODEL_LIFT_TOPK)
    feature_sets = artifacts.load_csv(artifacts.MODEL_LIFT_FEATURE_SETS)

    top10 = topk.loc[topk["top_frac"].round(2) == 0.10].copy()
    if not top10.empty:
        leaderboard = (
            top10.groupby(["feature_set", "model"], as_index=False)
            .agg(
                lift_at_10=("lift_at_k", "max"),
                precision_at_10=("precision_at_k", "max"),
                recall_at_10=("recall_at_k", "max"),
            )
            .sort_values("lift_at_10", ascending=False)
        )
        st.markdown("**Top-10% queue leaderboard**")
        st.dataframe(leaderboard.round(4), use_container_width=True, hide_index=True)

    column_1, column_2 = st.columns(2)
    feature_set_names = sorted(topk["feature_set"].unique())
    selected_set = column_1.selectbox("Feature set", feature_set_names, key="experiment_set")
    model_names = sorted(topk.loc[topk["feature_set"] == selected_set, "model"].unique())
    selected_model = column_2.selectbox("Model", model_names, key="experiment_model")

    view = (
        topk.loc[(topk["feature_set"] == selected_set) & (topk["model"] == selected_model)]
        .sort_values("top_frac")
        .copy()
    )
    fig = charts.line_chart(
        view,
        x="top_frac",
        series=[
            ("precision_at_k", "Precision at k"),
            ("recall_at_k", "Recall at k"),
            ("lift_at_k", "Lift at k"),
        ],
        x_format="percent",
        y_format="score",
        height=400,
    )
    fig.update_xaxes(title="Queue size (top fraction)")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Classification metrics and feature-set definitions"):
        st.dataframe(
            experiments.sort_values(["feature_set", "roc_auc"], ascending=[True, False]).round(4),
            use_container_width=True,
            hide_index=True,
        )
        if feature_sets is not None:
            st.dataframe(
                feature_sets[["feature_set", "n_features", "train_rows", "test_rows"]],
                use_container_width=True,
                hide_index=True,
            )


def render() -> None:
    page_header(
        "Model Performance",
        "How we know the queue is worth working: every claim is tested on orders from "
        "later in time than anything the model trained on.",
        tech_note="Chronological train/test splits, queue operating curves, threshold "
        "tradeoffs, calibration diagnostics, and candidate model-lift experiments.",
    )
    honesty_note(
        "All headline metrics use the time-aware future-order test set. Random-split numbers "
        "are kept only as a baseline comparison in the pipeline."
    )

    tab_validation, tab_curves, tab_thresholds, tab_calibration, tab_experiments = st.tabs(
        ["Validation", "Queue curves", "Thresholds", "Calibration", "Experiments"]
    )
    with tab_validation:
        _validation_tab()
    with tab_curves:
        _operating_curves_tab()
    with tab_thresholds:
        _thresholds_tab()
    with tab_calibration:
        _calibration_tab()
    with tab_experiments:
        _experiments_tab()
