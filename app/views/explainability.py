"""Explainability: global model drivers, feature families, and queue reasons."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
from components import artifacts
from components.layout import honesty_note, page_header

from cx_risk.preprocessing import (
    ENHANCED_HISTORICAL_FEATURE_COLUMNS,
    FEATURE_FAMILY_MAP,
    HISTORICAL_FEATURE_COLUMNS,
)
from cx_risk.scoring import DEFAULT_QUEUE_MODEL

_EXCLUDED_SIGNALS = [
    ("Review content", "Text, titles, and timestamps of the review the model is predicting."),
    ("Actual delivery outcomes", "Delivery dates and lateness are unknown at purchase time."),
    ("Raw identifiers", "Order, customer, product, and seller IDs never enter the model."),
    ("Future aggregates", "Historical risk rates use strictly prior orders only."),
]


def _feature_family(feature: str) -> str:
    if feature in HISTORICAL_FEATURE_COLUMNS or feature in ENHANCED_HISTORICAL_FEATURE_COLUMNS:
        return "historical risk"
    return FEATURE_FAMILY_MAP.get(feature, "other")


def render() -> None:
    page_header(
        "Why Orders Get Flagged",
        "No black box: what drives risk scores across the board, and why each individual "
        "order lands in the queue.",
        tech_note="Global drivers use permutation importance on the future holdout set. "
        "Per-order reasons are deterministic business rules over leakage-safe fields.",
    )

    if not artifacts.require_artifacts([artifacts.GLOBAL_IMPORTANCE]):
        return
    importance = artifacts.load_csv(artifacts.GLOBAL_IMPORTANCE)

    model_names = sorted(importance["model"].unique())
    default_index = model_names.index(DEFAULT_QUEUE_MODEL) if DEFAULT_QUEUE_MODEL in model_names else 0
    column_1, column_2 = st.columns([2, 1])
    selected_model = column_1.selectbox("Model", model_names, index=default_index)
    top_n = column_2.slider("Top features", min_value=5, max_value=30, value=15)

    model_importance = (
        importance.loc[importance["model"] == selected_model].sort_values("rank").head(top_n).copy()
    )

    st.subheader("What drives risk scores")
    st.markdown(
        "Each bar answers a simple question: how much worse does the model get when this "
        "signal is scrambled? Bigger bars mean the model leans on that signal more."
    )
    st.plotly_chart(
        charts.hbar_chart(
            model_importance,
            value_column="importance_mean",
            label_column="feature",
            error_column="importance_std",
            value_format="score",
        ),
        use_container_width=True,
    )
    scoring_metric = (
        str(model_importance["scoring"].iloc[0]) if "scoring" in model_importance.columns else "n/a"
    )
    st.caption(f"Importance is the drop in {scoring_metric} when a feature is permuted on the holdout set.")

    st.subheader("Importance by feature family")
    family_importance = importance.loc[importance["model"] == selected_model].copy()
    family_importance["family"] = family_importance["feature"].map(_feature_family)
    family_summary = (
        family_importance.groupby("family", as_index=False)
        .agg(total_importance=("importance_mean", "sum"))
        .sort_values("total_importance", ascending=False)
    )
    st.plotly_chart(
        charts.hbar_chart(
            family_summary,
            value_column="total_importance",
            label_column="family",
            value_format="score",
            height=300,
        ),
        use_container_width=True,
    )

    queue = artifacts.load_csv(artifacts.RISK_QUEUE)
    if queue is not None and {"model", "reason_1"}.issubset(queue.columns):
        st.subheader("Most common queue reasons")
        model_queue = queue.loc[queue["model"] == selected_model]
        reasons: list[str] = []
        for column in ["reason_1", "reason_2", "reason_3"]:
            if column in model_queue.columns:
                reasons.extend(
                    reason for reason in model_queue[column].dropna().astype(str) if reason.strip()
                )
        if reasons:
            reason_counts = (
                pd.Series(reasons).value_counts().rename_axis("reason").reset_index(name="count")
            )
            st.plotly_chart(
                charts.hbar_chart(
                    reason_counts,
                    value_column="count",
                    label_column="reason",
                    value_format="number",
                    height=300,
                ),
                use_container_width=True,
            )

    honesty_note(
        "Permutation importance measures predictive contribution, not causality. Queue reasons "
        "are operational triage rules, not model attributions."
    )

    st.subheader("What the model cannot see")
    columns = st.columns(2)
    for index, (title, description) in enumerate(_EXCLUDED_SIGNALS):
        with columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.markdown(description)
