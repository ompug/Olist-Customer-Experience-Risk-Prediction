"""Risk Queue: the operational product page for support and operations teams."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import design
from components import artifacts
from components.layout import honesty_note, kpi_row, page_header, risk_badge, style_risk_band_column

from cx_risk.scoring import DEFAULT_QUEUE_MODEL

_QUEUE_COLUMNS = [
    "queue_rank",
    "order_id",
    "risk_score",
    "risk_band",
    "recommended_action",
    "reason_1",
    "reason_2",
    "reason_3",
    "actual_low_review",
]

_CONTEXT_COLUMNS = [
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


def render() -> None:
    page_header(
        "Intervention Queue",
        "Today's orders worth a proactive touch: ranked by risk, labeled with a recommended "
        "action, and explained in plain language.",
        tech_note="Models train on earlier orders and score the future holdout window as if "
        "those orders were unlabeled; holdout labels appear only as evaluation columns.",
    )

    if not artifacts.require_artifacts(
        [artifacts.MODEL_REGISTRY, artifacts.RISK_QUEUE],
        context="Training and scoring run in the pipeline, never inside the dashboard.",
    ):
        return

    queue = artifacts.load_csv(artifacts.RISK_QUEUE)
    registry = artifacts.load_json(artifacts.MODEL_REGISTRY) or {}
    if queue is None or queue.empty:
        st.info("The risk queue artifact is empty.")
        return

    model_names = sorted(queue["model"].unique())
    default_model = registry.get("default_model", DEFAULT_QUEUE_MODEL)
    default_index = model_names.index(default_model) if default_model in model_names else 0

    filter_1, filter_2, filter_3 = st.columns(3)
    selected_model = filter_1.selectbox("Model", model_names, index=default_index)
    present_bands = [band for band in design.RISK_BAND_ORDER if band in set(queue["risk_band"])]
    selected_bands = filter_2.multiselect("Risk bands", present_bands, default=present_bands)
    actions = sorted(queue["recommended_action"].dropna().unique())
    selected_actions = filter_3.multiselect("Recommended actions", actions, default=actions)

    model_queue = queue.loc[queue["model"] == selected_model].copy()
    filtered = model_queue.loc[
        model_queue["risk_band"].isin(selected_bands)
        & model_queue["recommended_action"].isin(selected_actions)
    ].sort_values("queue_rank")

    high_priority = model_queue.loc[model_queue["risk_band"].isin(["critical", "high"])]
    has_labels = "actual_low_review" in model_queue.columns and model_queue["actual_low_review"].notna().any()
    if has_labels:
        total_positives = model_queue["actual_low_review"].sum()
        true_positives = high_priority["actual_low_review"].sum()
        precision = true_positives / len(high_priority) if len(high_priority) else 0.0
        recall = true_positives / total_positives if total_positives else 0.0
        base_rate = model_queue["actual_low_review"].mean()
        lift = precision / base_rate if base_rate else 0.0
        kpi_row(
            [
                {
                    "label": "Scored orders",
                    "value": f"{len(model_queue):,}",
                    "help": "All future-holdout orders scored by the selected model.",
                },
                {
                    "label": "Needs attention",
                    "value": f"{len(high_priority):,}",
                    "help": "Orders in the critical and high risk bands.",
                },
                {
                    "label": "Hit rate",
                    "value": f"{precision:.1%}",
                    "help": "Precision: share of critical/high orders that truly ended in a "
                    "low review.",
                },
                {
                    "label": "Bad reviews caught",
                    "value": f"{recall:.1%}",
                    "help": "Recall: share of all low reviews in the holdout that the "
                    "critical/high bands contain.",
                },
                {
                    "label": "Density vs random",
                    "value": f"{lift:.2f}x",
                    "help": "Lift: hit rate divided by the base low-review rate.",
                },
            ]
        )
        honesty_note(
            "Holdout labels appear only as evaluation columns added after scoring. "
            "They are never model inputs."
        )

    band_summary = " ".join(
        f"{risk_badge(band)} {int((model_queue['risk_band'] == band).sum()):,}"
        for band in present_bands
    )
    st.markdown(band_summary, unsafe_allow_html=True)

    st.subheader("Ranked queue")
    visible_columns = [column for column in _QUEUE_COLUMNS if column in filtered.columns]
    max_rows = max(1, len(filtered))
    top_n = st.slider("Rows to show", min_value=1, max_value=max_rows, value=min(100, max_rows))
    display = filtered.head(top_n)
    st.dataframe(
        style_risk_band_column(display[visible_columns]),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download filtered queue (CSV)",
        data=filtered.to_csv(index=False),
        file_name="risk_queue_filtered.csv",
        mime="text/csv",
    )

    st.subheader("Order detail")
    if display.empty:
        st.info("No orders match the selected filters.")
        return

    order_options = display["order_id"].astype(str).tolist()
    selected_order = st.selectbox("Order", order_options)
    detail = display.loc[display["order_id"].astype(str) == selected_order].iloc[0]

    kpis = [
        {"label": "Queue rank", "value": f"#{int(detail['queue_rank'])}"},
        {"label": "Risk score", "value": f"{detail['risk_score']:.4f}"},
        {"label": "Recommended action", "value": str(detail["recommended_action"])},
    ]
    if has_labels and pd.notna(detail.get("actual_low_review")):
        kpis.append({"label": "Actual low review", "value": str(int(detail["actual_low_review"]))})
    kpi_row(kpis)
    st.markdown(risk_badge(detail["risk_band"]), unsafe_allow_html=True)

    reasons = [
        str(detail[column])
        for column in ["reason_1", "reason_2", "reason_3"]
        if column in detail.index and pd.notna(detail[column]) and str(detail[column]).strip()
    ]
    if reasons:
        st.markdown("**Why this order is queued**")
        for reason in reasons:
            st.markdown(f"- {reason}")

    context = {
        column: detail[column]
        for column in _CONTEXT_COLUMNS
        if column in detail.index and pd.notna(detail[column])
    }
    if context:
        st.markdown("**Order context**")
        st.dataframe(pd.DataFrame([context]).round(4), use_container_width=True, hide_index=True)
