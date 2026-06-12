"""Welcome: the title page. Sells the product to business and technical visitors."""

from __future__ import annotations

from typing import Optional

import streamlit as st

import design
from components import artifacts
from components.layout import footer, hero, stat_card

_PIPELINE_DOT = """
digraph {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fillcolor="#F4F6FA", color="#1F3A5F",
          fontname="Helvetica", fontsize=11, fontcolor="#1B2333", margin="0.18,0.1"];
    edge [color="#5C677D", arrowsize=0.7];

    raw [label="Raw Olist tables\\norders / items / payments / reviews"];
    features [label="Leakage-safe features\\n+ strictly-prior history"];
    split [label="Time-aware split\\ntrain earlier, test later"];
    models [label="Model training\\nLR / RF / HistGB"];
    queue [label="Ranked risk queue\\nbands, actions, reasons"];
    serve [label="Dashboard + API"];

    raw -> features -> split -> models -> queue -> serve;
}
"""


def _headline_stats() -> Optional[dict]:
    """Pull live headline numbers from artifacts; None when unavailable."""
    intervention = artifacts.load_csv(artifacts.INTERVENTION)
    if intervention is None or intervention.empty:
        return None
    top_10 = intervention.loc[intervention["top_frac"].round(2) == 0.10]
    if top_10.empty:
        return None
    best = top_10.sort_values("lift_over_random", ascending=False).iloc[0]
    return {
        "lift": f"{best['lift_over_random']:.1f}x",
        "recall": f"1 in {round(1 / best['recall'])}" if best["recall"] > 0 else "n/a",
        "queue": f"{int(best['n_flagged']):,}",
    }


def render() -> None:
    hero(design.PRODUCT_NAME, design.PRODUCT_TAGLINE, design.PRODUCT_POSITIONING)

    stats = _headline_stats()
    column_1, column_2, column_3 = st.columns(3)
    with column_1:
        stat_card(
            stats["lift"] if stats else "2.4x",
            "more low-review orders per contact than picking orders at random",
        )
    with column_2:
        stat_card(
            stats["recall"] if stats else "1 in 4",
            "future bad reviews caught while touching only 10% of orders",
        )
    with column_3:
        stat_card(
            "Future-only",
            "every result is validated on orders the model has never seen - from later in time",
        )

    st.markdown("")

    st.subheader("The problem")
    st.markdown(
        "By the time a 1-star review is posted, the damage is done. Support teams have the "
        "capacity to intervene on a fraction of orders - the question is *which* fraction. "
        f"{design.PRODUCT_NAME} answers it with a ranked queue: the orders most likely to end "
        "in a bad experience, scored before the customer ever writes a word."
    )

    st.subheader("Pick your path")
    # Imported lazily: navigation imports this module to build its Page objects.
    import navigation

    business_column, technical_column = st.columns(2)
    with business_column:
        with st.container(border=True):
            st.markdown("**For operations and CX leaders**")
            st.markdown(
                "- A daily **Intervention Queue** with risk bands, recommended actions, "
                "and plain-language reasons per order\n"
                "- An **ROI Planner** where you set the cost and effectiveness assumptions "
                "and see net value and break-even points\n"
                "- No black box: every flagged order says *why* it is flagged"
            )
            st.page_link(navigation.INTERVENTION_QUEUE, label="Start with the Intervention Queue")
            st.page_link(navigation.ROI_PLANNER, label="Plan ROI scenarios")
    with technical_column:
        with st.container(border=True):
            st.markdown("**For technical reviewers**")
            st.markdown(
                "- **Leakage-safe by construction**: forbidden-column guards, strictly-prior "
                "historical aggregates, no review-derived features\n"
                "- **Time-aware validation**: chronological splits, rolling backtests, "
                "calibration, segment and drift diagnostics\n"
                "- **Artifact-driven architecture**: the dashboard and FastAPI service only "
                "read pipeline outputs - nothing trains at request time"
            )
            st.page_link(navigation.MODEL_PERFORMANCE, label="Inspect model performance")
            st.page_link(navigation.RELIABILITY, label="Read the methodology")

    st.subheader("How it works")
    step_1, step_2, step_3 = st.columns(3)
    with step_1:
        with st.container(border=True):
            st.markdown("**1. Score**")
            st.markdown(
                "Every order is scored at purchase time using only what is knowable then: "
                "basket, payment, geography, delivery estimate, and prior seller/category track record."
            )
    with step_2:
        with st.container(border=True):
            st.markdown("**2. Rank**")
            st.markdown(
                "Scores become a ranked queue sized to your team's capacity - top 10% of orders "
                "by default - with risk bands from low to critical."
            )
    with step_3:
        with st.container(border=True):
            st.markdown("**3. Act**")
            st.markdown(
                "Each queued order carries a recommended action - shipping check, seller "
                "escalation, proactive outreach - and the reasons behind it."
            )

    st.graphviz_chart(_PIPELINE_DOT, use_container_width=True)

    footer(
        f"{design.PRODUCT_NAME} is a portfolio project built on the public Brazilian Olist "
        "e-commerce dataset. Python, scikit-learn, Streamlit, FastAPI, Docker. "
        f'Source: <a href="{design.REPO_URL}">GitHub repository</a>.'
    )
