"""Business Value: interactive intervention scenario simulator."""

from __future__ import annotations

import streamlit as st

import charts
from components import artifacts
from components.layout import honesty_note, kpi_row, page_header

from cx_risk.business_value import calculate_business_value
from cx_risk.scoring import DEFAULT_QUEUE_MODEL


def _format_money(value: float) -> str:
    return f"${value:,.0f}"


def render() -> None:
    page_header(
        "ROI Planner",
        "What intervening costs, what it returns, and the assumptions you control: tune "
        "cost, effectiveness, and value to your own business.",
        tech_note="Scenarios combine measured queue performance (flagged orders, true "
        "positives) with explicit assumptions - never with causal claims.",
    )
    honesty_note(
        "These are planning scenarios built on assumed save rates and values, not measured "
        "causal effects of any intervention."
    )

    if not artifacts.require_artifacts([artifacts.BUSINESS_VALUE, artifacts.BUSINESS_VALUE_SENSITIVITY]):
        return
    business = artifacts.load_csv(artifacts.BUSINESS_VALUE)
    sensitivity = artifacts.load_csv(artifacts.BUSINESS_VALUE_SENSITIVITY)

    column_1, column_2, column_3 = st.columns(3)
    model_names = sorted(business["model"].unique())
    default_index = model_names.index(DEFAULT_QUEUE_MODEL) if DEFAULT_QUEUE_MODEL in model_names else 0
    selected_model = column_1.selectbox("Model", model_names, index=default_index)

    model_view = business.loc[business["model"] == selected_model].copy()
    top_fracs = sorted(model_view["top_frac"].dropna().unique())
    selected_frac = column_2.selectbox(
        "Queue size",
        top_fracs,
        index=top_fracs.index(0.10) if 0.10 in top_fracs else 0,
        format_func=lambda value: f"Top {value:.0%}",
    )
    queue_view = model_view.loc[model_view["top_frac"] == selected_frac].copy()
    intervention_types = sorted(queue_view["intervention_type"].unique())
    selected_intervention = column_3.selectbox("Intervention", intervention_types)
    selected_row = queue_view.loc[queue_view["intervention_type"] == selected_intervention].iloc[0]

    st.subheader("Assumptions")
    assumption_1, assumption_2, assumption_3 = st.columns(3)
    cost_per_order = assumption_1.number_input(
        "Cost per contacted order ($)",
        min_value=0.0,
        value=float(selected_row["cost_per_order"]),
        step=0.50,
    )
    expected_save_rate = assumption_2.slider(
        "Expected save rate",
        min_value=0.0,
        max_value=1.0,
        value=float(selected_row["expected_save_rate"]),
        step=0.01,
        help="Share of correctly flagged low reviews the intervention actually prevents.",
    )
    value_per_saved = assumption_3.number_input(
        "Value per saved low review ($)",
        min_value=1.0,
        value=float(selected_row["value_per_saved_low_review"]),
        step=5.0,
    )

    scenario = calculate_business_value(
        n_flagged=float(selected_row["n_flagged"]),
        true_positives=float(selected_row["true_positives"]),
        cost_per_order=cost_per_order,
        expected_save_rate=expected_save_rate,
        value_per_saved_low_review=value_per_saved,
    )

    st.subheader("Scenario outcome")
    kpi_row(
        [
            {
                "label": "Orders contacted",
                "value": f"{int(selected_row['n_flagged']):,}",
                "help": "Queue size at the selected capacity: every contacted order incurs "
                "the per-order cost.",
            },
            {
                "label": "Expected saves",
                "value": f"{scenario['expected_saved_low_reviews']:.1f}",
                "help": "Correctly flagged low reviews multiplied by the assumed save rate.",
            },
            {
                "label": "Intervention cost",
                "value": _format_money(scenario["intervention_cost"]),
                "help": "Orders contacted multiplied by cost per order.",
            },
            {
                "label": "Net value",
                "value": _format_money(scenario["net_value"]),
                "help": "Gross value of expected saves minus intervention cost.",
            },
            {
                "label": "ROI",
                "value": f"{scenario['roi']:.2f}x",
                "help": "Net value divided by intervention cost.",
            },
        ]
    )
    st.caption(
        f"Break-even save rate: {scenario['break_even_save_rate']:.1%} - the intervention pays "
        "for itself above this effectiveness."
    )

    st.subheader("Net value by intervention (default assumptions)")
    comparison = queue_view.sort_values("net_value", ascending=False)
    st.plotly_chart(
        charts.hbar_chart(
            comparison,
            value_column="net_value",
            label_column="intervention_type",
            value_format="money",
            height=300,
        ),
        use_container_width=True,
    )
    with st.expander("Scenario comparison table"):
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
        st.dataframe(comparison[comparison_columns].round(4), use_container_width=True, hide_index=True)

    if sensitivity is not None and not sensitivity.empty:
        st.subheader("Save-rate sensitivity")
        st.markdown(
            "Net value across interventions when the assumed save rate is scaled up or down. "
            "Robust strategies stay positive even at pessimistic multipliers."
        )
        sensitivity_view = sensitivity.loc[
            (sensitivity["model"] == selected_model) & (sensitivity["top_frac"] == selected_frac)
        ].copy()
        if not sensitivity_view.empty:
            pivot = sensitivity_view.pivot_table(
                index="intervention_type",
                columns="save_rate_multiplier",
                values="net_value",
            ).sort_index()
            pivot.columns = [f"{column:.2f}x" for column in pivot.columns]
            st.plotly_chart(
                charts.diverging_heatmap(pivot, value_format="money", height=340),
                use_container_width=True,
            )
