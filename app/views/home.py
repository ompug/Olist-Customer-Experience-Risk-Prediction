"""Results Overview: the headline numbers in one place."""

from __future__ import annotations

import streamlit as st

import charts
from components import artifacts
from components.layout import honesty_note, kpi_row, page_header


def render() -> None:
    page_header(
        "Results Overview",
        "What the model delivers: a support queue several times denser in bad experiences "
        "than picking orders at random.",
        tech_note="All numbers come from the time-aware future-order test set with "
        "leakage-safe historical features.",
    )

    intervention = artifacts.load_csv(artifacts.INTERVENTION)
    history = artifacts.load_csv(artifacts.HISTORY_VALIDATION)

    if intervention is not None and not intervention.empty:
        top_10 = intervention.loc[intervention["top_frac"].round(2) == 0.10].copy()
        if not top_10.empty:
            best = top_10.sort_values("lift_over_random", ascending=False).iloc[0]
            best_auc = (
                f"{history['roc_auc'].max():.4f}" if history is not None and not history.empty else "n/a"
            )
            kpi_row(
                [
                    {
                        "label": "Queue density vs random",
                        "value": f"{best['lift_over_random']:.2f}x",
                        "help": "Lift at the top-10% queue: precision divided by the base "
                        "low-review rate on the future test window.",
                    },
                    {
                        "label": "Bad reviews caught",
                        "value": f"{best['recall']:.1%}",
                        "help": "Recall at the top-10% queue: share of all future low reviews "
                        "inside the flagged set.",
                    },
                    {
                        "label": "Daily queue size",
                        "value": f"{int(best['n_flagged']):,} orders",
                        "help": "Orders flagged when capacity is set to 10% of future orders.",
                    },
                    {
                        "label": "Best ROC-AUC",
                        "value": best_auc,
                        "help": "Time-aware chronological validation with strictly-prior "
                        "historical features. Modest by design: the model only sees "
                        "purchase-time information.",
                    },
                ]
            )
            st.caption(
                f"Best top-10% queue: {best['model']} with {best['precision']:.1%} precision "
                f"against a {best['base_positive_rate']:.1%} base low-review rate."
            )

        st.subheader("Queue concentration vs queue size")
        st.markdown(
            "The tradeoff every support lead cares about: flag more orders and you catch more "
            "problems, but each contact finds fewer of them. The curve shows the concentration "
            "advantage at every capacity level."
        )
        curve = intervention.loc[intervention["selection_strategy"] == "top_fraction"].copy()
        if not curve.empty:
            lift_wide = (
                curve.pivot_table(index="pct_orders_flagged", columns="model", values="lift_over_random")
                .sort_index()
                .reset_index()
            )
            model_names = [column for column in lift_wide.columns if column != "pct_orders_flagged"]
            fig = charts.line_chart(
                lift_wide,
                x="pct_orders_flagged",
                series=[(name, name) for name in model_names],
                x_format="percent",
                y_format="lift",
                height=420,
            )
            fig.update_xaxes(title="Share of future orders flagged")
            fig.update_yaxes(title="Lift over random selection")
            charts.add_operating_point(fig, 0.10, "10% operating point")
            st.plotly_chart(fig, use_container_width=True)
    else:
        artifacts.require_artifacts([artifacts.INTERVENTION, artifacts.HISTORY_VALIDATION])

    honesty_note(
        "These results measure prioritization quality on future orders, not causal intervention "
        "impact. The queue tells a team where to look, not what would have happened."
    )
