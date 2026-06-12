"""Monitoring & Methodology: stability, drift, segments, and the honest writeup."""

from __future__ import annotations

import streamlit as st

import charts
from components import artifacts
from components.layout import honesty_note, kpi_row, page_header

from cx_risk.scoring import DEFAULT_QUEUE_MODEL

_REPRODUCE_COMMANDS = [
    ("Validate on future orders", "make time-validation"),
    ("Add leakage-safe historical features", "make time-validation-history"),
    ("Simulate intervention queues", "make intervention"),
    ("Score the production-style risk queue", "make risk-queue"),
    ("Model business-value scenarios", "make business-value"),
    ("Run rolling backtests and drift checks", "make monitoring"),
    ("Run model-lift experiments", "make model-lift"),
    ("Run the test suite", "make test"),
]


def _stability_tab() -> None:
    if not artifacts.require_artifacts([artifacts.ROLLING_BACKTEST]):
        return
    rolling = artifacts.load_csv(artifacts.ROLLING_BACKTEST)

    st.markdown(
        "Expanding-window backtests retrain on growing history and test on the next slice of "
        "future orders. Stable metrics across windows are the deployment signal."
    )
    model_names = sorted(rolling["model"].unique())
    default_index = model_names.index(DEFAULT_QUEUE_MODEL) if DEFAULT_QUEUE_MODEL in model_names else 0
    selected_model = st.selectbox("Model", model_names, index=default_index, key="stability_model")
    view = rolling.loc[rolling["model"] == selected_model].sort_values("window").copy()

    kpi_row(
        [
            {
                "label": "Typical ranking quality",
                "value": f"{view['roc_auc'].mean():.4f}",
                "help": "Mean ROC-AUC across backtest windows (1.0 is perfect ranking, "
                "0.5 is random).",
            },
            {
                "label": "Worst window",
                "value": f"{view['roc_auc'].min():.4f}",
                "help": "Lowest ROC-AUC observed in any backtest window - the stability floor.",
            },
            {
                "label": "Queue-focused quality",
                "value": f"{view['pr_auc_average_precision'].mean():.4f}",
                "help": "Mean PR-AUC (average precision), which rewards concentrating low "
                "reviews at the top of the ranking.",
            },
            {
                "label": "Backtest windows",
                "value": f"{view['window'].nunique()}",
                "help": "Number of expanding-window retrain/test cycles.",
            },
        ]
    )

    series = [
        (column, label)
        for column, label in [
            ("roc_auc", "ROC-AUC"),
            ("pr_auc_average_precision", "PR-AUC"),
            ("recall_class_1", "Recall (low review)"),
            ("precision_class_1", "Precision (low review)"),
        ]
        if column in view.columns
    ]
    fig = charts.line_chart(view, x="window", series=series, x_format="number", y_format="score", height=400)
    fig.update_xaxes(title="Backtest window", dtick=1)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Backtest table"):
        st.dataframe(view.round(4), use_container_width=True, hide_index=True)


def _segments_tab() -> None:
    if not artifacts.require_artifacts([artifacts.SEGMENT_PERFORMANCE]):
        return
    segments = artifacts.load_csv(artifacts.SEGMENT_PERFORMANCE)

    st.markdown(
        "Per-segment diagnostics surface where the model is weak before a queue is trusted "
        "for that slice of orders."
    )
    column_1, column_2 = st.columns(2)
    model_names = sorted(segments["model"].unique())
    default_index = model_names.index(DEFAULT_QUEUE_MODEL) if DEFAULT_QUEUE_MODEL in model_names else 0
    selected_model = column_1.selectbox("Model", model_names, index=default_index, key="segments_model")
    model_segments = segments.loc[segments["model"] == selected_model].copy()
    segment_columns = sorted(model_segments["segment_column"].dropna().unique())
    selected_segment = column_2.selectbox("Segment", segment_columns, key="segments_column")

    view = model_segments.loc[model_segments["segment_column"] == selected_segment].copy()
    summary = (
        view.groupby("segment_value", as_index=False)
        .agg(
            n_orders=("n_orders", "sum"),
            average_precision=("average_precision", "mean"),
            roc_auc=("roc_auc", "mean"),
            low_review_rate=("low_review_rate", "mean"),
        )
        .sort_values("average_precision", ascending=False)
    )
    st.plotly_chart(
        charts.hbar_chart(
            summary,
            value_column="average_precision",
            label_column="segment_value",
            value_format="score",
        ),
        use_container_width=True,
    )
    with st.expander("Segment table (all windows)"):
        st.dataframe(
            view.sort_values(["window", "n_orders"], ascending=[True, False]).round(4),
            use_container_width=True,
            hide_index=True,
        )


def _drift_tab() -> None:
    if not artifacts.require_artifacts([artifacts.FEATURE_DRIFT]):
        return
    drift = artifacts.load_csv(artifacts.FEATURE_DRIFT)

    st.markdown(
        "Train-vs-future distribution shifts. Large standardized deltas mean the scoring "
        "window no longer looks like the training window."
    )
    windows = sorted(drift["window"].unique())
    selected_window = st.selectbox("Drift window", windows, key="drift_window")
    view = (
        drift.loc[drift["window"] == selected_window]
        .sort_values(["standardized_delta", "absolute_delta"], ascending=False, na_position="last")
        .head(20)
        .copy()
    )
    numeric_view = view.loc[view["standardized_delta"].notna()]
    if not numeric_view.empty:
        st.plotly_chart(
            charts.hbar_chart(
                numeric_view,
                value_column="standardized_delta",
                label_column="feature",
                value_format="score",
            ),
            use_container_width=True,
        )
    with st.expander("Drift report table"):
        st.dataframe(view.round(4), use_container_width=True, hide_index=True)


def _methodology_tab() -> None:
    st.subheader("Leakage prevention")
    st.markdown(
        """
        The defining constraint of this project: every feature must be knowable **before the
        review exists**. The pipeline enforces this with an explicit forbidden-column guard
        that runs at split time and scoring time.

        Excluded from the model:

        - Review text, titles, scores, and review timestamps
        - Actual delivery and lateness fields (unknown at purchase time)
        - Raw high-cardinality identifiers (order, customer, product, seller IDs)
        - Any aggregate that includes the current or future orders

        Historical seller, category, and customer-state risk rates are computed from strictly
        earlier purchase timestamps. Orders sharing the same timestamp are excluded from one
        another's history.
        """
    )
    honesty_note(
        "Intervention and business-value numbers measure prioritization quality and scenario "
        "planning. They are not causal estimates of intervention impact."
    )

    st.subheader("Validation design")
    st.markdown(
        """
        1. **Chronological holdout** - train on the earliest 80% of orders, test on the final 20%.
        2. **Rolling backtests** - expanding training windows, each tested on the next future slice.
        3. **Calibration, segment, and drift diagnostics** - generated by the monitoring pipeline.

        The model card in `docs/model_card.md` documents intended use, boundaries, and limitations.
        """
    )

    st.subheader("Reproduce everything")
    st.markdown("Place the raw Olist CSVs in `olist_data/`, then run from the project root:")
    for description, command in _REPRODUCE_COMMANDS:
        column_1, column_2 = st.columns([1, 1])
        column_1.markdown(description)
        column_2.code(command, language="bash")
    st.caption(
        "The dashboard only reads artifacts - it never trains models. The FastAPI scoring "
        "service starts with `make api` once `make risk-queue` has produced model artifacts."
    )


def render() -> None:
    page_header(
        "Reliability & Methods",
        "Two questions, one page: is it still working, and how does it avoid fooling itself?",
        tech_note="Expanding-window backtests, segment diagnostics, train-vs-future drift "
        "reports, and the full leakage-prevention methodology.",
    )
    tab_stability, tab_segments, tab_drift, tab_methodology = st.tabs(
        ["Stability", "Segments", "Drift", "Methodology"]
    )
    with tab_stability:
        _stability_tab()
    with tab_segments:
        _segments_tab()
    with tab_drift:
        _drift_tab()
    with tab_methodology:
        _methodology_tab()
