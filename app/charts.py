"""Plotly chart factory with one consistent visual style for every page."""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
import plotly.graph_objects as go

import design

AXIS_TICKFORMAT = {
    "percent": ".0%",
    "number": ",",
    "money": "$,.0f",
    "score": ".2f",
    "lift": ".1f",
}

HOVER_VALUEFORMAT = {
    "percent": ".2%",
    "number": ",.0f",
    "money": "$,.0f",
    "score": ".4f",
    "lift": ".2f",
}


def base_figure(height: int = 380) -> go.Figure:
    """Create an empty figure carrying the shared dashboard layout."""
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans Pro, sans-serif", size=13, color=design.TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor=design.GRID, font_color=design.TEXT),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=design.GRID, ticks="outside", tickcolor=design.GRID)
    fig.update_yaxes(gridcolor=design.GRID, zeroline=False)
    return fig


def line_chart(
    frame: pd.DataFrame,
    x: str,
    series: Sequence[tuple[str, str]],
    x_format: str = "number",
    y_format: str = "score",
    height: int = 380,
    markers: bool = True,
) -> go.Figure:
    """Multi-series line chart. ``series`` is a list of (column, label) pairs."""
    fig = base_figure(height)
    hover = HOVER_VALUEFORMAT[y_format]
    for index, (column, label) in enumerate(series):
        fig.add_trace(
            go.Scatter(
                x=frame[x],
                y=frame[column],
                name=label,
                mode="lines+markers" if markers else "lines",
                line=dict(color=design.SERIES[index % len(design.SERIES)], width=2.5),
                marker=dict(size=6),
                hovertemplate=f"%{{y:{hover}}}<extra>{label}</extra>",
            )
        )
    fig.update_xaxes(tickformat=AXIS_TICKFORMAT[x_format])
    fig.update_yaxes(tickformat=AXIS_TICKFORMAT[y_format])
    return fig


def add_operating_point(fig: go.Figure, x_value: float, label: str) -> go.Figure:
    """Mark the current operating point with a dotted vertical line."""
    fig.add_vline(
        x=x_value,
        line_dash="dot",
        line_color=design.ACCENT,
        line_width=2,
        annotation_text=label,
        annotation_position="top right",
        annotation_font=dict(color=design.ACCENT, size=12),
    )
    return fig


def hbar_chart(
    frame: pd.DataFrame,
    value_column: str,
    label_column: str,
    error_column: Optional[str] = None,
    value_format: str = "score",
    height: Optional[int] = None,
) -> go.Figure:
    """Horizontal bar chart sorted so the largest value sits on top."""
    ordered = frame.sort_values(value_column, ascending=True)
    fig = base_figure(height or max(280, 26 * len(ordered) + 60))
    hover = HOVER_VALUEFORMAT[value_format]
    fig.add_trace(
        go.Bar(
            x=ordered[value_column],
            y=ordered[label_column],
            orientation="h",
            marker_color=design.NAVY,
            error_x=(
                dict(type="data", array=ordered[error_column], color=design.MUTED, thickness=1)
                if error_column is not None and error_column in ordered.columns
                else None
            ),
            hovertemplate=f"%{{y}}: %{{x:{hover}}}<extra></extra>",
        )
    )
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(tickformat=AXIS_TICKFORMAT[value_format])
    return fig


def grouped_bar_chart(
    frame: pd.DataFrame,
    x: str,
    series: Sequence[tuple[str, str]],
    y_format: str = "score",
    height: int = 380,
) -> go.Figure:
    """Grouped vertical bar chart. ``series`` is a list of (column, label) pairs."""
    fig = base_figure(height)
    hover = HOVER_VALUEFORMAT[y_format]
    for index, (column, label) in enumerate(series):
        fig.add_trace(
            go.Bar(
                x=frame[x],
                y=frame[column],
                name=label,
                marker_color=design.SERIES[index % len(design.SERIES)],
                hovertemplate=f"%{{y:{hover}}}<extra>{label}</extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=AXIS_TICKFORMAT[y_format])
    return fig


def reliability_chart(
    bins_frame: pd.DataFrame,
    predicted_column: str = "mean_predicted_probability",
    observed_column: str = "observed_low_review_rate",
    height: int = 400,
) -> go.Figure:
    """Calibration reliability diagram with a perfect-calibration reference line."""
    fig = base_figure(height)
    limit = max(
        float(bins_frame[predicted_column].max()),
        float(bins_frame[observed_column].max()),
        0.1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, limit],
            y=[0, limit],
            name="Perfect calibration",
            mode="lines",
            line=dict(color=design.MUTED, dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bins_frame[predicted_column],
            y=bins_frame[observed_column],
            name="Model",
            mode="lines+markers",
            line=dict(color=design.NAVY, width=2.5),
            marker=dict(size=8),
            hovertemplate="Predicted %{x:.2%}<br>Observed %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="Mean predicted probability", tickformat=".0%")
    fig.update_yaxes(title="Observed low-review rate", tickformat=".0%")
    return fig


def diverging_heatmap(
    pivot: pd.DataFrame,
    value_format: str = "money",
    height: int = 360,
) -> go.Figure:
    """Heatmap diverging around zero, e.g. net value across scenario grids."""
    hover = HOVER_VALUEFORMAT[value_format]
    fig = base_figure(height)
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=[str(column) for column in pivot.columns],
            y=list(pivot.index),
            zmid=0,
            colorscale=[[0.0, "#C92A2A"], [0.5, "#F8F9FA"], [1.0, "#2B8A3E"]],
            text=pivot.values,
            texttemplate=f"%{{text:{hover}}}",
            hovertemplate=f"%{{y}} at %{{x}}: %{{z:{hover}}}<extra></extra>",
            colorbar=dict(tickformat=AXIS_TICKFORMAT[value_format], thickness=12),
        )
    )
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(gridcolor="white")
    return fig
