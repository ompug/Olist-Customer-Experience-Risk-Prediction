"""Layout primitives: page headers, KPI rows, callouts, and risk badges."""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

import design

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


@lru_cache(maxsize=8)
def asset_data_uri(filename: str) -> Optional[str]:
    """Return a data URI for a bundled asset, or None when it is missing."""
    path = ASSETS_DIR / filename
    if not path.exists():
        return None
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"

_CSS = f"""
<style>
.block-container {{
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}}

.cx-subtitle {{
    color: {design.MUTED};
    font-size: 1.05rem;
    margin-top: -0.4rem;
    margin-bottom: 1.4rem;
    max-width: 60rem;
}}

.cx-callout {{
    border-left: 4px solid {design.NAVY};
    background: #F4F6FA;
    color: {design.MUTED};
    padding: 0.65rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
    margin: 0.8rem 0 1.2rem 0;
}}

.cx-badge {{
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.35rem;
}}

.cx-technote {{
    color: {design.MUTED};
    font-size: 0.88rem;
    font-style: italic;
    margin-top: -1.1rem;
    margin-bottom: 1.4rem;
    max-width: 60rem;
}}

.cx-hero {{
    background: linear-gradient(135deg, {design.NAVY} 0%, #2C4F7C 100%);
    border-radius: 14px;
    padding: 3.4rem 3rem 3rem 3rem;
    margin-bottom: 1.6rem;
    color: #FFFFFF;
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 300px;
}}

.cx-hero-name {{
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.05;
    margin: 0 0 0.6rem 0;
    color: #FFFFFF;
}}

.cx-hero-tagline {{
    font-size: 1.35rem;
    font-weight: 500;
    color: #DCE6F5;
    margin: 0 0 1rem 0;
}}

.cx-hero-positioning {{
    font-size: 1rem;
    color: #AFC3DF;
    max-width: 46rem;
    margin: 0;
}}

.cx-stat {{
    background: #F4F6FA;
    border: 1px solid {design.GRID};
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    height: 100%;
}}

.cx-stat-value {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {design.NAVY};
    line-height: 1.1;
    margin-bottom: 0.3rem;
}}

.cx-stat-label {{
    font-size: 0.92rem;
    color: {design.MUTED};
}}

.cx-footer {{
    border-top: 1px solid {design.GRID};
    margin-top: 2.4rem;
    padding-top: 1rem;
    color: {design.MUTED};
    font-size: 0.88rem;
}}

[data-testid="stMetric"] {{
    background: #F4F6FA;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}}

[data-testid="stMetricLabel"] {{
    color: {design.MUTED};
}}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: Optional[str] = None, tech_note: Optional[str] = None) -> None:
    """Page header with a business-first subtitle and an optional technical second line."""
    st.title(title)
    if subtitle:
        st.markdown(f'<p class="cx-subtitle">{subtitle}</p>', unsafe_allow_html=True)
    if tech_note:
        st.markdown(f'<p class="cx-technote">{tech_note}</p>', unsafe_allow_html=True)


def hero(name: str, tagline: str, positioning: str, background_asset: Optional[str] = None) -> None:
    """Render the landing-page hero band, optionally over a brand illustration."""
    style = ""
    background = asset_data_uri(background_asset) if background_asset else None
    if background:
        style = (
            "background:"
            "linear-gradient(100deg, rgba(24,42,74,0.94) 30%, rgba(24,42,74,0.55) 58%, rgba(24,42,74,0.08)),"
            f"url('{background}') center right / cover no-repeat;"
        )
    st.markdown(
        f"""
        <div class="cx-hero" style="{style}">
            <p class="cx-hero-name">{name}</p>
            <p class="cx-hero-tagline">{tagline}</p>
            <p class="cx-hero-positioning">{positioning}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(value: str, label: str) -> None:
    """Render one large proof-strip stat card."""
    st.markdown(
        f"""
        <div class="cx-stat">
            <div class="cx-stat-value">{value}</div>
            <div class="cx-stat-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer(text: str) -> None:
    """Render the muted footer strip."""
    st.markdown(f'<div class="cx-footer">{text}</div>', unsafe_allow_html=True)


def kpi_row(items: list[dict]) -> None:
    """Render a row of KPI metrics. Items: {label, value, delta?, help?}."""
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        column.metric(
            label=item["label"],
            value=item["value"],
            delta=item.get("delta"),
            help=item.get("help"),
        )


def honesty_note(text: str) -> None:
    """Render the standard caveat callout used across pages."""
    st.markdown(f'<div class="cx-callout">{text}</div>', unsafe_allow_html=True)


def risk_badge(band: str) -> str:
    """Return badge HTML for a risk band, using the shared color system."""
    band_key = str(band).lower()
    foreground = design.RISK_BAND_COLORS.get(band_key, design.MUTED)
    background = design.RISK_BAND_BACKGROUNDS.get(band_key, "#EEF1F5")
    return (
        f'<span class="cx-badge" style="color:{foreground};background:{background};">'
        f"{band_key.capitalize()}</span>"
    )


def style_risk_band_column(frame: pd.DataFrame, column: str = "risk_band"):
    """Return a pandas Styler coloring the risk-band column."""

    def _style(value: object) -> str:
        band_key = str(value).lower()
        if band_key not in design.RISK_BAND_COLORS:
            return ""
        return (
            f"color:{design.RISK_BAND_COLORS[band_key]};"
            f"background-color:{design.RISK_BAND_BACKGROUNDS[band_key]};"
            "font-weight:600;"
        )

    styler = frame.style.map(_style, subset=[column]) if column in frame.columns else frame.style
    numeric_columns = frame.select_dtypes(include="number").columns
    return styler.format({col: "{:.4f}" for col in numeric_columns}, na_rep="-")
