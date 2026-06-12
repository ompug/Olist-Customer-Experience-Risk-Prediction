"""Cached artifact loaders and the shared missing-artifact empty state.

All artifact paths come from ``cx_risk.config`` so the dashboard and the
pipeline never disagree about where outputs live.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from cx_risk.config import MODEL_DIR, PROJECT_ROOT, TABLES_DIR

TIME_VALIDATION = TABLES_DIR / "time_validation_metrics.csv"
HISTORY_VALIDATION = TABLES_DIR / "time_validation_with_history_metrics.csv"
INTERVENTION = TABLES_DIR / "intervention_simulation.csv"
THRESHOLD = TABLES_DIR / "threshold_analysis.csv"
MODEL_REGISTRY = MODEL_DIR / "model_registry.json"
RISK_QUEUE = TABLES_DIR / "risk_queue.csv"
GLOBAL_IMPORTANCE = TABLES_DIR / "global_feature_importance.csv"
BUSINESS_VALUE = TABLES_DIR / "business_value_simulation.csv"
BUSINESS_VALUE_SENSITIVITY = TABLES_DIR / "business_value_sensitivity.csv"
ROLLING_BACKTEST = TABLES_DIR / "rolling_backtest_metrics.csv"
CALIBRATION_METRICS = TABLES_DIR / "calibration_metrics.csv"
CALIBRATION_BINS = TABLES_DIR / "calibration_bins.csv"
SEGMENT_PERFORMANCE = TABLES_DIR / "segment_performance.csv"
FEATURE_DRIFT = TABLES_DIR / "feature_drift_report.csv"
MODEL_LIFT_EXPERIMENTS = TABLES_DIR / "model_lift_experiments.csv"
MODEL_LIFT_TOPK = TABLES_DIR / "model_lift_topk.csv"
MODEL_LIFT_FEATURE_SETS = TABLES_DIR / "model_lift_feature_sets.csv"

ARTIFACT_MAKE_TARGETS = {
    TIME_VALIDATION.name: "make time-validation",
    HISTORY_VALIDATION.name: "make time-validation-history",
    INTERVENTION.name: "make intervention",
    THRESHOLD.name: "make intervention",
    MODEL_REGISTRY.name: "make risk-queue",
    RISK_QUEUE.name: "make risk-queue",
    GLOBAL_IMPORTANCE.name: "make risk-queue",
    BUSINESS_VALUE.name: "make business-value",
    BUSINESS_VALUE_SENSITIVITY.name: "make business-value",
    ROLLING_BACKTEST.name: "make monitoring",
    CALIBRATION_METRICS.name: "make monitoring",
    CALIBRATION_BINS.name: "make monitoring",
    SEGMENT_PERFORMANCE.name: "make monitoring",
    FEATURE_DRIFT.name: "make monitoring",
    MODEL_LIFT_EXPERIMENTS.name: "make model-lift",
    MODEL_LIFT_TOPK.name: "make model-lift",
    MODEL_LIFT_FEATURE_SETS.name: "make model-lift",
}


@st.cache_data(ttl=120, show_spinner=False)
def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=120, show_spinner=False)
def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def missing_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def require_artifacts(paths: list[Path], context: str = "") -> bool:
    """Render one clean empty-state card when artifacts are missing.

    Returns True when every artifact exists, so pages can simply early-return.
    """
    missing = missing_paths(paths)
    if not missing:
        return True

    commands: list[str] = []
    for path in missing:
        command = ARTIFACT_MAKE_TARGETS.get(path.name, "see README")
        if command not in commands:
            commands.append(command)

    with st.container(border=True):
        st.subheader("Pipeline artifacts needed")
        message = "This page reads precomputed artifacts that have not been generated yet."
        if context:
            message = f"{message} {context}"
        st.markdown(message)
        st.code("\n".join(commands), language="bash")
        st.caption(
            "Run from the project root with the raw Olist CSVs in `olist_data/`. "
            "The full pipeline guide lives on the Monitoring & Methodology page."
        )
    return False


def relative_to_project(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))
