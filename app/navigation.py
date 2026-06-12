"""Single source of truth for page definitions and navigation groups.

Both the entry point (st.navigation) and cross-page links (st.page_link)
use these Page objects, so titles, icons, and URLs never drift apart.
"""

import streamlit as st

from views import (
    business_value,
    explainability,
    home,
    model_performance,
    monitoring,
    risk_queue,
    welcome,
)

WELCOME = st.Page(
    welcome.render,
    title="Welcome",
    icon=":material/waving_hand:",
    url_path="welcome",
    default=True,
)
RESULTS_OVERVIEW = st.Page(
    home.render,
    title="Results Overview",
    icon=":material/insights:",
    url_path="overview",
)
INTERVENTION_QUEUE = st.Page(
    risk_queue.render,
    title="Intervention Queue",
    icon=":material/checklist:",
    url_path="intervention-queue",
)
MODEL_PERFORMANCE = st.Page(
    model_performance.render,
    title="Model Performance",
    icon=":material/monitoring:",
    url_path="model-performance",
)
WHY_FLAGGED = st.Page(
    explainability.render,
    title="Why Orders Get Flagged",
    icon=":material/psychology:",
    url_path="why-orders-get-flagged",
)
ROI_PLANNER = st.Page(
    business_value.render,
    title="ROI Planner",
    icon=":material/payments:",
    url_path="roi-planner",
)
RELIABILITY = st.Page(
    monitoring.render,
    title="Reliability & Methods",
    icon=":material/health_and_safety:",
    url_path="reliability-and-methods",
)

GROUPS = {
    "Start here": [WELCOME],
    "Operate": [RESULTS_OVERVIEW, INTERVENTION_QUEUE],
    "Evaluate": [MODEL_PERFORMANCE, WHY_FLAGGED],
    "Plan & Monitor": [ROI_PLANNER, RELIABILITY],
}
