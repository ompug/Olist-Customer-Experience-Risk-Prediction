"""Lighthouse dashboard entry point.

Thin shell: navigation, theme, and global chrome. Page definitions live in
``app/navigation.py``, content in ``app/views/``, and shared building blocks
in ``app/components/``.
"""

from pathlib import Path
import sys

_APP_DIR = Path(__file__).resolve().parent
_SRC_DIR = _APP_DIR.parent / "src"
for path in (str(_SRC_DIR), str(_APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st

import design

_LOGO_PATH = _APP_DIR / "assets" / "lighthouse_logo.png"

st.set_page_config(
    page_title=f"{design.PRODUCT_NAME} - CX Risk Radar",
    page_icon=str(_LOGO_PATH) if _LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.layout import inject_css
from navigation import GROUPS

inject_css()

navigation = st.navigation(GROUPS)

with st.sidebar:
    if _LOGO_PATH.exists():
        st.image(str(_LOGO_PATH), width=88)
    st.markdown(f"**{design.PRODUCT_NAME}**")
    st.caption(design.PRODUCT_TAGLINE)
    st.caption(
        "A leakage-aware ML product built on the public Brazilian Olist e-commerce dataset."
    )

navigation.run()
