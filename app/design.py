"""Design tokens and brand strings shared across the dashboard."""

PRODUCT_NAME = "Lighthouse"
PRODUCT_TAGLINE = "See bad customer experiences coming - before the review lands."
PRODUCT_POSITIONING = (
    "An order-level risk radar for e-commerce: it scores every order for low-review risk "
    "the moment it is placed, and turns those scores into a daily intervention queue your "
    "support team can actually work."
)
REPO_URL = "https://github.com/ompug/Olist-Customer-Experience-Risk-Prediction"

NAVY = "#1F3A5F"
ACCENT = "#E8590C"
GREEN = "#2B8A3E"
TEXT = "#1B2333"
MUTED = "#5C677D"
GRID = "#E3E8EF"

SERIES = [NAVY, ACCENT, GREEN, "#7048E8", "#0B7285", "#C92A2A"]

RISK_BAND_ORDER = ["critical", "high", "medium", "low"]

RISK_BAND_COLORS = {
    "critical": "#C92A2A",
    "high": "#E8590C",
    "medium": "#B07D10",
    "low": "#2B8A3E",
}

RISK_BAND_BACKGROUNDS = {
    "critical": "#FBE7E7",
    "high": "#FDEEE3",
    "medium": "#FBF2DC",
    "low": "#E8F5EC",
}
