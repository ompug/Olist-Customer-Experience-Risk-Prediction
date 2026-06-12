"""Business-value scenario simulation for intervention queues."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_INTERVENTION_ASSUMPTIONS = [
    {
        "intervention_type": "Proactive customer outreach",
        "cost_per_order": 1.50,
        "expected_save_rate": 0.08,
        "value_per_saved_low_review": 35.00,
    },
    {
        "intervention_type": "Seller escalation",
        "cost_per_order": 4.00,
        "expected_save_rate": 0.12,
        "value_per_saved_low_review": 45.00,
    },
    {
        "intervention_type": "Shipping check",
        "cost_per_order": 3.00,
        "expected_save_rate": 0.10,
        "value_per_saved_low_review": 40.00,
    },
    {
        "intervention_type": "Refund voucher",
        "cost_per_order": 8.00,
        "expected_save_rate": 0.16,
        "value_per_saved_low_review": 55.00,
    },
    {
        "intervention_type": "Premium support review",
        "cost_per_order": 6.00,
        "expected_save_rate": 0.14,
        "value_per_saved_low_review": 50.00,
    },
]

DEFAULT_SAVE_RATE_MULTIPLIERS = [0.50, 0.75, 1.00, 1.25, 1.50]


def default_assumptions_table() -> pd.DataFrame:
    """Return default intervention business assumptions."""
    return pd.DataFrame(DEFAULT_INTERVENTION_ASSUMPTIONS)


def validate_assumptions(assumptions: pd.DataFrame) -> None:
    """Validate intervention assumption fields before scenario simulation."""
    required = [
        "intervention_type",
        "cost_per_order",
        "expected_save_rate",
        "value_per_saved_low_review",
    ]
    missing = [column for column in required if column not in assumptions.columns]
    if missing:
        raise ValueError(f"Missing assumption columns: {missing}")
    if (assumptions["cost_per_order"] < 0).any():
        raise ValueError("cost_per_order must be nonnegative.")
    if ((assumptions["expected_save_rate"] < 0) | (assumptions["expected_save_rate"] > 1)).any():
        raise ValueError("expected_save_rate must be between 0 and 1.")
    if (assumptions["value_per_saved_low_review"] <= 0).any():
        raise ValueError("value_per_saved_low_review must be positive.")


def calculate_business_value(
    n_flagged: float,
    true_positives: float,
    cost_per_order: float,
    expected_save_rate: float,
    value_per_saved_low_review: float,
) -> dict:
    """Calculate expected business value for one queue and intervention assumption."""
    assumptions = pd.DataFrame(
        [
            {
                "intervention_type": "single",
                "cost_per_order": cost_per_order,
                "expected_save_rate": expected_save_rate,
                "value_per_saved_low_review": value_per_saved_low_review,
            }
        ]
    )
    validate_assumptions(assumptions)
    if n_flagged < 0 or true_positives < 0:
        raise ValueError("n_flagged and true_positives must be nonnegative.")

    expected_saved = true_positives * expected_save_rate
    intervention_cost = n_flagged * cost_per_order
    gross_value = expected_saved * value_per_saved_low_review
    net_value = gross_value - intervention_cost
    roi = net_value / intervention_cost if intervention_cost > 0 else np.nan
    denominator = true_positives * value_per_saved_low_review
    break_even_save_rate = intervention_cost / denominator if denominator > 0 else np.nan

    return {
        "expected_saved_low_reviews": expected_saved,
        "intervention_cost": intervention_cost,
        "gross_value": gross_value,
        "net_value": net_value,
        "roi": roi,
        "break_even_save_rate": break_even_save_rate,
    }


def build_business_value_table(
    intervention_table: pd.DataFrame,
    assumptions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Cross-join intervention operating points with business assumptions."""
    if assumptions is None:
        assumptions = default_assumptions_table()
    validate_assumptions(assumptions)

    required = ["model", "n_flagged", "true_positives", "precision", "recall", "lift_over_random"]
    missing = [column for column in required if column not in intervention_table.columns]
    if missing:
        raise ValueError(f"Missing intervention columns: {missing}")

    rows = []
    for _, intervention_row in intervention_table.iterrows():
        for _, assumption_row in assumptions.iterrows():
            value = calculate_business_value(
                n_flagged=intervention_row["n_flagged"],
                true_positives=intervention_row["true_positives"],
                cost_per_order=assumption_row["cost_per_order"],
                expected_save_rate=assumption_row["expected_save_rate"],
                value_per_saved_low_review=assumption_row["value_per_saved_low_review"],
            )
            rows.append(
                {
                    "model": intervention_row["model"],
                    "selection_strategy": intervention_row.get("selection_strategy"),
                    "top_frac": intervention_row.get("top_frac"),
                    "probability_threshold": intervention_row.get("probability_threshold"),
                    "n_orders": intervention_row.get("n_orders"),
                    "n_flagged": intervention_row["n_flagged"],
                    "true_positives": intervention_row["true_positives"],
                    "precision": intervention_row["precision"],
                    "recall": intervention_row["recall"],
                    "lift_over_random": intervention_row["lift_over_random"],
                    "intervention_type": assumption_row["intervention_type"],
                    "cost_per_order": assumption_row["cost_per_order"],
                    "expected_save_rate": assumption_row["expected_save_rate"],
                    "value_per_saved_low_review": assumption_row["value_per_saved_low_review"],
                    **value,
                }
            )
    return pd.DataFrame(rows)


def build_sensitivity_table(
    intervention_table: pd.DataFrame,
    assumptions: pd.DataFrame | None = None,
    save_rate_multipliers: list[float] | None = None,
) -> pd.DataFrame:
    """Build scenario rows across expected save-rate multipliers."""
    if assumptions is None:
        assumptions = default_assumptions_table()
    if save_rate_multipliers is None:
        save_rate_multipliers = DEFAULT_SAVE_RATE_MULTIPLIERS
    if any(multiplier < 0 for multiplier in save_rate_multipliers):
        raise ValueError("Save-rate multipliers must be nonnegative.")

    frames = []
    for multiplier in save_rate_multipliers:
        adjusted = assumptions.copy()
        adjusted["expected_save_rate"] = (adjusted["expected_save_rate"] * multiplier).clip(upper=1.0)
        frame = build_business_value_table(intervention_table, adjusted)
        frame.insert(0, "save_rate_multiplier", multiplier)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
