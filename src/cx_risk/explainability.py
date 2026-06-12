"""Explainability helpers for model drivers and queue reason codes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import RANDOM_STATE
from .preprocessing import assert_no_forbidden_columns

REASON_COLUMNS = ["reason_1", "reason_2", "reason_3", "reason_summary"]


def _is_true(value) -> bool:
    return pd.notna(value) and bool(value)


def generate_reason_codes(row: pd.Series, max_reasons: int = 3) -> list[str]:
    """Generate readable, leakage-safe business reason codes for a queued order."""
    reasons = []

    seller_count = row.get("seller_prior_order_count")
    seller_rate = row.get("seller_prior_low_review_rate")
    if (
        pd.notna(seller_count)
        and pd.notna(seller_rate)
        and seller_count >= 5
        and seller_rate >= 0.20
    ):
        reasons.append("Seller has elevated prior low-review rate")

    category_rate = row.get("category_prior_low_review_rate")
    if pd.notna(category_rate) and category_rate >= 0.15:
        reasons.append("Category has elevated prior low-review rate")

    estimated_days = row.get("estimated_delivery_days")
    if pd.notna(estimated_days) and estimated_days >= 20:
        reasons.append("Long estimated delivery window")

    freight_share = row.get("freight_share")
    if pd.notna(freight_share) and freight_share >= 0.30:
        reasons.append("High freight share")

    if _is_true(row.get("multi_seller_flag")):
        reasons.append("Multi-seller order")

    if _is_true(row.get("high_installment_flag")):
        reasons.append("High-installment purchase")

    if _is_true(row.get("expensive_order_flag")):
        reasons.append("High basket value")

    if not reasons:
        reasons.append("No dominant rule-based reason")

    return reasons[:max_reasons]


def add_reason_codes_to_queue(queue_df: pd.DataFrame, max_reasons: int = 3) -> pd.DataFrame:
    """Add reason-code columns to an existing risk queue without changing row order."""
    output = queue_df.copy()
    reason_lists = output.apply(lambda row: generate_reason_codes(row, max_reasons=max_reasons), axis=1)
    for index in range(max_reasons):
        output[f"reason_{index + 1}"] = reason_lists.apply(
            lambda reasons: reasons[index] if index < len(reasons) else ""
        )
    output["reason_summary"] = reason_lists.apply(lambda reasons: "; ".join(reasons))
    return output


def compute_permutation_importance_table(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
    scoring: str = "average_precision",
) -> pd.DataFrame:
    """Compute model-agnostic permutation importance on raw pipeline inputs."""
    if X.empty:
        raise ValueError("Cannot compute permutation importance for an empty feature frame.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")
    assert_no_forbidden_columns(X.columns, include_raw_ids=True)

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "model": model_name,
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "scoring": scoring,
        }
    )
    table = table.sort_values(
        ["importance_mean", "importance_std"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table[["model", "rank", "feature", "importance_mean", "importance_std", "scoring"]]


def build_global_importance_tables(
    models: dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Compute permutation importance for all fitted models."""
    frames = [
        compute_permutation_importance_table(
            model,
            X,
            y,
            model_name=model_name,
            n_repeats=n_repeats,
        )
        for model_name, model in models.items()
    ]
    return pd.concat(frames, ignore_index=True)
