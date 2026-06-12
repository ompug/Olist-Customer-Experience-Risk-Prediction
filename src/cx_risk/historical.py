"""Leakage-safe historical aggregate features."""

import numpy as np
import pandas as pd

from .config import TARGET_COLUMN

HISTORICAL_FEATURE_COLUMNS = [
    "seller_prior_order_count",
    "seller_prior_low_review_rate",
    "category_prior_order_count",
    "category_prior_low_review_rate",
    "customer_state_prior_order_count",
    "customer_state_prior_low_review_rate",
]

ENHANCED_HISTORICAL_FEATURE_COLUMNS = [
    "seller_state_prior_order_count",
    "seller_state_prior_low_review_rate",
    "category_state_prior_order_count",
    "category_state_prior_low_review_rate",
    "payment_profile_prior_order_count",
    "payment_profile_prior_low_review_rate",
    "product_volume_tier_prior_order_count",
    "product_volume_tier_prior_low_review_rate",
    "purchase_month_prior_order_count",
    "purchase_month_prior_low_review_rate",
]


def _add_prior_group_rate_features(
    frame: pd.DataFrame,
    group_column: str,
    timestamp_column: str,
    target_column: str,
    prefix: str,
) -> pd.DataFrame:
    """Add prior count/rate using only strictly earlier timestamp buckets."""
    required_columns = [group_column, timestamp_column, target_column]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns for historical feature calculation: {missing}")

    output = frame.copy()
    count_column = f"{prefix}_prior_order_count"
    rate_column = f"{prefix}_prior_low_review_rate"
    prior_sum_column = f"{prefix}_prior_low_review_sum"

    output[count_column] = 0
    output[rate_column] = np.nan

    valid_mask = output[group_column].notna() & output[timestamp_column].notna()
    if not valid_mask.any():
        return output

    buckets = (
        output.loc[valid_mask]
        .groupby([group_column, timestamp_column], as_index=False)
        .agg(
            current_timestamp_order_count=(target_column, "size"),
            current_timestamp_low_review_sum=(target_column, "sum"),
        )
        .sort_values([group_column, timestamp_column], kind="mergesort")
    )
    grouped = buckets.groupby(group_column, sort=False)
    buckets[count_column] = (
        grouped["current_timestamp_order_count"].cumsum()
        - buckets["current_timestamp_order_count"]
    )
    buckets[prior_sum_column] = (
        grouped["current_timestamp_low_review_sum"].cumsum()
        - buckets["current_timestamp_low_review_sum"]
    )
    buckets[rate_column] = np.where(
        buckets[count_column] > 0,
        buckets[prior_sum_column] / buckets[count_column],
        np.nan,
    )

    output = output.merge(
        buckets[[group_column, timestamp_column, count_column, rate_column]],
        on=[group_column, timestamp_column],
        how="left",
        suffixes=("", "_computed"),
        validate="many_to_one",
    )
    computed_count = f"{count_column}_computed"
    computed_rate = f"{rate_column}_computed"
    output[count_column] = output[computed_count].fillna(0).astype(int)
    output[rate_column] = output[computed_rate]
    return output.drop(columns=[computed_count, computed_rate])


def add_historical_risk_features(
    frame: pd.DataFrame,
    timestamp_column: str = "order_purchase_timestamp",
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Add prior seller, category, and customer-state risk features.

    Seller history uses the order-level `primary_seller_id`, which is the mode
    seller ID for the order. That identifier is only a grouping key and is not
    included in the model predictor list.
    """
    output = frame.copy()
    output = _add_prior_group_rate_features(
        output,
        group_column="primary_seller_id",
        timestamp_column=timestamp_column,
        target_column=target_column,
        prefix="seller",
    )
    output = _add_prior_group_rate_features(
        output,
        group_column="primary_category",
        timestamp_column=timestamp_column,
        target_column=target_column,
        prefix="category",
    )
    output = _add_prior_group_rate_features(
        output,
        group_column="customer_state",
        timestamp_column=timestamp_column,
        target_column=target_column,
        prefix="customer_state",
    )
    return output


def add_enhanced_historical_risk_features(
    frame: pd.DataFrame,
    timestamp_column: str = "order_purchase_timestamp",
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Add additional strictly-prior risk aggregates for model-lift experiments."""
    output = frame.copy()
    output["seller_state_history_key"] = output["primary_seller_state"]
    output["category_state_history_key"] = (
        output["primary_category"].astype("string").fillna("missing")
        + "|"
        + output["customer_state"].astype("string").fillna("missing")
    )
    output["payment_profile_history_key"] = np.select(
        [
            output.get("has_boleto", pd.Series(0, index=output.index)).fillna(0).astype(bool),
            output.get("high_installment_flag", pd.Series(0, index=output.index)).fillna(0).astype(bool),
            output.get("has_credit_card", pd.Series(0, index=output.index)).fillna(0).astype(bool),
        ],
        ["boleto", "high_installment", "credit_card"],
        default="other",
    )
    output["product_volume_tier_history_key"] = pd.cut(
        output.get("avg_product_volume_cm3_safe", pd.Series(np.nan, index=output.index)),
        bins=[-np.inf, 1_000, 10_000, 50_000, np.inf],
        labels=["tiny", "small", "medium", "large"],
    ).astype("string")
    output["purchase_month_history_key"] = output["purchase_month"].astype("string")

    output = _add_prior_group_rate_features(
        output,
        group_column="seller_state_history_key",
        timestamp_column=timestamp_column,
        target_column=target_column,
        prefix="seller_state",
    )
    output = _add_prior_group_rate_features(
        output,
        group_column="category_state_history_key",
        timestamp_column=timestamp_column,
        target_column=target_column,
        prefix="category_state",
    )
    output = _add_prior_group_rate_features(
        output,
        group_column="payment_profile_history_key",
        timestamp_column=timestamp_column,
        target_column=target_column,
        prefix="payment_profile",
    )
    output = _add_prior_group_rate_features(
        output,
        group_column="product_volume_tier_history_key",
        timestamp_column=timestamp_column,
        target_column=target_column,
        prefix="product_volume_tier",
    )
    output = _add_prior_group_rate_features(
        output,
        group_column="purchase_month_history_key",
        timestamp_column=timestamp_column,
        target_column=target_column,
        prefix="purchase_month",
    )
    return output
