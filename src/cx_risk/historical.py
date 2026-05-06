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
