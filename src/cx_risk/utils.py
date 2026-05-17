"""Small reusable utilities for pipeline validation and artifacts."""

from pathlib import Path

import pandas as pd


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_unique_key(df: pd.DataFrame, key: str, name: str) -> None:
    if key not in df.columns:
        raise KeyError(f"{name} is missing required key column: {key}")
    if df[key].nunique(dropna=False) != len(df):
        raise ValueError(f"{name} must be unique by {key}.")


def merge_order_level(
    base_df: pd.DataFrame,
    right_df: pd.DataFrame,
    source_name: str,
    validate: str = "many_to_one",
) -> pd.DataFrame:
    """Merge an order-level table while preserving row count and order_id uniqueness."""
    require_unique_key(right_df, "order_id", source_name)
    before_rows = len(base_df)
    before_unique_orders = base_df["order_id"].nunique()

    merged = base_df.merge(right_df, on="order_id", how="left", validate=validate)

    if len(merged) != before_rows or merged["order_id"].nunique() != before_unique_orders:
        raise ValueError(
            f"{source_name} merge violated grain: rows {before_rows} -> {len(merged)}, "
            f"unique orders {before_unique_orders} -> {merged['order_id'].nunique()}."
        )
    if not merged["order_id"].is_unique:
        raise ValueError(f"{source_name} merge produced duplicate order_id values.")
    return merged


def save_table(df: pd.DataFrame, path: Path) -> Path:
    ensure_directory(path.parent)
    df.to_csv(path, index=False)
    return path

