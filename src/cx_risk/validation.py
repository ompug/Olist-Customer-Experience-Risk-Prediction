"""Validation split utilities."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .config import TARGET_COLUMN
from .preprocessing import PRIMARY_FEATURE_COLUMNS, assert_no_forbidden_columns


@dataclass(frozen=True)
class TemporalSplitSummary:
    split_timestamp_column: str
    train_rows: int
    test_rows: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_class_balance: dict[int, float]
    test_class_balance: dict[int, float]

    def as_dict(self) -> dict:
        return {
            "split_timestamp_column": self.split_timestamp_column,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_class_balance": self.train_class_balance,
            "test_class_balance": self.test_class_balance,
        }


def temporal_train_test_split(
    modeling_df: pd.DataFrame,
    timestamp_column: str = "order_purchase_timestamp",
    train_size: float = 0.80,
    feature_columns: Optional[list[str]] = None,
    return_summary: bool = False,
):
    """Split orders chronologically, training on earlier purchases and testing on later ones."""
    if timestamp_column not in modeling_df.columns:
        raise KeyError(f"Temporal split requires timestamp column: {timestamp_column}")
    if TARGET_COLUMN not in modeling_df.columns:
        raise KeyError(f"Temporal split requires target column: {TARGET_COLUMN}")
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")

    feature_columns = PRIMARY_FEATURE_COLUMNS if feature_columns is None else feature_columns
    missing_features = [column for column in feature_columns if column not in modeling_df.columns]
    if missing_features:
        raise ValueError(f"Modeling dataframe is missing primary feature columns: {missing_features}")

    sorted_df = (
        modeling_df.dropna(subset=[timestamp_column])
        .sort_values(timestamp_column, kind="mergesort")
        .reset_index(drop=True)
    )
    if sorted_df.empty:
        raise ValueError("Temporal split cannot run on an empty dataframe after dropping missing timestamps.")

    split_index = int(len(sorted_df) * train_size)
    split_index = min(max(split_index, 1), len(sorted_df) - 1)
    cutoff_timestamp = sorted_df.loc[split_index, timestamp_column]

    train_df = sorted_df.loc[sorted_df[timestamp_column] < cutoff_timestamp].copy()
    test_df = sorted_df.loc[sorted_df[timestamp_column] >= cutoff_timestamp].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Temporal split produced an empty train or test set.")

    max_train_timestamp = train_df[timestamp_column].max()
    min_test_timestamp = test_df[timestamp_column].min()
    if not max_train_timestamp < min_test_timestamp:
        raise ValueError(
            "Temporal split boundary failed: max train timestamp must be earlier than min test timestamp."
        )

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    assert_no_forbidden_columns(X_train.columns)
    assert_no_forbidden_columns(X_test.columns)
    y_train = train_df[TARGET_COLUMN].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    summary = TemporalSplitSummary(
        split_timestamp_column=timestamp_column,
        train_rows=len(train_df),
        test_rows=len(test_df),
        train_start=train_df[timestamp_column].min(),
        train_end=max_train_timestamp,
        test_start=min_test_timestamp,
        test_end=test_df[timestamp_column].max(),
        train_class_balance=y_train.value_counts(normalize=True).sort_index().to_dict(),
        test_class_balance=y_test.value_counts(normalize=True).sort_index().to_dict(),
    )

    if return_summary:
        return X_train, X_test, y_train, y_test, summary
    return X_train, X_test, y_train, y_test
