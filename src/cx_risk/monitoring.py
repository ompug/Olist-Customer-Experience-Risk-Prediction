"""Validation and monitoring utilities for deployment-style model checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import TARGET_COLUMN
from .preprocessing import assert_no_forbidden_columns


@dataclass(frozen=True)
class RollingWindow:
    window: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def create_rolling_windows(
    modeling_df: pd.DataFrame,
    timestamp_column: str = "order_purchase_timestamp",
    n_windows: int = 4,
    initial_train_fraction: float = 0.50,
    test_fraction: float = 0.10,
) -> list[RollingWindow]:
    """Create expanding-window chronological train/test splits."""
    if timestamp_column not in modeling_df.columns:
        raise KeyError(f"Rolling windows require timestamp column: {timestamp_column}")
    if TARGET_COLUMN not in modeling_df.columns:
        raise KeyError(f"Rolling windows require target column: {TARGET_COLUMN}")
    if n_windows < 1:
        raise ValueError("n_windows must be at least 1.")
    if not 0 < initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must be between 0 and 1.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    sorted_df = (
        modeling_df.dropna(subset=[timestamp_column])
        .sort_values(timestamp_column, kind="mergesort")
        .reset_index(drop=True)
    )
    if sorted_df.empty:
        raise ValueError("Rolling windows cannot run on an empty dataframe.")

    n_rows = len(sorted_df)
    test_size = max(1, int(n_rows * test_fraction))
    initial_train_size = max(1, int(n_rows * initial_train_fraction))
    if initial_train_size + test_size > n_rows:
        raise ValueError("Initial train and test windows exceed available rows.")

    max_step_count = max(1, n_windows - 1)
    max_train_size = n_rows - test_size
    step_size = max(1, int((max_train_size - initial_train_size) / max_step_count))

    windows = []
    for window_index in range(n_windows):
        train_end_index = min(initial_train_size + window_index * step_size, max_train_size)
        test_end_index = min(train_end_index + test_size, n_rows)
        train_df = sorted_df.iloc[:train_end_index].copy()
        test_df = sorted_df.iloc[train_end_index:test_end_index].copy()
        if train_df.empty or test_df.empty:
            continue
        if not train_df[timestamp_column].max() < test_df[timestamp_column].min():
            raise ValueError("Rolling window chronology failed.")
        windows.append(
            RollingWindow(
                window=window_index + 1,
                train_df=train_df,
                test_df=test_df,
                train_start=train_df[timestamp_column].min(),
                train_end=train_df[timestamp_column].max(),
                test_start=test_df[timestamp_column].min(),
                test_end=test_df[timestamp_column].max(),
            )
        )
    if not windows:
        raise ValueError("No valid rolling windows were created.")
    return windows


def build_calibration_bins(y_true, y_proba, n_bins: int = 10) -> pd.DataFrame:
    """Build calibration-bin statistics for predicted probabilities."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    y_true = pd.Series(y_true).astype(int).reset_index(drop=True)
    y_proba = pd.Series(y_proba, dtype=float).reset_index(drop=True)
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Cannot build calibration bins for empty inputs.")

    bins = np.linspace(0, 1, n_bins + 1)
    frame = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    frame["bin"] = pd.cut(frame["y_proba"], bins=bins, include_lowest=True, labels=False)
    grouped = (
        frame.groupby("bin", dropna=False)
        .agg(
            n_orders=("y_true", "size"),
            mean_predicted_probability=("y_proba", "mean"),
            observed_low_review_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    grouped["bin_lower"] = grouped["bin"].apply(lambda value: bins[int(value)] if pd.notna(value) else np.nan)
    grouped["bin_upper"] = grouped["bin"].apply(lambda value: bins[int(value) + 1] if pd.notna(value) else np.nan)
    return grouped[
        ["bin", "bin_lower", "bin_upper", "n_orders", "mean_predicted_probability", "observed_low_review_rate"]
    ]


def expected_calibration_error(calibration_bins: pd.DataFrame) -> float:
    """Compute expected calibration error from calibration-bin output."""
    if calibration_bins.empty:
        raise ValueError("calibration_bins cannot be empty.")
    total = calibration_bins["n_orders"].sum()
    if total == 0:
        return np.nan
    weighted_gap = (
        calibration_bins["n_orders"]
        * (calibration_bins["mean_predicted_probability"] - calibration_bins["observed_low_review_rate"]).abs()
    ).sum()
    return float(weighted_gap / total)


def compute_calibration_metrics(y_true, y_proba, n_bins: int = 10) -> dict:
    """Compute calibration summary metrics."""
    bins = build_calibration_bins(y_true, y_proba, n_bins=n_bins)
    return {
        "brier_score": brier_score_loss(y_true, y_proba),
        "expected_calibration_error": expected_calibration_error(bins),
    }


def _safe_roc_auc(y_true, y_proba):
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return roc_auc_score(y_true, y_proba)


def _safe_average_precision(y_true, y_proba):
    y_true = pd.Series(y_true)
    if y_true.sum() == 0:
        return np.nan
    return average_precision_score(y_true, y_proba)


def build_segment_performance(
    scored_df: pd.DataFrame,
    segment_columns: list[str],
    min_segment_rows: int = 30,
) -> pd.DataFrame:
    """Compute performance metrics by selected diagnostic segments."""
    required = ["model", "actual_low_review", "predicted_probability", "predicted_label"]
    missing = [column for column in required if column not in scored_df.columns]
    if missing:
        raise ValueError(f"Missing scored columns: {missing}")

    rows = []
    for segment_column in segment_columns:
        if segment_column not in scored_df.columns:
            continue
        for segment_value, segment_df in scored_df.groupby(segment_column, dropna=False):
            if len(segment_df) < min_segment_rows:
                continue
            y_true = segment_df["actual_low_review"].astype(int)
            y_pred = segment_df["predicted_label"].astype(int)
            y_proba = segment_df["predicted_probability"].astype(float)
            rows.append(
                {
                    "model": segment_df["model"].iloc[0],
                    "segment_column": segment_column,
                    "segment_value": segment_value,
                    "n_orders": len(segment_df),
                    "low_review_rate": float(y_true.mean()),
                    "predicted_positive_rate": float(y_pred.mean()),
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "roc_auc": _safe_roc_auc(y_true, y_proba),
                    "average_precision": _safe_average_precision(y_true, y_proba),
                }
            )
    return pd.DataFrame(rows)


def add_monitoring_segments(frame: pd.DataFrame) -> pd.DataFrame:
    """Add coarse diagnostic segment columns from leakage-safe features."""
    output = frame.copy()
    seller_rate = output.get("seller_prior_low_review_rate", pd.Series(np.nan, index=output.index))
    output["seller_risk_tier"] = pd.cut(
        seller_rate.fillna(-1),
        bins=[-2, -0.001, 0.10, 0.20, 1.0],
        labels=["no_history", "low", "medium", "high"],
    ).astype(str)

    basket_value = output.get("total_basket_value", pd.Series(np.nan, index=output.index))
    if basket_value.notna().nunique() >= 3:
        output["order_value_tier"] = pd.qcut(
            basket_value,
            q=3,
            labels=["low", "medium", "high"],
            duplicates="drop",
        ).astype(str)
    else:
        output["order_value_tier"] = "unknown"

    output["payment_profile"] = np.select(
        [
            output.get("has_boleto", pd.Series(0, index=output.index)).fillna(0).astype(bool),
            output.get("high_installment_flag", pd.Series(0, index=output.index)).fillna(0).astype(bool),
            output.get("has_credit_card", pd.Series(0, index=output.index)).fillna(0).astype(bool),
        ],
        ["boleto", "high_installment", "credit_card"],
        default="other",
    )
    return output


def build_feature_drift_report(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_categorical_levels: int = 25,
) -> pd.DataFrame:
    """Compare train and test feature distributions for monitoring drift."""
    assert_no_forbidden_columns(X_train.columns, include_raw_ids=True)
    assert_no_forbidden_columns(X_test.columns, include_raw_ids=True)
    rows = []

    numeric_columns = X_train.select_dtypes(include=["number", "bool"]).columns
    for column in numeric_columns:
        train_values = pd.to_numeric(X_train[column], errors="coerce")
        test_values = pd.to_numeric(X_test[column], errors="coerce")
        train_mean = train_values.mean()
        test_mean = test_values.mean()
        train_std = train_values.std()
        standardized_delta = abs(test_mean - train_mean) / train_std if train_std and train_std > 0 else np.nan
        rows.append(
            {
                "feature": column,
                "feature_type": "numeric",
                "train_value": train_mean,
                "test_value": test_mean,
                "absolute_delta": abs(test_mean - train_mean),
                "standardized_delta": standardized_delta,
                "train_missing_rate": train_values.isna().mean(),
                "test_missing_rate": test_values.isna().mean(),
            }
        )

    categorical_columns = X_train.select_dtypes(include=["object", "category", "string"]).columns
    for column in categorical_columns:
        train_counts = X_train[column].fillna("missing").astype(str).value_counts(normalize=True)
        test_counts = X_test[column].fillna("missing").astype(str).value_counts(normalize=True)
        levels = train_counts.head(max_categorical_levels).index.union(test_counts.head(max_categorical_levels).index)
        max_delta = 0.0
        top_level = None
        train_share = 0.0
        test_share = 0.0
        for level in levels:
            delta = abs(test_counts.get(level, 0.0) - train_counts.get(level, 0.0))
            if delta >= max_delta:
                max_delta = delta
                top_level = level
                train_share = train_counts.get(level, 0.0)
                test_share = test_counts.get(level, 0.0)
        rows.append(
            {
                "feature": column,
                "feature_type": "categorical",
                "train_value": train_share,
                "test_value": test_share,
                "absolute_delta": max_delta,
                "standardized_delta": np.nan,
                "train_missing_rate": X_train[column].isna().mean(),
                "test_missing_rate": X_test[column].isna().mean(),
                "largest_shift_level": top_level,
            }
        )
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    return report.sort_values(["standardized_delta", "absolute_delta"], ascending=False, na_position="last")
