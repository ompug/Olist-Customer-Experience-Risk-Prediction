from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cx_risk.config import TARGET_COLUMN
from cx_risk.monitoring import (
    add_monitoring_segments,
    build_calibration_bins,
    build_feature_drift_report,
    build_segment_performance,
    compute_calibration_metrics,
    create_rolling_windows,
    expected_calibration_error,
)


def test_create_rolling_windows_preserves_chronology():
    frame = pd.DataFrame(
        {
            "order_purchase_timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
            TARGET_COLUMN: [0, 1] * 50,
            "feature": range(100),
        }
    )

    windows = create_rolling_windows(frame, n_windows=3, initial_train_fraction=0.50, test_fraction=0.10)

    assert len(windows) == 3
    for window in windows:
        assert not window.train_df.empty
        assert not window.test_df.empty
        assert window.train_end < window.test_start


def test_calibration_bins_and_ece_are_bounded():
    bins = build_calibration_bins([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_bins=2)
    ece = expected_calibration_error(bins)
    metrics = compute_calibration_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_bins=2)

    assert bins["n_orders"].sum() == 4
    assert 0 <= ece <= 1
    assert 0 <= metrics["expected_calibration_error"] <= 1
    assert 0 <= metrics["brier_score"] <= 1


def test_segment_performance_emits_metrics_for_large_segments():
    scored = pd.DataFrame(
        {
            "model": ["Model"] * 6,
            "segment": ["A", "A", "A", "B", "B", "B"],
            "actual_low_review": [1, 0, 1, 0, 0, 1],
            "predicted_probability": [0.9, 0.2, 0.8, 0.1, 0.4, 0.7],
            "predicted_label": [1, 0, 1, 0, 0, 1],
        }
    )

    result = build_segment_performance(scored, ["segment"], min_segment_rows=1)

    assert set(result["segment_value"]) == {"A", "B"}
    assert {"precision", "recall", "roc_auc", "average_precision"}.issubset(result.columns)


def test_add_monitoring_segments_creates_expected_columns():
    frame = pd.DataFrame(
        {
            "seller_prior_low_review_rate": [np.nan, 0.05, 0.25],
            "total_basket_value": [10, 50, 100],
            "has_boleto": [0, 1, 0],
            "high_installment_flag": [0, 0, 1],
            "has_credit_card": [1, 0, 1],
        }
    )

    result = add_monitoring_segments(frame)

    assert {"seller_risk_tier", "order_value_tier", "payment_profile"}.issubset(result.columns)


def test_feature_drift_report_handles_numeric_and_categorical_features():
    X_train = pd.DataFrame({"num": [1, 2, 3, 4], "cat": ["a", "a", "b", "b"]})
    X_test = pd.DataFrame({"num": [10, 11, 12, 13], "cat": ["a", "c", "c", "c"]})

    report = build_feature_drift_report(X_train, X_test)

    assert set(report["feature"]) == {"num", "cat"}
    assert {"absolute_delta", "train_missing_rate", "test_missing_rate"}.issubset(report.columns)
