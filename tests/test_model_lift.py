from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cx_risk.config import TARGET_COLUMN
from cx_risk.historical import add_enhanced_historical_risk_features
from cx_risk.model_lift import build_topk_ranking_table, ranking_metrics_at_fraction
from cx_risk.preprocessing import ENHANCED_HISTORICAL_FEATURE_COLUMNS, get_primary_feature_columns


def test_enhanced_historical_features_exclude_same_timestamp_and_future_targets():
    frame = pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d"],
            "order_purchase_timestamp": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-02 10:00:00",
                    "2024-01-02 10:00:00",
                    "2024-01-03 10:00:00",
                ]
            ),
            "primary_seller_state": ["SP", "SP", "SP", "SP"],
            "primary_category": ["books", "books", "books", "books"],
            "customer_state": ["RJ", "RJ", "RJ", "RJ"],
            "has_boleto": [1, 1, 1, 1],
            "high_installment_flag": [0, 0, 0, 0],
            "has_credit_card": [0, 0, 0, 0],
            "avg_product_volume_cm3_safe": [500, 500, 500, 500],
            "purchase_month": [1, 1, 1, 1],
            TARGET_COLUMN: [1, 0, 1, 0],
        }
    )

    result = add_enhanced_historical_risk_features(frame)
    first_order = result.loc[result["order_id"] == "a"].iloc[0]
    same_timestamp_orders = result.loc[result["order_id"].isin(["b", "c"])]
    last_order = result.loc[result["order_id"] == "d"].iloc[0]

    assert first_order["seller_state_prior_order_count"] == 0
    assert pd.isna(first_order["seller_state_prior_low_review_rate"])
    assert same_timestamp_orders["seller_state_prior_order_count"].tolist() == [1, 1]
    assert same_timestamp_orders["seller_state_prior_low_review_rate"].tolist() == [1.0, 1.0]
    assert last_order["seller_state_prior_order_count"] == 3
    assert last_order["seller_state_prior_low_review_rate"] == 2 / 3


def test_enhanced_feature_columns_only_added_in_enhanced_mode():
    baseline = get_primary_feature_columns(include_historical_features=True)
    enhanced = get_primary_feature_columns(include_enhanced_historical_features=True)

    assert not set(ENHANCED_HISTORICAL_FEATURE_COLUMNS).intersection(baseline)
    assert set(ENHANCED_HISTORICAL_FEATURE_COLUMNS).issubset(enhanced)


def test_ranking_metrics_at_fraction_match_known_values():
    y_true = np.array([1, 0, 1, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.1])

    result = ranking_metrics_at_fraction(y_true, y_proba, top_frac=0.40)

    assert result["n_flagged"] == 2
    assert result["true_positives"] == 1
    assert result["precision_at_k"] == 0.5
    assert result["recall_at_k"] == 0.5
    assert result["lift_at_k"] == 1.25


def test_topk_table_and_invalid_fraction():
    table = build_topk_ranking_table([1, 0, 1], [0.9, 0.8, 0.1], top_fracs=[0.34, 1.0])

    assert table["n_flagged"].tolist() == [2, 3]
    assert set(["precision_at_k", "recall_at_k", "lift_at_k"]).issubset(table.columns)
    with pytest.raises(ValueError):
        ranking_metrics_at_fraction([1, 0], [0.5, 0.4], top_frac=0)
