from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cx_risk.explainability import (
    add_reason_codes_to_queue,
    compute_permutation_importance_table,
    generate_reason_codes,
)


def test_generate_reason_codes_prioritizes_business_rules_and_caps_output():
    row = pd.Series(
        {
            "seller_prior_order_count": 10,
            "seller_prior_low_review_rate": 0.35,
            "category_prior_low_review_rate": 0.25,
            "estimated_delivery_days": 25,
            "freight_share": 0.5,
            "multi_seller_flag": 1,
            "high_installment_flag": 1,
            "expensive_order_flag": 1,
        }
    )

    reasons = generate_reason_codes(row)

    assert reasons == [
        "Seller has elevated prior low-review rate",
        "Category has elevated prior low-review rate",
        "Long estimated delivery window",
    ]


def test_generate_reason_codes_returns_fallback_when_no_rule_matches():
    reasons = generate_reason_codes(pd.Series({"estimated_delivery_days": 5}))

    assert reasons == ["No dominant rule-based reason"]


def test_add_reason_codes_to_queue_preserves_row_order():
    queue = pd.DataFrame(
        {
            "order_id": ["a", "b"],
            "queue_rank": [1, 2],
            "estimated_delivery_days": [25, 5],
            "seller_prior_order_count": [0, 0],
            "seller_prior_low_review_rate": [0.0, 0.0],
            "category_prior_low_review_rate": [0.0, 0.0],
        }
    )

    result = add_reason_codes_to_queue(queue)

    assert result["order_id"].tolist() == ["a", "b"]
    assert result.loc[0, "reason_1"] == "Long estimated delivery window"
    assert result.loc[1, "reason_1"] == "No dominant rule-based reason"
    assert "reason_summary" in result.columns


class ToyProbabilityModel:
    _estimator_type = "classifier"

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        scores = np.asarray(X["signal"], dtype=float)
        scores = np.clip(scores, 0.01, 0.99)
        return np.column_stack([1 - scores, scores])


def test_compute_permutation_importance_table_shape_and_rank():
    X = pd.DataFrame(
        {
            "signal": [0.9, 0.8, 0.2, 0.1, 0.85, 0.15],
            "noise": [0.1, 0.4, 0.3, 0.2, 0.9, 0.7],
        }
    )
    y = pd.Series([1, 1, 0, 0, 1, 0])
    model = ToyProbabilityModel().fit(X, y)

    result = compute_permutation_importance_table(model, X, y, "Toy", n_repeats=2)

    assert result["model"].unique().tolist() == ["Toy"]
    assert result["rank"].tolist() == [1, 2]
    assert set(result.columns) == {
        "model",
        "rank",
        "feature",
        "importance_mean",
        "importance_std",
        "scoring",
    }


def test_compute_permutation_importance_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        compute_permutation_importance_table(
            ToyProbabilityModel(),
            pd.DataFrame(),
            pd.Series(dtype=int),
            "Toy",
        )
