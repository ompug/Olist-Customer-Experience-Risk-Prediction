from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cx_risk.scoring import (
    assign_risk_band,
    build_risk_queue,
    model_slug,
    recommend_intervention,
    validate_scoring_features,
)


def test_model_slug_is_stable_for_baseline_names():
    assert model_slug("Logistic Regression") == "logistic_regression"
    assert model_slug("Random Forest") == "random_forest"
    assert model_slug("HistGradientBoostingClassifier") == "hist_gradient_boosting_classifier"


def test_assign_risk_band_boundaries():
    bands = assign_risk_band([0.10, 0.25, 0.50, 0.75])

    assert bands.tolist() == ["low", "medium", "high", "critical"]


def test_recommend_intervention_uses_safe_order_attributes():
    shipping_row = pd.Series(
        {
            "risk_band": "high",
            "estimated_delivery_days": 25,
            "seller_prior_low_review_rate": 0.05,
            "seller_prior_order_count": 20,
        }
    )
    seller_row = pd.Series(
        {
            "risk_band": "critical",
            "estimated_delivery_days": 10,
            "seller_prior_low_review_rate": 0.30,
            "seller_prior_order_count": 5,
        }
    )
    medium_row = pd.Series({"risk_band": "medium"})
    low_row = pd.Series({"risk_band": "low"})

    assert recommend_intervention(shipping_row) == "Shipping check"
    assert recommend_intervention(seller_row) == "Seller escalation"
    assert recommend_intervention(medium_row) == "Monitor"
    assert recommend_intervention(low_row) == "No action"


def test_validate_scoring_features_rejects_missing_unexpected_and_forbidden_columns():
    feature_columns = ["purchase_month", "review_score"]
    X = pd.DataFrame({"purchase_month": [1], "review_score": [5]})

    with pytest.raises(ValueError, match="Forbidden"):
        validate_scoring_features(X, feature_columns)

    with pytest.raises(ValueError, match="missing"):
        validate_scoring_features(pd.DataFrame({"purchase_month": [1]}), ["purchase_month", "n_items"])

    with pytest.raises(ValueError, match="unexpected"):
        validate_scoring_features(
            pd.DataFrame({"purchase_month": [1], "n_items": [2], "extra": [3]}),
            ["purchase_month", "n_items"],
        )


def test_build_risk_queue_ranks_highest_risk_first_and_keeps_required_columns():
    X_scoring = pd.DataFrame(
        {
            "customer_state": ["SP", "RJ", "MG"],
            "primary_category": ["books", "toys", "auto"],
            "estimated_delivery_days": [8, 22, 10],
            "freight_share": [0.1, 0.4, 0.2],
            "multi_seller_flag": [0, 0, 1],
            "high_installment_flag": [0, 1, 0],
            "expensive_order_flag": [0, 1, 1],
            "seller_prior_order_count": [1, 10, 6],
            "seller_prior_low_review_rate": [0.0, 0.05, 0.25],
            "category_prior_low_review_rate": [0.1, 0.2, 0.3],
        }
    )
    queue = build_risk_queue(
        "Test Model",
        order_ids=pd.Series(["a", "b", "c"]),
        X_scoring=X_scoring,
        risk_scores=np.array([0.2, 0.8, 0.6]),
        y_true=pd.Series([0, 1, 1]),
    )

    assert queue["order_id"].tolist() == ["b", "c", "a"]
    assert queue["queue_rank"].tolist() == [1, 2, 3]
    assert queue.loc[0, "risk_band"] == "critical"
    assert queue.loc[0, "recommended_action"] == "Shipping check"
    assert queue.loc[1, "recommended_action"] == "Seller escalation"
    assert queue["actual_low_review"].tolist() == [1, 1, 0]
