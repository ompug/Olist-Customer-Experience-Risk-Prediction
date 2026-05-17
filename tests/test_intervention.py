from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cx_risk.intervention import (
    build_intervention_curve,
    compute_lift,
    evaluate_thresholds,
    evaluate_top_k_intervention,
)


def test_top_fraction_flags_expected_count_and_valid_metrics():
    y_true = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    result = evaluate_top_k_intervention(y_true, y_proba, top_frac=0.10)

    assert result["n_flagged"] == 1
    assert result["pct_orders_flagged"] == 0.1
    assert 0 <= result["precision"] <= 1
    assert 0 <= result["recall"] <= 1
    assert result["lift_over_random"] >= 0
    assert result["true_positives"] == 1


def test_top_fraction_uses_ceiling_for_selected_count():
    y_true = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_proba = np.linspace(1, 0, len(y_true))

    result = evaluate_top_k_intervention(y_true, y_proba, top_frac=0.10)

    assert result["n_flagged"] == 2


def test_lift_matches_precision_over_base_rate():
    y_true = np.array([1, 0, 1, 0])
    selected_mask = np.array([True, False, False, False])

    assert compute_lift(y_true, selected_mask) == 2.0


def test_curve_and_threshold_helpers_return_expected_rows():
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.9, 0.8, 0.2, 0.1])

    curve = build_intervention_curve(y_true, y_proba, top_fracs=[0.25, 0.50])
    thresholds = evaluate_thresholds(y_true, y_proba, thresholds=[0.2, 0.8])

    assert len(curve) == 2
    assert len(thresholds) == 2
    assert set(curve["selection_strategy"]) == {"top_fraction"}
    assert set(thresholds["selection_strategy"]) == {"probability_threshold"}


def test_intervention_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        evaluate_top_k_intervention([1, 0], [0.5], top_frac=0.10)
    with pytest.raises(ValueError):
        evaluate_top_k_intervention([1, 0], [0.5, 0.4], top_frac=1.10)
    with pytest.raises(ValueError):
        evaluate_top_k_intervention([], [], top_frac=0.10)
