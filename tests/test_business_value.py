from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cx_risk.business_value import (
    build_business_value_table,
    build_sensitivity_table,
    calculate_business_value,
    default_assumptions_table,
    validate_assumptions,
)


def _sample_intervention_table():
    return pd.DataFrame(
        {
            "model": ["Model A", "Model A"],
            "selection_strategy": ["top_fraction", "top_fraction"],
            "top_frac": [0.10, 0.20],
            "probability_threshold": [np.nan, np.nan],
            "n_orders": [1000, 1000],
            "n_flagged": [100, 200],
            "true_positives": [25, 40],
            "precision": [0.25, 0.20],
            "recall": [0.10, 0.16],
            "lift_over_random": [2.5, 2.0],
        }
    )


def test_calculate_business_value_matches_expected_formulas():
    result = calculate_business_value(
        n_flagged=100,
        true_positives=25,
        cost_per_order=2.0,
        expected_save_rate=0.10,
        value_per_saved_low_review=50.0,
    )

    assert result["expected_saved_low_reviews"] == 2.5
    assert result["intervention_cost"] == 200.0
    assert result["gross_value"] == 125.0
    assert result["net_value"] == -75.0
    assert result["roi"] == -0.375
    assert result["break_even_save_rate"] == 0.16


def test_calculate_business_value_handles_zero_true_positives_break_even():
    result = calculate_business_value(
        n_flagged=100,
        true_positives=0,
        cost_per_order=2.0,
        expected_save_rate=0.10,
        value_per_saved_low_review=50.0,
    )

    assert np.isnan(result["break_even_save_rate"])


def test_validate_assumptions_rejects_invalid_values():
    assumptions = default_assumptions_table()

    invalid_cost = assumptions.copy()
    invalid_cost.loc[0, "cost_per_order"] = -1
    with pytest.raises(ValueError, match="cost_per_order"):
        validate_assumptions(invalid_cost)

    invalid_rate = assumptions.copy()
    invalid_rate.loc[0, "expected_save_rate"] = 1.5
    with pytest.raises(ValueError, match="expected_save_rate"):
        validate_assumptions(invalid_rate)

    invalid_value = assumptions.copy()
    invalid_value.loc[0, "value_per_saved_low_review"] = 0
    with pytest.raises(ValueError, match="value_per_saved_low_review"):
        validate_assumptions(invalid_value)


def test_business_value_table_cross_joins_interventions_and_assumptions():
    intervention_table = _sample_intervention_table()
    assumptions = pd.DataFrame(
        [
            {
                "intervention_type": "A",
                "cost_per_order": 1.0,
                "expected_save_rate": 0.10,
                "value_per_saved_low_review": 10.0,
            },
            {
                "intervention_type": "B",
                "cost_per_order": 2.0,
                "expected_save_rate": 0.20,
                "value_per_saved_low_review": 20.0,
            },
        ]
    )

    result = build_business_value_table(intervention_table, assumptions)

    assert len(result) == 4
    assert set(result["intervention_type"]) == {"A", "B"}
    assert {"expected_saved_low_reviews", "net_value", "roi"}.issubset(result.columns)


def test_sensitivity_table_emits_all_multipliers():
    result = build_sensitivity_table(
        _sample_intervention_table().head(1),
        assumptions=default_assumptions_table().head(1),
        save_rate_multipliers=[0.5, 1.0, 1.5],
    )

    assert result["save_rate_multiplier"].tolist() == [0.5, 1.0, 1.5]
    assert len(result) == 3
