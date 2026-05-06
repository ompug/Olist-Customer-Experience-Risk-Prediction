from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cx_risk.config import DATA_DIR, DATA_FILES, TARGET_COLUMN
from cx_risk.data import load_raw_data, validate_required_files
from cx_risk.features import build_modeling_dataframe
from cx_risk.historical import add_historical_risk_features
from cx_risk.preprocessing import (
    HISTORICAL_FEATURE_COLUMNS,
    PRIMARY_FEATURE_COLUMNS,
    assert_no_forbidden_columns,
    get_primary_feature_columns,
    validate_primary_feature_list,
)
from cx_risk.validation import temporal_train_test_split

import pandas as pd


def test_required_files_exist():
    validate_required_files(DATA_DIR)
    assert all((DATA_DIR / filename).exists() for filename in DATA_FILES.values())


def test_loaded_tables_are_non_empty():
    tables = load_raw_data(DATA_DIR)
    for name in DATA_FILES:
        assert not tables[name].empty, name


def test_modeling_table_target_and_unique_order_id():
    tables = load_raw_data(DATA_DIR)
    modeling_df = build_modeling_dataframe(tables, include_order_id=True)
    assert modeling_df["order_id"].is_unique
    assert TARGET_COLUMN in modeling_df.columns
    assert set(modeling_df[TARGET_COLUMN].unique()) == {0, 1}


def test_primary_features_exclude_forbidden_leakage_columns():
    validate_primary_feature_list()
    assert_no_forbidden_columns(PRIMARY_FEATURE_COLUMNS)


def test_temporal_split_preserves_chronology_and_target_classes():
    tables = load_raw_data(DATA_DIR)
    modeling_df = build_modeling_dataframe(tables, include_split_timestamp=True)
    X_train, X_test, y_train, y_test, summary = temporal_train_test_split(
        modeling_df,
        return_summary=True,
    )

    assert not X_train.empty
    assert not X_test.empty
    assert summary.train_end < summary.test_start
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
    assert_no_forbidden_columns(X_train.columns)
    assert_no_forbidden_columns(X_test.columns)


def test_historical_features_exclude_current_and_future_targets():
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
            "primary_seller_id": ["seller_1", "seller_1", "seller_1", "seller_1"],
            "primary_category": ["cat_1", "cat_1", "cat_1", "cat_1"],
            "customer_state": ["SP", "SP", "SP", "SP"],
            TARGET_COLUMN: [1, 0, 1, 0],
        }
    )

    result = add_historical_risk_features(frame)
    first_order = result.loc[result["order_id"] == "a"].iloc[0]
    same_timestamp_orders = result.loc[result["order_id"].isin(["b", "c"])]
    last_order = result.loc[result["order_id"] == "d"].iloc[0]

    assert first_order["seller_prior_order_count"] == 0
    assert pd.isna(first_order["seller_prior_low_review_rate"])
    assert same_timestamp_orders["seller_prior_order_count"].tolist() == [1, 1]
    assert same_timestamp_orders["seller_prior_low_review_rate"].tolist() == [1.0, 1.0]
    assert last_order["seller_prior_order_count"] == 3
    assert last_order["seller_prior_low_review_rate"] == 2 / 3


def test_temporal_split_with_history_uses_safe_predictors():
    tables = load_raw_data(DATA_DIR)
    modeling_df = build_modeling_dataframe(
        tables,
        include_split_timestamp=True,
        include_historical_features=True,
    )
    feature_columns = get_primary_feature_columns(include_historical_features=True)
    X_train, X_test, y_train, y_test, summary = temporal_train_test_split(
        modeling_df,
        feature_columns=feature_columns,
        return_summary=True,
    )

    assert not X_train.empty
    assert not X_test.empty
    assert summary.train_end < summary.test_start
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
    assert set(HISTORICAL_FEATURE_COLUMNS).issubset(X_train.columns)
    assert "primary_seller_id" not in X_train.columns
    assert_no_forbidden_columns(X_train.columns)
    assert_no_forbidden_columns(X_test.columns)
