from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cx_risk.config import DATA_DIR, DATA_FILES, TARGET_COLUMN
from cx_risk.data import load_raw_data, validate_required_files
from cx_risk.features import build_modeling_dataframe
from cx_risk.preprocessing import FORBIDDEN_LEAKAGE_COLUMNS, PRIMARY_FEATURE_COLUMNS


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
    assert set(modeling_df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_primary_features_exclude_forbidden_leakage_columns():
    assert not set(FORBIDDEN_LEAKAGE_COLUMNS).intersection(PRIMARY_FEATURE_COLUMNS)
