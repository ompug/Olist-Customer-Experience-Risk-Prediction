"""Raw Olist data loading."""

from pathlib import Path

import pandas as pd

from .config import DATA_DIR, DATA_FILES, DATETIME_COLUMNS


def validate_required_files(data_dir: Path = DATA_DIR) -> None:
    missing = [filename for filename in DATA_FILES.values() if not (data_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected raw data files: {missing}")


def load_raw_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load raw Olist CSV files, parse notebook timestamp columns, and translate products."""
    validate_required_files(data_dir)
    tables = {
        name: pd.read_csv(data_dir / filename)
        for name, filename in DATA_FILES.items()
    }

    for table_name, columns in DATETIME_COLUMNS.items():
        for column in columns:
            tables[table_name][column] = pd.to_datetime(tables[table_name][column], errors="coerce")

    products = tables["products_raw"].merge(
        tables["category_translation"],
        on="product_category_name",
        how="left",
        validate="many_to_one",
    )
    tables["products"] = products
    return tables

