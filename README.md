# Olist Customer-Experience Risk Prediction

This project predicts whether an Olist e-commerce order is at risk of receiving a low customer review using leakage-safe information available before the customer submits a review.

The original class-project deliverable is preserved as `olist_cx_risk.ipynb`. Portfolio-oriented reusable code is being developed on branch `beyond-final-project` under `src/cx_risk/`.

## Business Problem

Can an e-commerce platform identify orders that are likely to result in poor customer experience early enough to support proactive intervention?

The target is:

- `low_review = 1` when `review_score <= 2`
- `low_review = 0` when `review_score >= 3`

## Data

Raw Olist CSV files are expected in `olist_data/` at the project root.

Required files:

- `olist_customers_dataset.csv`
- `olist_geolocation_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `product_category_name_translation.csv`

The extracted pipeline validates these files before loading data.

## Project Structure

```text
ML_FINAL_PROJECT/
|-- olist_data/                  # Raw Olist CSV files
|-- outputs/
|   |-- figures/                 # Notebook EDA/evaluation figures
|   |-- comparison_table.csv     # Notebook model comparison artifact
|   |-- model_best.pkl           # Notebook serialized best model artifact
|   `-- df_features.csv          # Notebook processed modeling table artifact
|-- scripts/
|   `-- run_baseline_pipeline.py # Reusable baseline pipeline runner
|-- src/
|   `-- cx_risk/
|       |-- config.py            # Paths, constants, target settings
|       |-- data.py              # Raw data loading and timestamp parsing
|       |-- features.py          # Order-level feature table construction
|       |-- preprocessing.py     # Feature list, leakage guard, sklearn preprocessing
|       |-- models.py            # Baseline model pipeline factories
|       |-- evaluation.py        # Classification metric helpers
|       `-- utils.py             # Validation and artifact helpers
|-- tests/
|   `-- test_data_pipeline.py    # Lightweight pipeline invariant tests
|-- olist_cx_risk.ipynb          # Original notebook/report artifact
|-- requirements.txt             # Python dependencies
|-- Makefile                     # Convenience commands
`-- README.md
```

## Setup

Use the course conda environment if available:

```bash
conda activate itcs-3156
pip install -r requirements.txt
```

Or create a fresh environment with a stable Python version supported by the pinned scientific stack:

```bash
conda create -n olist-cx-risk python=3.9
conda activate olist-cx-risk
pip install -r requirements.txt
```

Avoid running the project with the system Python 3.14 interpreter unless all pinned dependencies are installed and compatible.

## Run The Baseline Pipeline

From the project root:

```bash
python scripts/run_baseline_pipeline.py
```

Equivalent Makefile command:

```bash
make baseline
```

The script loads raw data, builds the order-level modeling table, applies the leakage guard, trains the same three baseline model families used in the notebook, and prints a compact metrics table:

- Logistic Regression
- Random Forest
- HistGradientBoostingClassifier

## Run Tests

From the project root:

```bash
python -m pytest -q
```

Equivalent Makefile command:

```bash
make test
```

The tests intentionally stay lightweight and check only critical pipeline invariants:

- required raw files exist
- loaded tables are non-empty
- modeling table has unique `order_id`
- `low_review` exists and has both classes
- the primary feature list excludes forbidden leakage columns

## Leakage-Safe Primary Modeling

The primary model excludes review-derived fields, target-source fields, raw identifiers, and actual-delivery-derived fields from the feature set. This includes review text, review timestamps, review scores, review diagnostics, actual customer delivery date, carrier handoff date, and lateness-style fields.

The production leakage guard lives in `src/cx_risk/preprocessing.py` and is called before model training in `scripts/run_baseline_pipeline.py`.

## Notebook Artifact

`olist_cx_risk.ipynb` remains the original class-project report. The notebook includes EDA, narrative validation tables, plots, final model artifacts, and interpretation/error-analysis sections that have not all been migrated into reusable modules yet.

## Current Scope

This refactor does not introduce new model families, new features, dashboard code, or inference APIs. It stabilizes the existing notebook logic into reusable Python modules so future portfolio work can proceed safely.

## Limitations

This is an observational project, so the model identifies associations rather than causal drivers. The target is imbalanced, and the leakage-safe design intentionally excludes some highly informative downstream information. Future work should add time-aware validation, historical seller/category risk features computed without future leakage, threshold optimization, intervention simulation, and a small scoring dashboard or packaged inference pipeline.
