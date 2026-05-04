# Olist Customer-Experience Risk Prediction

This project predicts whether an Olist e-commerce order is at risk of receiving a low customer review using only leakage-safe information available before the customer submits a review.

The main deliverable is `olist_cx_risk.ipynb`, a report-style notebook built phase by phase from the project blueprint in `.cursor/project_blueprint.md`.

## Business Question

Can an e-commerce platform identify orders that are likely to result in poor customer experience early enough to support proactive intervention?

The target is:

- `low_review = 1` when `review_score <= 2`
- `low_review = 0` when `review_score >= 3`

The primary model intentionally excludes review text, review timestamps, actual customer delivery date, actual lateness, and carrier handoff timing.

## Project Structure

```text
ML_FINAL_PROJECT/
|-- olist_data/                  # Raw Olist CSV files
|-- outputs/
|   |-- figures/                 # EDA, evaluation, and interpretation figures
|   |-- comparison_table.csv     # Held-out model comparison metrics
|   |-- model_best.pkl           # Serialized best model pipeline
|   `-- df_features.csv          # Final processed modeling table
|-- scripts/
|   `-- run_baseline_pipeline.py # Lightweight extracted pipeline runner
|-- src/
|   `-- cx_risk/                 # Reusable data, feature, model, and evaluation code
|-- tests/                       # Lightweight pipeline validation checks
|-- olist_cx_risk.ipynb          # Main notebook/report
|-- requirements.txt             # Python dependencies
`-- README.md
```

## Portfolio Refactor

The original class-project notebook, `olist_cx_risk.ipynb`, is preserved as the submitted report artifact. A first safe refactor is now underway on branch `beyond-final-project`: stable reusable notebook logic has been extracted into `src/cx_risk/`, with a simple runnable baseline script in `scripts/run_baseline_pipeline.py` and lightweight validation checks in `tests/`.

This refactor does not introduce new model families or portfolio features. It keeps the notebook's target definition, review deduplication rule, order-level feature construction, preprocessing approach, and baseline model families intact.

## Method Summary

The notebook builds an order-level modeling table from the relational Olist dataset, defines a binary low-review target, engineers leakage-safe purchase-time features, and compares three sklearn model families:

- Logistic Regression
- Random Forest
- HistGradientBoostingClassifier

All models use a shared preprocessing pipeline with numeric imputation/scaling and categorical imputation/one-hot encoding. The final evaluation uses a held-out stratified test split and reports accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, ROC curves, and precision-recall curves.

## Main Findings

`HistGradientBoostingClassifier` was the strongest overall model by held-out ROC-AUC and class-1 F1. Logistic Regression had the highest class-1 recall by a small margin at the default threshold, but HGB provided a better overall balance for intervention use.

The strongest risk-associated signals included product category, customer state, seller state, purchase month, freight cost, item count, estimated delivery window, and average freight.

## Reproducibility

Use the course conda environment if available:

```bash
conda activate itcs-3156
```

If recreating the environment manually, use a stable Python version supported by the pinned scientific stack, then install dependencies and run the notebook top-to-bottom:

```bash
pip install -r requirements.txt
```

The project was validated in the `itcs-3156` conda environment used for the course workspace. Avoid running it with the system Python 3.14 interpreter, which may not have compatible prebuilt wheels for the pinned dependencies.

## Limitations

This is an observational project, so the model identifies associations rather than causal drivers. The target is imbalanced, and the leakage-safe design intentionally excludes some highly informative downstream information. Future work should add time-aware validation, historical seller/category risk features computed without future leakage, threshold optimization, intervention simulation, and a small scoring dashboard or packaged inference pipeline.
