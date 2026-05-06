# Architecture

This project keeps the original notebook as a report artifact while moving reusable ML logic into `src/cx_risk/`.

## Pipeline Flow

1. `data.py` validates and loads raw Olist CSV files from `olist_data/`.
2. `features.py` builds one order-level modeling row from relational order, item, payment, product, seller, customer, and review tables.
3. `preprocessing.py` defines the leakage-safe feature list, leakage guard, type detection, and sklearn preprocessing transformer.
4. `models.py` creates the same three model pipelines used in the notebook baseline.
5. `evaluation.py` computes classification metrics and comparison tables.
6. `validation.py` creates chronological train/test splits for future-order validation.
7. `historical.py` computes seller, category, and customer-state historical risk features using only strictly prior orders.
8. `intervention.py` evaluates top-k and probability-threshold intervention policies.
9. `app/streamlit_app.py` reads saved artifacts and displays results without retraining.

## Leakage Guard

The primary feature list excludes review-derived fields, raw identifiers, actual-delivery-derived fields, and target-source columns. Historical features use cumulative aggregates from earlier timestamp buckets only, so the current order and future orders are not included.

## Artifact Flow

- `scripts/run_time_validation.py` writes `outputs/tables/time_validation_metrics.csv`.
- `scripts/run_time_validation_with_history.py` writes `outputs/tables/time_validation_with_history_metrics.csv`.
- `scripts/run_intervention_simulation.py` writes intervention tables and figures.
- The Streamlit app reads those artifacts for presentation.
