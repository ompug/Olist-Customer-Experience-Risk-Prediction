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
9. `scoring.py` saves/loads fitted model artifacts and builds ranked operational risk queues.
10. `explainability.py` computes permutation importance and rule-based queue reason codes.
11. `business_value.py` estimates scenario value, ROI, and break-even rates for intervention strategies.
12. `monitoring.py` creates rolling validation windows, calibration summaries, segment diagnostics, and drift reports.
13. `model_lift.py` computes top-k ranking metrics for model and feature experiments.
14. `cx_risk_api/` exposes saved model artifacts through a FastAPI scoring service.
15. `app/streamlit_app.py` reads saved artifacts, can run scoring scripts, and displays results.

## Leakage Guard

The primary feature list excludes review-derived fields, raw identifiers, actual-delivery-derived fields, and target-source columns. Historical features use cumulative aggregates from earlier timestamp buckets only, so the current order and future orders are not included.

## Artifact Flow

- `scripts/run_time_validation.py` writes `outputs/tables/time_validation_metrics.csv`.
- `scripts/run_time_validation_with_history.py` writes `outputs/tables/time_validation_with_history_metrics.csv`.
- `scripts/run_intervention_simulation.py` writes intervention tables and figures.
- `scripts/run_business_value_simulation.py` writes business-value and sensitivity scenario tables.
- `scripts/run_monitoring.py` writes rolling validation, calibration, segment, and drift tables.
- `scripts/run_model_lift_experiments.py` writes model-lift experiment and top-k ranking tables.
- `scripts/train_scoring_models.py` writes fitted model artifacts and `outputs/models/model_registry.json`.
- `scripts/score_holdout_queue.py` writes `outputs/tables/risk_queue.csv` and global importance tables.
- The FastAPI service reads `outputs/models/model_registry.json`, loads model artifacts, and writes `outputs/scoring_api.db`.
- The Streamlit app reads those artifacts for presentation.

## Risk Queue Workflow

The production-style queue uses a future holdout simulation. Models train on earlier reviewed orders, then score later orders as if they were unlabeled candidates. Holdout labels are included only after scoring for evaluation columns in the generated queue.

The queue includes all baseline model families, a default HistGradientBoostingClassifier view, risk bands, deterministic intervention recommendations, safe order context, and evaluation-only target columns.

## Explainability

Global model drivers use permutation importance on the same future holdout split used for queue scoring. Per-order reasons are deterministic business rules from leakage-safe fields such as seller history, category history, freight share, delivery estimate, and basket/payment flags. These reason codes are intended for operational trust and triage, not causal attribution.

## Business Value Simulation

Business-value scenarios combine intervention queue performance with explicit cost, expected save-rate, and value assumptions. The simulator reports expected saved low reviews, intervention costs, gross value, net value, ROI, and break-even save rates. These values are scenario-planning estimates and should not be interpreted as measured causal impact.

## Monitoring

Monitoring artifacts use expanding-window temporal backtests with leakage-safe historical features. The workflow reports metric stability, calibration quality, segment-level performance, and train-vs-future feature drift. The model card in `docs/model_card.md` documents intended use, validation design, leakage boundaries, and limitations.

## Model Lift Experiments

Model-lift experiments compare the current historical feature set against enhanced strictly-prior aggregates for seller-state, category-state, payment profile, product-volume tier, and purchase month. The workflow also evaluates calibrated variants of the baseline scikit-learn models and reports ranking-first metrics such as lift, precision, and recall at top queue fractions.

## API Layer

The FastAPI service is a local data-product interface over saved scoring artifacts. It accepts engineered feature payloads matching the model registry feature list, validates leakage-safe columns, scores with a selected saved model, returns risk bands and reason codes, and persists score responses to SQLite. Raw Olist relational feature building remains a batch workflow.
