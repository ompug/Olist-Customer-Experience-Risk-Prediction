# Olist Customer-Experience Risk Prediction

Leakage-aware machine learning pipeline and Streamlit dashboard for predicting low-review risk on future e-commerce orders.

## One-Sentence Summary

This project turns a notebook-only Olist ML final project into a reusable Python codebase that builds an order-level customer-experience risk model, validates it on future orders, and simulates business intervention queues for support teams.

## Business Problem

E-commerce platforms need to identify orders likely to produce poor customer experiences before a customer submits a review. The business question is:

Can we prioritize proactive interventions for orders most likely to receive a low review?

Target definition:

- `low_review = 1` when `review_score <= 2`
- `low_review = 0` when `review_score >= 3`

## Key Result

Using a time-aware future-order test set with leakage-safe historical features, the best ROC-AUC model was Random Forest at `0.6146`. For intervention planning, the best top-10% queue came from `HistGradientBoostingClassifier`:

- Orders flagged: `1,917`
- Future low-review orders caught: `451`
- Precision: `23.53%`
- Recall: `24.31%`
- Lift over random selection: `2.43x`
- Base future low-review rate: `9.68%`

This means the model-created queue is substantially more concentrated with low-review cases than random selection, while still catching about one quarter of all future low-review orders in the test period.

## Dashboard Preview

Run the local Streamlit dashboard:

```bash
make app
```

or:

```bash
streamlit run app/streamlit_app.py
```

The dashboard shows:

- Project overview and leakage-safe design
- Time-aware validation results
- Historical-feature comparison
- Intervention simulation curves
- Threshold tradeoff analysis
- Production-style risk queue workflow
- Explainability with model drivers and per-order reason codes
- Business-value scenario simulator for intervention strategies
- Monitoring views for rolling validation, calibration, segments, and drift
- Model-lift experiments for enhanced features, calibration, and top-k ranking metrics
- FastAPI scoring service with SQLite score persistence and container support
- Reproducibility commands

Screenshots are not fabricated in this repo. To add them:

1. Run `make app`.
2. Capture the dashboard pages.
3. Save screenshots in `docs/screenshots/`.
4. Reference the screenshot paths here once added.

Placeholder directory:

```text
docs/screenshots/
```

## Technical Highlights

- Refactored notebook logic into a reusable package under `src/cx_risk/`.
- Aggregated a relational e-commerce dataset across orders, payments, items, products, sellers, customers, and reviews.
- Preserved the original random-split baseline while adding more realistic future-order validation.
- Built leakage-safe historical seller, category, and customer-state risk features using only strictly prior orders.
- Added threshold and top-k intervention simulation to translate probabilities into business actions.
- Added train/save/score workflow for a production-style holdout risk queue.
- Added permutation-importance model drivers and rule-based reason codes for queued orders.
- Added business-value scenario modeling for intervention costs, expected saves, ROI, and break-even rates.
- Added rolling monitoring artifacts for validation stability, calibration, segment performance, and drift.
- Added model-lift experiments for enhanced leakage-safe features, calibrated variants, and ranking-first metrics.
- Added FastAPI scoring endpoints, local SQLite score persistence, and Docker deployment files.
- Shipped a lightweight Streamlit dashboard that reads artifacts without retraining models at startup.

## Dataset

This project uses the public Brazilian Olist e-commerce dataset. Raw CSV files are expected in `olist_data/` at the project root.

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

Raw data files are intentionally ignored by git.

## Methodology

1. Load raw Olist CSVs and parse timestamps.
2. Deduplicate review rows at order level using the minimum review score.
3. Define `low_review` from review score.
4. Filter to delivered orders with review targets.
5. Aggregate order-item, payment, product, seller, and customer signals to one row per order.
6. Build leakage-safe primary features.
7. Train the same model families as the notebook baseline:
   - Logistic Regression
   - Random Forest
   - HistGradientBoostingClassifier
8. Evaluate random split, time-aware split, time-aware split with historical features, and intervention queues.

## Results

Time-aware validation is intentionally harder than a random split because train orders occur earlier and test orders occur later.

### Time Split Without Historical Features

| Model | ROC-AUC |
|---|---:|
| Logistic Regression | 0.5999 |
| Random Forest | 0.5989 |
| HistGradientBoostingClassifier | 0.5999 |

### Time Split With Historical Features

| Model | ROC-AUC |
|---|---:|
| Logistic Regression | 0.6087 |
| Random Forest | 0.6146 |
| HistGradientBoostingClassifier | 0.6075 |

### Intervention Simulation

At the top 10% of future orders ranked by predicted risk:

| Metric | Value |
|---|---:|
| Best model | HistGradientBoostingClassifier |
| Orders flagged | 1,917 |
| Low-review orders caught | 451 |
| Precision | 23.53% |
| Recall | 24.31% |
| Lift over random | 2.43x |
| Base low-review rate | 9.68% |

These metrics should be interpreted as prioritization quality, not a claim of causal impact. The simulation shows which orders would be queued for intervention, not whether an intervention would change outcomes.

## Project Structure

```text
ML_FINAL_PROJECT/
|-- app/
|   `-- streamlit_app.py
|-- docs/
|   |-- architecture.md
|   |-- resume_bullets.md
|   `-- screenshots/
|-- olist_data/                  # Raw Olist CSV files, gitignored
|-- outputs/
|   |-- figures/                 # Saved notebook and portfolio figures
|   |-- models/                  # Generated model artifacts, gitignored
|   `-- tables/                  # Validation and intervention CSV artifacts
|-- scripts/
|   |-- run_baseline_pipeline.py
|   |-- run_time_validation.py
|   |-- run_time_validation_with_history.py
|   |-- run_intervention_simulation.py
|   |-- train_scoring_models.py
|   `-- score_holdout_queue.py
|-- src/
|   `-- cx_risk/                 # Reusable ML pipeline package
|-- tests/
|-- olist_cx_risk.ipynb          # Original class-project notebook artifact
|-- requirements.txt
|-- Makefile
`-- README.md
```

## How To Run

Set up an environment:

```bash
conda create -n olist-cx-risk python=3.9
conda activate olist-cx-risk
pip install -r requirements.txt
```

If using the course environment:

```bash
conda activate itcs-3156
pip install -r requirements.txt
```

Run the main commands:

```bash
make baseline
make time-validation
make time-validation-history
make intervention
make business-value
make monitoring
make model-lift
make risk-queue
make api
make app
make test
```

Equivalent direct commands:

```bash
python scripts/run_baseline_pipeline.py
python scripts/run_time_validation.py
python scripts/run_time_validation_with_history.py
python scripts/run_intervention_simulation.py
python scripts/run_business_value_simulation.py
python scripts/run_monitoring.py
python scripts/run_model_lift_experiments.py
python scripts/train_scoring_models.py
python scripts/score_holdout_queue.py
uvicorn cx_risk_api.main:app --app-dir src --reload --host 0.0.0.0 --port 8000
streamlit run app/streamlit_app.py
python -m pytest -q
```

## Reproducibility Notes

- The project was validated in `itcs-3156` with Python 3.9.
- Avoid the system Python 3.14 interpreter unless compatible wheels are installed.
- The Streamlit app does not retrain models; it reads generated artifacts.
- Generate dashboard artifacts with:

```bash
make time-validation
make time-validation-history
make intervention
make business-value
make monitoring
make model-lift
make risk-queue
```

Key artifacts:

```text
outputs/models/model_registry.json
outputs/models/*.joblib
outputs/tables/time_validation_metrics.csv
outputs/tables/time_validation_with_history_metrics.csv
outputs/tables/intervention_simulation.csv
outputs/tables/threshold_analysis.csv
outputs/tables/risk_queue.csv
outputs/tables/global_feature_importance.csv
outputs/tables/global_feature_importance_top20.csv
outputs/tables/business_value_simulation.csv
outputs/tables/business_value_sensitivity.csv
outputs/tables/rolling_backtest_metrics.csv
outputs/tables/calibration_metrics.csv
outputs/tables/calibration_bins.csv
outputs/tables/segment_performance.csv
outputs/tables/feature_drift_report.csv
outputs/tables/model_lift_experiments.csv
outputs/tables/model_lift_topk.csv
outputs/tables/model_lift_feature_sets.csv
outputs/figures/intervention_recall_by_flagged_share.png
outputs/figures/intervention_precision_by_flagged_share.png
outputs/figures/intervention_lift_by_flagged_share.png
```

## Leakage Prevention Notes

The primary model excludes:

- Review text and review timestamps
- Current-order review score and review-derived diagnostics
- Raw high-cardinality identifiers as predictors
- Actual delivery and lateness fields unavailable before review submission
- Future outcomes in historical aggregate features

Historical risk features are computed from strictly earlier `order_purchase_timestamp` buckets. Orders at the same timestamp are excluded from one another's historical rates.

## Risk Queue Workflow

Generate production-style scoring artifacts with:

```bash
make risk-queue
```

This trains all three baseline model families on earlier orders with leakage-safe historical features, saves fitted pipelines under `outputs/models/`, scores the future holdout window, and writes `outputs/tables/risk_queue.csv`.

The holdout labels are included only as evaluation columns after scoring. They are not used as model inputs.

The same command also writes model-driver artifacts under `outputs/tables/global_feature_importance*.csv`. These use permutation importance on the future holdout set. Per-order queue reasons are rule-based operational explanations from leakage-safe order attributes; they are not SHAP values or causal explanations.

## Business Value Workflow

Generate default business-value scenarios after intervention artifacts exist:

```bash
make intervention
make business-value
```

This writes `outputs/tables/business_value_simulation.csv` and `outputs/tables/business_value_sensitivity.csv`. The scenario simulator estimates expected saved low reviews, intervention costs, gross value, net value, ROI, and break-even save rates from explicit assumptions. These are planning scenarios, not measured causal intervention effects.

## Monitoring Workflow

Generate rolling validation and monitoring artifacts with:

```bash
make monitoring
```

This runs expanding-window temporal backtests with leakage-safe historical features and writes rolling metrics, calibration summaries, calibration bins, segment performance, and feature drift reports under `outputs/tables/`.

The model card in `docs/model_card.md` summarizes intended use, leakage boundaries, validation design, and limitations.

## Model Lift Workflow

Run ranking-focused model and feature experiments with:

```bash
make model-lift
```

This compares the current historical feature set with enhanced leakage-safe historical features, evaluates calibrated model variants, and writes top-k ranking metrics under `outputs/tables/model_lift_*.csv`.

These are candidate experiments. They should not replace the production queue default unless they improve the relevant ranking metrics, especially lift and precision at the intended queue size.

## API Workflow

Generate model artifacts, then start the local scoring API:

```bash
make risk-queue
make api
```

The API exposes health, model metadata, single scoring, batch scoring, and recent score history. It scores engineered feature payloads that match `outputs/models/model_registry.json`; raw Olist relational records should still be processed through the existing batch pipelines.

Example requests are documented in `docs/api_examples.md`. The API persists score records to `outputs/scoring_api.db`.

## Portfolio Extensions Completed

- Notebook-to-package refactor into `src/cx_risk/`
- Reproducible baseline runner and tests
- Time-aware future-order validation
- Leakage-safe historical seller/category/customer-state risk features
- Intervention simulation for top-k and probability-threshold policies
- Production-style model artifacts and holdout risk queue
- Explainability artifacts with permutation importance and per-order reason codes
- Business-value scenario simulator for intervention strategies
- Monitoring artifacts and model card documentation
- Model-lift experiments for ranking-quality improvement
- FastAPI scoring service and SQLite score persistence
- Streamlit dashboard for portfolio presentation
- Architecture and resume documentation under `docs/`

## Future Work

- Add calibration diagnostics for predicted probabilities.
- Add cost-sensitive intervention assumptions and ROI scenarios.
- Add time-windowed historical aggregates for stronger temporal realism.
- Package the pipeline with a CLI or lightweight batch scoring entry point.
- Add dashboard screenshots to `docs/screenshots/`.
