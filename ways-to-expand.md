# Ways to Expand Olist Customer-Experience Risk Prediction

This project can grow from a leakage-aware ML analysis and portfolio dashboard into a fuller customer-experience risk operations product. The expansion tracks below are intended to be implemented over time.

## 1. Production Scoring and Case Queue

Turn the model into an operational workflow that scores new orders and produces a ranked queue for support or operations teams.

Possible deliverables:

- Train-and-save model artifact with version metadata.
- Batch scoring script for new order CSVs.
- Risk queue table with `order_id`, risk score, risk band, top drivers, recommended intervention, and queue rank.
- Streamlit page for reviewing, filtering, and exporting the intervention queue.
- Tests for schema validation, missing columns, and no target leakage in scoring inputs.

Why it matters:

The current dashboard explains model results, but it does not yet let a user operate the model. A risk queue turns the project into a decision product.

## 2. Explainability and Trust

Add global and per-order explanations so users can understand why orders are risky.

Possible deliverables:

- Global feature importance for tree-based models.
- Permutation importance on the time-based test set.
- Per-order reason codes such as high freight share, long estimated delivery window, risky seller history, or risky category history.
- Dashboard views for model drivers and selected-order explanations.

Why it matters:

Support and operations users need explanations before acting on a risk score. Reason codes also make the project easier to understand in a portfolio or demo setting.

## 3. Intervention Strategy Simulator

Expand from "which orders should we flag?" into "what should we do, and what is it worth?"

Possible deliverables:

- Configurable intervention types such as proactive message, seller escalation, shipping check, refund voucher, and premium support review.
- Cost, capacity, expected save rate, and customer-value assumptions.
- ROI table by queue size and intervention type.
- Scenario comparison dashboard.
- Sensitivity analysis for uncertain intervention effectiveness.

Why it matters:

The current intervention simulation evaluates prioritization quality but avoids causal claims. A business simulator can make intervention assumptions explicit and show break-even points.

## 4. Better Validation and Monitoring

Make validation closer to real deployment and add ongoing model health checks.

Possible deliverables:

- Rolling-window backtests instead of one 80/20 chronological split.
- Calibration metrics and reliability plots.
- Segment performance by state, category, seller-risk tier, payment type, and order value.
- Drift reports comparing training, validation, and future scoring windows.
- Model card documenting intended use, leakage boundaries, limitations, and fairness considerations.

Why it matters:

The current time-aware split is a strong improvement over random splitting, but operational ML needs repeated temporal validation and monitoring.

## 5. Model and Feature Lift

Test whether stronger modeling and richer leakage-safe features improve ranking quality.

Possible deliverables:

- Calibrated classifiers and tuned thresholds for ranking objectives.
- LightGBM, XGBoost, or CatBoost experiments if dependency constraints allow.
- Safe historical features for seller-state, category-state, product volume buckets, payment profile, and month-level trends.
- Strictly time-aware or cross-fit target encoding.
- Optimization for average precision, recall at fixed capacity, and lift at top-k rather than ROC-AUC alone.

Why it matters:

The current ROC-AUC is modest, but the ranking lift is useful. Future modeling should optimize the operational queue directly.

## 6. Data Product and API Layer

Wrap the project in interfaces that look like a real internal ML service.

Possible deliverables:

- FastAPI scoring endpoint.
- Pydantic request and response schemas.
- Dockerfile and local compose file.
- Lightweight SQLite or DuckDB persistence for scored orders and intervention outcomes.
- API tests and example requests.

Why it matters:

This turns the repo into a deployable ML product architecture instead of only a local analysis project.

## 7. Outcome Feedback Loop

Add a workflow for tracking intervention decisions and observed outcomes.

Possible deliverables:

- Intervention log schema.
- Fields for assigned action, owner, status, timestamp, cost, and eventual review outcome.
- Dashboard view for action completion and post-action low-review rate.
- Offline evaluation comparing contacted vs. uncontacted risk bands, with clear non-causal caveats.

Why it matters:

The product becomes stronger when it learns from how teams actually use the queue and what happens afterward.

## Suggested Implementation Order

1. Production scoring and case queue.
2. Explainability and trust.
3. Intervention strategy simulator.
4. Better validation and monitoring.
5. Model and feature lift.
6. Data product and API layer.
7. Outcome feedback loop.

This order builds from the current project structure outward: first create an operational queue, then make it explainable, then attach business value, deployment interfaces, and feedback.

