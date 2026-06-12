# Expansion Plan

## Current State

The project is a leakage-aware portfolio ML system for predicting low-review risk on future Olist e-commerce orders. It already includes reusable feature engineering, time-aware validation, historical aggregate features, intervention simulation, tests for core leakage behavior, saved output artifacts, and a Streamlit dashboard.

The strongest current product story is not binary classification. It is risk-ranked queue building for customer-support or operations teams. The top 10% queue catches 451 future low-review orders out of 1,855 low-review cases in the test period, with 23.53% precision and 2.43x lift over random selection.

## Strategic Direction

Move the project from "ML analysis and dashboard" toward "customer-experience risk operations platform." The product should help a team decide which orders need attention, why they are risky, what action to take, and whether those actions improve outcomes.

## Expansion Tracks

### 1. Production Scoring and Case Queue

Build a scoring workflow that outputs order-level risk records for operational use.

Deliverables:

- Train-and-save model artifact with version metadata.
- Batch scoring script for new order CSVs.
- Risk queue table with `order_id`, risk score, risk band, top drivers, recommended intervention, and queue rank.
- Streamlit page for reviewing, filtering, and exporting the intervention queue.
- Tests covering schema validation, missing columns, and no target leakage in scoring inputs.

Why it matters:

The current dashboard reports model results, but does not let a user operate the model. A queue turns it into a decision product.

### 2. Explainability and Trust

Add global and row-level explanations so users can understand what drives risk.

Deliverables:

- Global feature importance for tree-based models.
- Permutation importance on the time-based test set.
- Per-order reason codes such as high freight share, long estimated delivery window, risky seller history, or risky category history.
- Dashboard views for model drivers and selected-order explanations.

Why it matters:

Support and operations users need explanations before acting on a risk score. Reason codes are also more portfolio-friendly than raw model probabilities.

### 3. Intervention Strategy Simulator

Expand from "who to flag" into "what should we do and what is it worth."

Deliverables:

- Configurable intervention types such as proactive message, seller escalation, shipping check, refund voucher, and premium support review.
- Cost, capacity, expected save rate, and customer-value assumptions.
- ROI table by queue size and intervention type.
- Scenario comparison dashboard.
- Sensitivity analysis for uncertain intervention effectiveness.

Why it matters:

The current simulation evaluates prioritization quality but avoids causal claims. A business simulator can make assumptions explicit and show break-even points.

### 4. Better Validation and Monitoring

Make validation closer to real deployment and add ongoing model health checks.

Deliverables:

- Rolling-window backtests instead of one 80/20 chronological split.
- Calibration metrics and reliability plots.
- Segment performance by state, category, seller-risk tier, payment type, and order value.
- Drift reports comparing training, validation, and future scoring windows.
- Model card documenting intended use, leakage boundaries, limitations, and fairness considerations.

Why it matters:

The current time split is a strong improvement over random splitting, but operational ML needs repeated temporal validation and monitoring.

### 5. Model and Feature Lift

Test whether more expressive modeling and richer safe features improve ranking quality.

Deliverables:

- Add calibrated classifiers and tune thresholds for ranking objectives.
- Try LightGBM, XGBoost, or CatBoost if dependency constraints allow.
- Add safe historical features for seller-state, category-state, product volume buckets, payment profile, and month-level trends.
- Add target encoding only with strict time-aware or cross-fit safeguards.
- Optimize for average precision, recall at fixed capacity, and lift at top-k rather than ROC-AUC alone.

Why it matters:

ROC-AUC is modest at about 0.61, but the ranking lift is useful. Future modeling should optimize the operational queue directly.

### 6. Data Product and API Layer

Wrap the project in interfaces that look like a real internal service.

Deliverables:

- FastAPI scoring endpoint.
- Pydantic request and response schemas.
- Dockerfile and local compose file.
- Lightweight SQLite or DuckDB persistence for scored orders and intervention outcomes.
- API tests and example requests.

Why it matters:

This turns the portfolio repo into a deployable ML product architecture.

### 7. Outcome Feedback Loop

Add a workflow for tracking intervention decisions and observed outcomes.

Deliverables:

- Intervention log schema.
- Fields for assigned action, owner, status, timestamp, cost, and eventual review outcome.
- Dashboard view for action completion and post-action low-review rate.
- Offline evaluation comparing contacted vs. uncontacted risk bands, with clear non-causal caveats.

Why it matters:

The product becomes stronger when it learns from what the team actually does with the queue.

## Suggested Roadmap

### Phase 1: Turn Analysis Into an Operational Queue

Duration: 1 to 2 weeks.

Tasks:

- Add model persistence.
- Add batch scoring for new order files.
- Create queue export artifact.
- Add queue review page to Streamlit.
- Add scoring-input schema checks and tests.

Success criteria:

- A user can generate a ranked queue without retraining from the dashboard.
- The queue includes order IDs, risk scores, risk bands, and reason codes.

### Phase 2: Explain and Tune the Queue

Duration: 1 to 2 weeks.

Tasks:

- Add feature importance and reason-code generation.
- Add queue-size controls in the dashboard.
- Add segment performance reporting.
- Add probability calibration analysis.

Success criteria:

- A user can explain why an order was flagged.
- The project reports performance at business-relevant capacities such as top 1%, 5%, 10%, and 20%.

### Phase 3: Add Business Value Simulation

Duration: 1 week.

Tasks:

- Define intervention assumptions.
- Build ROI and break-even calculations.
- Add scenario comparison UI.
- Document non-causal limits.

Success criteria:

- A user can compare intervention policies by cost, expected saved low reviews, and queue capacity.

### Phase 4: Make It Deployable

Duration: 2 weeks.

Tasks:

- Add FastAPI scoring service.
- Add Dockerfile.
- Add persisted scoring and intervention tables.
- Add integration tests.
- Add model card and deployment docs.

Success criteria:

- The project can run as a local service with reproducible scoring and documented model behavior.

### Phase 5: Improve Model Quality

Duration: ongoing.

Tasks:

- Add rolling backtests.
- Tune models for average precision and top-k lift.
- Evaluate new safe historical features.
- Compare stronger gradient boosting libraries where practical.

Success criteria:

- Top-10% lift and precision improve without weakening leakage controls.
- Performance is stable across time windows and key segments.

## Highest-Impact First Build

The best first expansion is the production scoring and queue workflow. It is tightly aligned with the existing code, requires limited new dependencies, and makes the project feel like a complete product rather than a completed analysis.

Minimum scope:

- `scripts/train_model.py`
- `scripts/score_orders.py`
- `outputs/models/`
- `outputs/tables/risk_queue.csv`
- `src/cx_risk/scoring.py`
- Streamlit "Risk Queue" page
- Tests for score output shape, required columns, and leakage-safe scoring inputs

