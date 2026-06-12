# Model Card

## Intended Use

This project predicts order-level risk of a low customer review for Olist e-commerce orders. The intended use is prioritizing proactive customer-experience review queues for support or operations teams.

## Target

`low_review = 1` when `review_score <= 2`; otherwise `low_review = 0` for review scores 3-5.

## Model Families

The project evaluates the baseline model families used throughout the repository:

- Logistic Regression
- Random Forest
- HistGradientBoostingClassifier

The production-style queue defaults to HistGradientBoostingClassifier because it produced the strongest top-10% intervention lift in the saved artifacts.

## Leakage Boundaries

The model excludes review text, review timestamps, review score, raw high-cardinality identifiers as predictors, actual delivery fields, lateness fields, and any feature derived from future outcomes.

Historical aggregate features are computed only from strictly prior order timestamps.

## Validation

The repository includes:

- a single chronological train/test split,
- rolling expanding-window backtests,
- calibration metrics and calibration-bin tables,
- segment performance diagnostics,
- feature drift reports comparing train and future test windows.

These checks are designed to approximate deployment monitoring in a local portfolio project.

## Limitations

The model is trained on historical Olist marketplace data and should not be assumed to generalize to other marketplaces without validation. Some important operational signals, such as live shipping events, customer-service contacts, and seller interventions, are not present in the raw dataset.

Intervention and business-value simulations are prioritization and scenario tools. They do not prove that contacting a customer or seller will causally improve the eventual review.

Segment diagnostics are monitoring aids, not a legal or formal fairness audit.

## Monitoring Expectations

Before using a scored queue operationally, regenerate monitoring artifacts and inspect:

- rolling ROC-AUC and average precision stability,
- calibration drift,
- low-performing customer/category/seller-risk segments,
- large feature distribution shifts.

If performance or drift materially changes, retrain and revalidate before relying on queue recommendations.
