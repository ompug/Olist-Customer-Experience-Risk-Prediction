"""Run time-aware validation with leakage-safe historical risk features."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cx_risk.config import TABLES_DIR
from cx_risk.data import load_raw_data
from cx_risk.evaluation import create_model_comparison_table
from cx_risk.features import build_modeling_dataframe
from cx_risk.models import create_model_pipelines, fit_baseline_models, predict_model_outputs
from cx_risk.preprocessing import build_preprocessor, get_primary_feature_columns, identify_feature_types
from cx_risk.utils import save_table
from cx_risk.validation import temporal_train_test_split


def _print_metric_comparison(history_metrics: pd.DataFrame) -> None:
    baseline_path = TABLES_DIR / "time_validation_metrics.csv"
    if not baseline_path.exists():
        print("\nNo existing time-validation metrics found for comparison.")
        return

    baseline_metrics = pd.read_csv(baseline_path)
    metric_columns = ["roc_auc", "pr_auc_average_precision", "f1_class_1", "recall_class_1"]
    comparison = baseline_metrics[["model"] + metric_columns].merge(
        history_metrics[["model"] + metric_columns],
        on="model",
        suffixes=("_time", "_time_history"),
        validate="one_to_one",
    )
    for metric in metric_columns:
        comparison[f"{metric}_delta"] = comparison[f"{metric}_time_history"] - comparison[f"{metric}_time"]

    print("\nDelta vs time-validation without historical features:")
    print(comparison.round(4).to_string(index=False))


def main() -> None:
    tables = load_raw_data()
    modeling_df = build_modeling_dataframe(
        tables,
        include_split_timestamp=True,
        include_historical_features=True,
    )
    feature_columns = get_primary_feature_columns(include_historical_features=True)
    X_train, X_test, y_train, y_test, split_summary = temporal_train_test_split(
        modeling_df,
        feature_columns=feature_columns,
        return_summary=True,
    )
    numeric_features, categorical_features = identify_feature_types(X_train)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    pipelines = create_model_pipelines(preprocessor)
    fitted_models, warnings_by_model = fit_baseline_models(pipelines, X_train, y_train)
    outputs = predict_model_outputs(fitted_models, X_test)
    comparison = create_model_comparison_table(y_test, outputs)
    comparison.insert(0, "validation_strategy", "time_aware_with_history")

    metrics_path = save_table(comparison, TABLES_DIR / "time_validation_with_history_metrics.csv")

    print("Time-aware split with historical features:")
    print(f"- Train rows: {split_summary.train_rows:,}")
    print(f"- Test rows: {split_summary.test_rows:,}")
    print(f"- Train dates: {split_summary.train_start.date()} to {split_summary.train_end.date()}")
    print(f"- Test dates: {split_summary.test_start.date()} to {split_summary.test_end.date()}")
    print(f"- Train class balance: {split_summary.train_class_balance}")
    print(f"- Test class balance: {split_summary.test_class_balance}")
    print("Fit warnings:")
    for model_name, warnings in warnings_by_model.items():
        print(f"- {model_name}: {warnings or ['None']}")
    print(f"\nSaved metrics: {metrics_path}")
    print("\nMetrics:")
    print(comparison.round(4).to_string(index=False))
    _print_metric_comparison(comparison)


if __name__ == "__main__":
    main()
