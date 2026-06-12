"""Run rolling validation, calibration, segment, and drift monitoring."""

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
from cx_risk.monitoring import (
    add_monitoring_segments,
    build_calibration_bins,
    build_feature_drift_report,
    build_segment_performance,
    compute_calibration_metrics,
    create_rolling_windows,
)
from cx_risk.preprocessing import build_preprocessor, get_primary_feature_columns, identify_feature_types
from cx_risk.utils import save_table

SEGMENT_COLUMNS = [
    "customer_state",
    "primary_category",
    "seller_risk_tier",
    "payment_profile",
    "order_value_tier",
]


def _window_summary_record(window) -> dict:
    return {
        "window": window.window,
        "train_rows": len(window.train_df),
        "test_rows": len(window.test_df),
        "train_start": window.train_start,
        "train_end": window.train_end,
        "test_start": window.test_start,
        "test_end": window.test_end,
    }


def main() -> None:
    tables = load_raw_data()
    modeling_df = build_modeling_dataframe(
        tables,
        include_split_timestamp=True,
        include_historical_features=True,
    )
    feature_columns = get_primary_feature_columns(include_historical_features=True)
    windows = create_rolling_windows(modeling_df)

    metric_frames = []
    calibration_metric_records = []
    calibration_bin_frames = []
    segment_frames = []
    drift_frames = []

    for window in windows:
        X_train = window.train_df[feature_columns].copy()
        X_test = window.test_df[feature_columns].copy()
        y_train = window.train_df["low_review"].copy()
        y_test = window.test_df["low_review"].copy()

        numeric_features, categorical_features = identify_feature_types(X_train)
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        pipelines = create_model_pipelines(preprocessor)
        fitted_models, warnings_by_model = fit_baseline_models(pipelines, X_train, y_train)
        outputs = predict_model_outputs(fitted_models, X_test)

        metrics = create_model_comparison_table(y_test, outputs)
        for key, value in _window_summary_record(window).items():
            metrics.insert(0, key, value)
        metric_frames.append(metrics)

        segmented_context = add_monitoring_segments(X_test).reset_index(drop=True)
        for model_name, output in outputs.items():
            y_proba = pd.Series(output["y_proba"]).reset_index(drop=True)
            y_pred = pd.Series(output["y_pred"]).reset_index(drop=True)
            calibration_metrics = compute_calibration_metrics(y_test.reset_index(drop=True), y_proba)
            calibration_metric_records.append(
                {
                    **_window_summary_record(window),
                    "model": model_name,
                    **calibration_metrics,
                }
            )

            bins = build_calibration_bins(y_test.reset_index(drop=True), y_proba)
            bins.insert(0, "model", model_name)
            bins.insert(0, "window", window.window)
            calibration_bin_frames.append(bins)

            scored_segments = segmented_context.copy()
            scored_segments["model"] = model_name
            scored_segments["actual_low_review"] = y_test.reset_index(drop=True)
            scored_segments["predicted_probability"] = y_proba
            scored_segments["predicted_label"] = y_pred
            segment_perf = build_segment_performance(scored_segments, SEGMENT_COLUMNS)
            segment_perf.insert(0, "window", window.window)
            segment_frames.append(segment_perf)

        drift = build_feature_drift_report(X_train, X_test)
        drift.insert(0, "window", window.window)
        drift_frames.append(drift)

        print(f"Completed monitoring window {window.window}: {window.train_start.date()} to {window.test_end.date()}")
        for model_name, warnings in warnings_by_model.items():
            if warnings:
                print(f"- {model_name} warnings: {warnings}")

    rolling_metrics = pd.concat(metric_frames, ignore_index=True)
    calibration_metrics = pd.DataFrame(calibration_metric_records)
    calibration_bins = pd.concat(calibration_bin_frames, ignore_index=True)
    segment_performance = pd.concat(segment_frames, ignore_index=True)
    drift_report = pd.concat(drift_frames, ignore_index=True)

    rolling_path = save_table(rolling_metrics, TABLES_DIR / "rolling_backtest_metrics.csv")
    calibration_metrics_path = save_table(calibration_metrics, TABLES_DIR / "calibration_metrics.csv")
    calibration_bins_path = save_table(calibration_bins, TABLES_DIR / "calibration_bins.csv")
    segment_path = save_table(segment_performance, TABLES_DIR / "segment_performance.csv")
    drift_path = save_table(drift_report, TABLES_DIR / "feature_drift_report.csv")

    print("\nSaved monitoring artifacts:")
    print(f"- Rolling metrics: {rolling_path}")
    print(f"- Calibration metrics: {calibration_metrics_path}")
    print(f"- Calibration bins: {calibration_bins_path}")
    print(f"- Segment performance: {segment_path}")
    print(f"- Feature drift: {drift_path}")

    summary = (
        rolling_metrics.groupby("model", as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            min_roc_auc=("roc_auc", "min"),
            max_roc_auc=("roc_auc", "max"),
            mean_average_precision=("pr_auc_average_precision", "mean"),
        )
        .sort_values("mean_roc_auc", ascending=False)
    )
    print("\nRolling validation summary:")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
