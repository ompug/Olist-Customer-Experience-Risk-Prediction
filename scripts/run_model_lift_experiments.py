"""Run model and feature lift experiments focused on ranking quality."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cx_risk.config import TABLES_DIR
from cx_risk.data import load_raw_data
from cx_risk.evaluation import create_model_comparison_table
from cx_risk.features import build_modeling_dataframe
from cx_risk.model_lift import TOP_K_FRACTIONS, build_topk_ranking_table
from cx_risk.models import create_model_pipelines, fit_baseline_models, predict_model_outputs
from cx_risk.preprocessing import build_preprocessor, get_primary_feature_columns, identify_feature_types
from cx_risk.utils import save_table
from cx_risk.validation import temporal_train_test_split

FEATURE_SET_CONFIGS = [
    {
        "feature_set": "baseline_history",
        "include_historical_features": True,
        "include_enhanced_historical_features": False,
    },
    {
        "feature_set": "enhanced_history",
        "include_historical_features": True,
        "include_enhanced_historical_features": True,
    },
]


def _create_calibrated_variants(pipelines):
    return {
        f"Calibrated {model_name}": CalibratedClassifierCV(
            estimator=clone(pipeline),
            method="sigmoid",
            cv=3,
        )
        for model_name, pipeline in pipelines.items()
    }


def _predict_outputs(models, X):
    outputs = {}
    for model_name, model in models.items():
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        outputs[model_name] = {"y_pred": y_pred, "y_proba": y_proba}
    return outputs


def _run_feature_set(tables, config):
    modeling_df = build_modeling_dataframe(
        tables,
        include_split_timestamp=True,
        include_historical_features=config["include_historical_features"],
        include_enhanced_historical_features=config["include_enhanced_historical_features"],
    )
    feature_columns = get_primary_feature_columns(
        include_historical_features=config["include_historical_features"],
        include_enhanced_historical_features=config["include_enhanced_historical_features"],
    )
    X_train, X_test, y_train, y_test, split_summary = temporal_train_test_split(
        modeling_df,
        feature_columns=feature_columns,
        return_summary=True,
    )
    numeric_features, categorical_features = identify_feature_types(X_train)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipelines = create_model_pipelines(preprocessor)

    fitted_baselines, warnings_by_model = fit_baseline_models(pipelines, X_train, y_train)
    calibrated_models = _create_calibrated_variants(pipelines)
    for model in calibrated_models.values():
        model.fit(X_train, y_train)

    outputs = predict_model_outputs(fitted_baselines, X_test)
    outputs.update(_predict_outputs(calibrated_models, X_test))
    metrics = create_model_comparison_table(y_test, outputs)
    metrics.insert(0, "feature_set", config["feature_set"])
    metrics.insert(1, "train_rows", split_summary.train_rows)
    metrics.insert(2, "test_rows", split_summary.test_rows)
    metrics.insert(3, "n_features", len(feature_columns))

    topk_frames = []
    for model_name, output in outputs.items():
        topk = build_topk_ranking_table(y_test, output["y_proba"], TOP_K_FRACTIONS)
        topk.insert(0, "model", model_name)
        topk.insert(0, "feature_set", config["feature_set"])
        topk_frames.append(topk)

    return metrics, pd.concat(topk_frames, ignore_index=True), {
        "feature_set": config["feature_set"],
        "n_features": len(feature_columns),
        "feature_columns": ";".join(feature_columns),
        "train_rows": split_summary.train_rows,
        "test_rows": split_summary.test_rows,
        "warnings": str(warnings_by_model),
    }


def main() -> None:
    tables = load_raw_data()
    metric_frames = []
    topk_frames = []
    feature_set_records = []

    for config in FEATURE_SET_CONFIGS:
        metrics, topk, feature_set_record = _run_feature_set(tables, config)
        metric_frames.append(metrics)
        topk_frames.append(topk)
        feature_set_records.append(feature_set_record)
        print(f"Completed model-lift feature set: {config['feature_set']}")

    experiment_metrics = pd.concat(metric_frames, ignore_index=True)
    topk_metrics = pd.concat(topk_frames, ignore_index=True)
    feature_sets = pd.DataFrame(feature_set_records)

    experiment_path = save_table(experiment_metrics, TABLES_DIR / "model_lift_experiments.csv")
    topk_path = save_table(topk_metrics, TABLES_DIR / "model_lift_topk.csv")
    feature_sets_path = save_table(feature_sets, TABLES_DIR / "model_lift_feature_sets.csv")

    best_top10 = (
        topk_metrics.loc[topk_metrics["top_frac"].round(2) == 0.10]
        .sort_values(["lift_at_k", "precision_at_k", "recall_at_k"], ascending=False)
        .iloc[0]
    )

    print("\nSaved model-lift artifacts:")
    print(f"- Experiment metrics: {experiment_path}")
    print(f"- Top-k metrics: {topk_path}")
    print(f"- Feature sets: {feature_sets_path}")
    print("\nBest top-10% ranking result:")
    print(f"- Feature set: {best_top10['feature_set']}")
    print(f"- Model: {best_top10['model']}")
    print(f"- Precision: {best_top10['precision_at_k']:.2%}")
    print(f"- Recall: {best_top10['recall_at_k']:.2%}")
    print(f"- Lift: {best_top10['lift_at_k']:.2f}x")


if __name__ == "__main__":
    main()
