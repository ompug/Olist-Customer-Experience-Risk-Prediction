"""Train and persist production-style scoring models for the risk queue."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cx_risk.config import MODEL_DIR
from cx_risk.data import load_raw_data
from cx_risk.evaluation import create_model_comparison_table
from cx_risk.features import build_modeling_dataframe
from cx_risk.models import create_model_pipelines, fit_baseline_models, predict_model_outputs
from cx_risk.preprocessing import build_preprocessor, get_primary_feature_columns, identify_feature_types
from cx_risk.scoring import (
    DEFAULT_QUEUE_MODEL,
    MODEL_REGISTRY_PATH,
    model_artifact_path,
    model_slug,
    save_model_pipeline,
    save_model_registry,
    utc_timestamp,
    validate_scoring_features,
)
from cx_risk.validation import temporal_train_test_split


def main() -> None:
    tables = load_raw_data()
    modeling_df = build_modeling_dataframe(
        tables,
        include_order_id=True,
        include_split_timestamp=True,
        include_historical_features=True,
    )
    feature_columns = get_primary_feature_columns(include_historical_features=True)
    X_train, X_test, y_train, y_test, split_summary = temporal_train_test_split(
        modeling_df,
        feature_columns=feature_columns,
        return_summary=True,
    )
    validate_scoring_features(X_train, feature_columns)
    validate_scoring_features(X_test, feature_columns)

    numeric_features, categorical_features = identify_feature_types(X_train)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipelines = create_model_pipelines(preprocessor)
    fitted_models, warnings_by_model = fit_baseline_models(pipelines, X_train, y_train)
    outputs = predict_model_outputs(fitted_models, X_test)
    metrics = create_model_comparison_table(y_test, outputs)

    model_records = []
    for model_name, pipeline in fitted_models.items():
        artifact_path = save_model_pipeline(model_name, pipeline, model_dir=MODEL_DIR)
        model_metric = metrics.loc[metrics["model"] == model_name].iloc[0].to_dict()
        model_records.append(
            {
                "name": model_name,
                "slug": model_slug(model_name),
                "artifact_path": str(artifact_path.relative_to(PROJECT_ROOT)),
                "warnings": warnings_by_model.get(model_name, []),
                "metrics": model_metric,
            }
        )

    registry = {
        "created_at": utc_timestamp(),
        "default_model": DEFAULT_QUEUE_MODEL,
        "scoring_mode": "future_holdout_simulation",
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "split_summary": split_summary.as_dict(),
        "models": model_records,
    }
    registry_path = save_model_registry(registry, MODEL_REGISTRY_PATH)

    print("Trained scoring models with leakage-safe historical features.")
    print(f"- Train rows: {split_summary.train_rows:,}")
    print(f"- Holdout rows: {split_summary.test_rows:,}")
    print(f"- Train dates: {split_summary.train_start.date()} to {split_summary.train_end.date()}")
    print(f"- Holdout dates: {split_summary.test_start.date()} to {split_summary.test_end.date()}")
    print(f"- Saved registry: {registry_path}")
    print("Saved model artifacts:")
    for record in model_records:
        print(f"- {record['name']}: {model_artifact_path(record['name']).relative_to(PROJECT_ROOT)}")
    print("\nHoldout metrics:")
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
