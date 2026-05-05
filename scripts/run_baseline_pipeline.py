"""Run the extracted baseline Olist customer-risk pipeline."""

from pathlib import Path
import sys

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cx_risk.config import RANDOM_STATE
from cx_risk.data import load_raw_data
from cx_risk.evaluation import create_model_comparison_table
from cx_risk.features import build_modeling_dataframe
from cx_risk.models import create_model_pipelines, fit_baseline_models, predict_model_outputs
from cx_risk.preprocessing import (
    assert_no_forbidden_columns,
    build_preprocessor,
    identify_feature_types,
    split_features_target,
)


def main() -> None:
    tables = load_raw_data()
    modeling_df = build_modeling_dataframe(tables)
    X, y = split_features_target(modeling_df)
    assert_no_forbidden_columns(X.columns)
    numeric_features, categorical_features = identify_feature_types(X)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    pipelines = create_model_pipelines(preprocessor)
    fitted_models, warnings_by_model = fit_baseline_models(pipelines, X_train, y_train)
    outputs = predict_model_outputs(fitted_models, X_test)
    comparison = create_model_comparison_table(y_test, outputs)

    print(f"Modeling rows: {len(modeling_df):,}")
    print(f"Features: {X.shape[1]} ({len(numeric_features)} numeric, {len(categorical_features)} categorical)")
    print("Fit warnings:")
    for model_name, warnings in warnings_by_model.items():
        print(f"- {model_name}: {warnings or ['None']}")
    print("\nMetrics:")
    print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
