"""Score the future holdout split and write an operational risk queue."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cx_risk.config import TABLES_DIR
from cx_risk.data import load_raw_data
from cx_risk.explainability import add_reason_codes_to_queue, build_global_importance_tables
from cx_risk.features import build_modeling_dataframe
from cx_risk.models import predict_model_outputs
from cx_risk.scoring import (
    MODEL_REGISTRY_PATH,
    build_combined_risk_queue,
    load_model_pipeline,
    load_model_registry,
    validate_scoring_features,
)
from cx_risk.utils import save_table
from cx_risk.validation import temporal_train_test_split

IMPORTANCE_SAMPLE_SIZE = 5000


def main() -> None:
    registry = load_model_registry(MODEL_REGISTRY_PATH)
    feature_columns = registry["feature_columns"]

    tables = load_raw_data()
    modeling_df = build_modeling_dataframe(
        tables,
        include_order_id=True,
        include_split_timestamp=True,
        include_historical_features=True,
    )
    X_train, X_test, y_train, y_test, split_summary = temporal_train_test_split(
        modeling_df,
        feature_columns=feature_columns,
        return_summary=True,
    )
    validate_scoring_features(X_test, feature_columns)

    models = {
        record["name"]: load_model_pipeline(record["name"])
        for record in registry["models"]
    }
    outputs = predict_model_outputs(models, X_test)
    model_scores = {
        model_name: output["y_proba"]
        for model_name, output in outputs.items()
    }

    sorted_df = (
        modeling_df.dropna(subset=["order_purchase_timestamp"])
        .sort_values("order_purchase_timestamp", kind="mergesort")
        .reset_index(drop=True)
    )
    cutoff_timestamp = sorted_df.loc[int(len(sorted_df) * 0.80), "order_purchase_timestamp"]
    holdout_df = sorted_df.loc[sorted_df["order_purchase_timestamp"] >= cutoff_timestamp].copy()

    if len(holdout_df) != len(X_test):
        raise ValueError("Holdout context rows do not match scored feature rows.")

    scoring_context = X_test.reset_index(drop=True).copy()
    scoring_context.insert(0, "order_purchase_timestamp", holdout_df["order_purchase_timestamp"].reset_index(drop=True))
    queue = build_combined_risk_queue(
        model_scores,
        order_ids=holdout_df["order_id"].reset_index(drop=True),
        X_scoring=scoring_context.reset_index(drop=True),
        y_true=y_test.reset_index(drop=True),
    )
    queue = add_reason_codes_to_queue(queue)
    queue_path = save_table(queue, TABLES_DIR / "risk_queue.csv")

    if len(X_test) > IMPORTANCE_SAMPLE_SIZE:
        importance_X = X_test.sample(n=IMPORTANCE_SAMPLE_SIZE, random_state=42)
        importance_y = y_test.loc[importance_X.index]
    else:
        importance_X = X_test
        importance_y = y_test
    importance_table = build_global_importance_tables(models, importance_X, importance_y)
    importance_path = save_table(importance_table, TABLES_DIR / "global_feature_importance.csv")
    top20_path = save_table(
        importance_table.loc[importance_table["rank"] <= 20].copy(),
        TABLES_DIR / "global_feature_importance_top20.csv",
    )

    default_model = registry.get("default_model", "HistGradientBoostingClassifier")
    default_queue = queue.loc[queue["model"] == default_model]
    high_priority = default_queue["risk_band"].isin(["critical", "high"]).sum()
    print("Scored future holdout orders for the risk queue.")
    print(f"- Holdout rows per model: {len(default_queue):,}")
    print(f"- Models scored: {len(models)}")
    print(f"- Default model: {default_model}")
    print(f"- High/critical queue size: {high_priority:,}")
    print(f"- Saved queue: {queue_path}")
    print(f"- Saved global importance: {importance_path}")
    print(f"- Saved top-20 importance: {top20_path}")
    print(f"- Importance rows sampled: {len(importance_X):,}")


if __name__ == "__main__":
    main()
