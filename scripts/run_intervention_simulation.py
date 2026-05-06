"""Simulate proactive intervention queues from model risk probabilities."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cx_risk.config import FIGURES_DIR, TABLES_DIR
from cx_risk.data import load_raw_data
from cx_risk.features import build_modeling_dataframe
from cx_risk.intervention import build_intervention_curve, evaluate_thresholds
from cx_risk.models import create_model_pipelines, fit_baseline_models, predict_model_outputs
from cx_risk.preprocessing import build_preprocessor, get_primary_feature_columns, identify_feature_types
from cx_risk.utils import ensure_directory, save_table
from cx_risk.validation import temporal_train_test_split

TOP_FRACTIONS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]
PROBABILITY_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def _train_history_models():
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
    return outputs, y_test, split_summary, warnings_by_model


def _build_intervention_tables(model_outputs, y_test):
    intervention_frames = []
    threshold_frames = []
    for model_name, outputs in model_outputs.items():
        intervention_curve = build_intervention_curve(
            y_test,
            outputs["y_proba"],
            TOP_FRACTIONS,
        )
        intervention_curve.insert(0, "model", model_name)
        intervention_frames.append(intervention_curve)

        threshold_analysis = evaluate_thresholds(
            y_test,
            outputs["y_proba"],
            PROBABILITY_THRESHOLDS,
        )
        threshold_analysis.insert(0, "model", model_name)
        threshold_frames.append(threshold_analysis)

    intervention_table = pd.concat(intervention_frames, ignore_index=True)
    threshold_table = pd.concat(threshold_frames, ignore_index=True)
    return intervention_table, threshold_table


def _plot_metric_by_flagged_share(intervention_table, metric: str, ylabel: str, filename: str) -> Path:
    ensure_directory(FIGURES_DIR)
    fig, ax = plt.subplots(figsize=(9, 6))
    for model_name, model_df in intervention_table.groupby("model"):
        plot_df = model_df.sort_values("pct_orders_flagged")
        ax.plot(
            plot_df["pct_orders_flagged"] * 100,
            plot_df[metric],
            marker="o",
            linewidth=2,
            label=model_name,
        )
    ax.set_xlabel("Orders flagged for intervention (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by intervention queue size")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_intervention_plots(intervention_table) -> list[Path]:
    return [
        _plot_metric_by_flagged_share(
            intervention_table,
            metric="recall",
            ylabel="Recall",
            filename="intervention_recall_by_flagged_share.png",
        ),
        _plot_metric_by_flagged_share(
            intervention_table,
            metric="precision",
            ylabel="Precision",
            filename="intervention_precision_by_flagged_share.png",
        ),
        _plot_metric_by_flagged_share(
            intervention_table,
            metric="lift_over_random",
            ylabel="Lift over random",
            filename="intervention_lift_by_flagged_share.png",
        ),
    ]


def _print_business_summary(intervention_table) -> None:
    top_10 = intervention_table.loc[intervention_table["top_frac"] == 0.10].copy()
    best_row = top_10.sort_values(
        ["lift_over_random", "precision", "recall"],
        ascending=False,
    ).iloc[0]

    print("\nBusiness summary at top 10% intervention queue:")
    print(f"- Best model: {best_row['model']}")
    print(f"- Test-set base low-review rate: {best_row['base_positive_rate']:.2%}")
    print(f"- Orders flagged: {int(best_row['n_flagged']):,}")
    print(f"- Low-review orders caught: {int(best_row['true_positives']):,}")
    print(f"- Precision: {best_row['precision']:.2%}")
    print(f"- Recall: {best_row['recall']:.2%}")
    print(f"- Lift over random selection: {best_row['lift_over_random']:.2f}x")
    print(
        "By flagging the top 10% riskiest future orders, this model identifies "
        f"{best_row['recall']:.2%} of future low-review cases with "
        f"{best_row['lift_over_random']:.2f}x lift over random selection."
    )


def main() -> None:
    model_outputs, y_test, split_summary, warnings_by_model = _train_history_models()
    intervention_table, threshold_table = _build_intervention_tables(model_outputs, y_test)

    intervention_path = save_table(intervention_table, TABLES_DIR / "intervention_simulation.csv")
    threshold_path = save_table(threshold_table, TABLES_DIR / "threshold_analysis.csv")
    figure_paths = _save_intervention_plots(intervention_table)

    print("Intervention simulation uses time-aware validation with historical features:")
    print(f"- Train rows: {split_summary.train_rows:,}")
    print(f"- Test rows: {split_summary.test_rows:,}")
    print(f"- Train dates: {split_summary.train_start.date()} to {split_summary.train_end.date()}")
    print(f"- Test dates: {split_summary.test_start.date()} to {split_summary.test_end.date()}")
    print("Fit warnings:")
    for model_name, warnings in warnings_by_model.items():
        print(f"- {model_name}: {warnings or ['None']}")
    print(f"\nSaved intervention table: {intervention_path}")
    print(f"Saved threshold table: {threshold_path}")
    print("Saved figures:")
    for path in figure_paths:
        print(f"- {path}")

    print("\nTop-fraction intervention results:")
    print(intervention_table.round(4).to_string(index=False))
    print("\nProbability-threshold analysis:")
    print(threshold_table.round(4).to_string(index=False))
    _print_business_summary(intervention_table)


if __name__ == "__main__":
    main()
