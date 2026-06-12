"""Generate business-value scenarios for intervention strategies."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cx_risk.business_value import build_business_value_table, build_sensitivity_table
from cx_risk.config import TABLES_DIR
from cx_risk.utils import save_table

import pandas as pd

INTERVENTION_PATH = TABLES_DIR / "intervention_simulation.csv"
BUSINESS_VALUE_PATH = TABLES_DIR / "business_value_simulation.csv"
SENSITIVITY_PATH = TABLES_DIR / "business_value_sensitivity.csv"


def main() -> None:
    if not INTERVENTION_PATH.exists():
        raise FileNotFoundError(
            f"Missing intervention simulation artifact: {INTERVENTION_PATH}. "
            "Run `make intervention` first."
        )

    intervention_table = pd.read_csv(INTERVENTION_PATH)
    business_value = build_business_value_table(intervention_table)
    sensitivity = build_sensitivity_table(intervention_table)

    business_value_path = save_table(business_value, BUSINESS_VALUE_PATH)
    sensitivity_path = save_table(sensitivity, SENSITIVITY_PATH)

    best_row = business_value.sort_values("net_value", ascending=False).iloc[0]
    print("Generated business-value scenarios from intervention simulation.")
    print(f"- Scenario rows: {len(business_value):,}")
    print(f"- Sensitivity rows: {len(sensitivity):,}")
    print(f"- Saved scenarios: {business_value_path}")
    print(f"- Saved sensitivity: {sensitivity_path}")
    print("\nBest default scenario by net value:")
    print(f"- Model: {best_row['model']}")
    print(f"- Queue share: {best_row['top_frac']:.0%}")
    print(f"- Intervention: {best_row['intervention_type']}")
    print(f"- Expected saves: {best_row['expected_saved_low_reviews']:.1f}")
    print(f"- Net value: ${best_row['net_value']:,.2f}")
    print(f"- ROI: {best_row['roi']:.2f}x")


if __name__ == "__main__":
    main()
