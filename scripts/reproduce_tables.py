"""Recompute the KPIs of Tables 4-9 and check them against the manuscript.

Usage:  python scripts/reproduce_tables.py
Writes: results/scenario_kpis.csv, results/consistency_report.csv, results/table9_summary.csv
Exit code 1 if any recomputed value falls outside the stated tolerance.
"""
from pathlib import Path
import sys

import pandas as pd

from rotaeq.analysis import consistency_report, cross_system_summary, scenario_kpis

OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    df = scenario_kpis()
    rep = consistency_report(df)
    summary = cross_system_summary(df)
    df.to_csv(OUT / "scenario_kpis.csv", index=False)
    rep.to_csv(OUT / "consistency_report.csv", index=False)
    summary.to_csv(OUT / "table9_summary.csv")
    print("Recomputed KPIs (Eqs. 1-4) vs. manuscript values\n")
    print(rep.to_string(index=False))
    print("\nCross-system summary (Table 9)\n")
    print(summary.to_string())
    checks = rep[["savings_ok", "cost_ok", "payback_ok"]]
    failed = (checks == False).any(axis=1)  # noqa: E712  (None = not reported)
    if failed.any():
        print("\nOutside tolerance:", ", ".join(rep.loc[failed, "scenario"]))
        return 1
    print("\nAll recomputed values agree with the manuscript within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
