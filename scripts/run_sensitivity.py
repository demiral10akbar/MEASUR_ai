"""Sensitivity analyses of Section 3.5 (emission factor, electricity price, implementation cost).

Usage:  python scripts/run_sensitivity.py
Writes: results/sensitivity_emission_factor.csv, results/sensitivity_thresholds.csv,
        results/sensitivity_payback_vs_price.csv
"""
from pathlib import Path

from rotaeq.sensitivity import emission_factor_comparison, max_tolerable_change, payback_vs_price

OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ef = emission_factor_comparison()
    thr = max_tolerable_change()
    pvp = payback_vs_price()
    ef.to_csv(OUT / "sensitivity_emission_factor.csv", index=False)
    thr.to_csv(OUT / "sensitivity_thresholds.csv", index=False)
    pvp.to_csv(OUT / "sensitivity_payback_vs_price.csv")
    print("CO2 reductions (t/yr) under alternative grid emission factors\n")
    print(ef.to_string(index=False))
    print("\nLargest price reduction / cost increase keeping payback <= 12 months\n")
    print(thr.round(2).to_string(index=False))
    print("\nPayback period (months) vs. electricity-price multiplier\n")
    print(pvp.to_string())


if __name__ == "__main__":
    main()
