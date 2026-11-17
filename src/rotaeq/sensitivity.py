"""Sensitivity analyses reported in Section 3.5 of the manuscript."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import scenario_kpis
from .data import load_assumptions


def emission_factor_comparison(assumptions: dict | None = None) -> pd.DataFrame:
    """CO2 reductions per scenario under each emission factor in assumptions.yaml."""
    assumptions = assumptions or load_assumptions()
    cols = {}
    base = None
    for key, ef in assumptions["emission_factors"].items():
        df = scenario_kpis(assumptions=assumptions, ef_key=key)
        if base is None:
            base = df[["facility", "scenario"]].copy()
        cols[f"co2_t_{key}"] = df["co2_reduction_t"].round(1)
    out = pd.concat([base, pd.DataFrame(cols)], axis=1)
    ref = "co2_t_measur_default_us_average"
    for key in assumptions["emission_factors"]:
        col = f"co2_t_{key}"
        if col != ref:
            out[f"factor_{key}"] = (out[col] / out[ref]).round(3)
    return out


def max_tolerable_change(threshold_months: float | None = None,
                         assumptions: dict | None = None) -> pd.DataFrame:
    """Largest electricity-price reduction and implementation-cost increase that keep
    each payback period within the threshold (12 months by default).

    For SPP = 12*C/dC, the price can fall to SPP/T of its value and the cost can rise
    to T/SPP of its value before SPP reaches the threshold T.
    """
    assumptions = assumptions or load_assumptions()
    t = threshold_months or assumptions["payback_threshold_months"]
    df = scenario_kpis(assumptions=assumptions)
    spp = df["payback_months"].replace(0, np.nan)
    out = df[["facility", "scenario", "payback_months"]].copy()
    out["max_price_reduction_pct"] = (100 * (1 - spp / t)).round(1)
    out["max_cost_increase_pct"] = (100 * (t / spp - 1)).round(1)
    return out


def payback_vs_price(factors=(0.7, 0.8, 0.85, 0.9, 1.0, 1.1, 1.2)) -> pd.DataFrame:
    """Payback periods (months) for a range of electricity-price multipliers."""
    frames = []
    for f in factors:
        df = scenario_kpis(price_factor=f)[["facility", "scenario", "payback_months"]]
        frames.append(df.assign(price_factor=f))
    return pd.concat(frames).pivot_table(index=["facility", "scenario"],
                                         columns="price_factor", values="payback_months").round(2)
