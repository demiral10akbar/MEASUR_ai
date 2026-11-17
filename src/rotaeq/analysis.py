"""Re-computation of the KPIs reported in Tables 4-9 from the MEASUR energy and demand results.

The MEASUR runs themselves are not executed here (MEASUR is a separate desktop/web
application). This module takes the baseline and scenario energy and peak-demand
values produced by MEASUR, applies Eqs. (1)-(4) with the assumptions in
data/assumptions.yaml, and checks the results against the values reported in the
manuscript.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from . import kpi
from .data import load_assumptions, load_facilities, value

EF_DEFAULT = "measur_default_us_average"


@dataclass
class ScenarioKPI:
    facility: str
    scenario: str
    name: str
    units: int
    baseline_kwh: float
    optimized_kwh: float
    delta_kwh: float
    savings_pct: float
    delta_peak_kw: float
    co2_reduction_t: float
    cost_saving_usd: float
    implementation_cost_usd: float
    payback_months: float
    reported_savings_pct: float | None
    reported_cost_saving_usd: float | None
    reported_payback_months: float | None


def scenario_kpis(facilities: dict | None = None, assumptions: dict | None = None,
                  ef_key: str = EF_DEFAULT, price_factor: float = 1.0,
                  cost_factor: float = 1.0) -> pd.DataFrame:
    """KPIs for every scenario. Facility-level totals are used (per-unit values x units).

    price_factor and cost_factor scale electricity prices and implementation costs
    (used by the sensitivity analysis).
    """
    facilities = facilities or load_facilities()
    assumptions = assumptions or load_assumptions()
    ef = assumptions["emission_factors"][ef_key]["value"]
    rows = []
    for fid, fac in facilities.items():
        tariff = assumptions["tariffs"][fac["tariff"]]
        price = tariff["energy_usd_per_kwh"] * price_factor
        demand = tariff.get("demand_usd_per_kw_yr", 0) * price_factor
        units = int(fac.get("units", 1))
        base = fac["baseline"]
        for sc in fac["scenarios"]:
            res = sc["result"]
            e_b = base["energy_kwh"] * units
            e_o = res["energy_kwh"] * units
            d_e = kpi.energy_saving(e_b, e_o)
            d_peak = (base.get("peak_kw", 0) - res.get("peak_kw", 0)) * units
            d_cost = kpi.cost_saving(d_e, price, d_peak, demand)
            impl_entry = sc["implementation_cost_usd"]
            impl = value(impl_entry) * (units if impl_entry.get("per_unit") else 1) * cost_factor
            rep = sc.get("reported", {})
            rows.append(ScenarioKPI(
                facility=fid, scenario=sc["id"], name=sc["name"], units=units,
                baseline_kwh=e_b, optimized_kwh=e_o, delta_kwh=d_e,
                savings_pct=kpi.energy_saving_pct(e_b, e_o),
                delta_peak_kw=d_peak,
                co2_reduction_t=kpi.co2_reduction_t(d_e, ef),
                cost_saving_usd=d_cost,
                implementation_cost_usd=impl,
                payback_months=kpi.simple_payback_months(impl, d_cost),
                reported_savings_pct=rep.get("savings_pct"),
                reported_cost_saving_usd=rep.get("total_cost_savings_usd"),
                reported_payback_months=rep.get("payback_months"),
            ))
    return pd.DataFrame([asdict(r) for r in rows])


def consistency_report(df: pd.DataFrame, rel_tol: float = 0.02,
                       payback_abs_tol: float = 0.6) -> pd.DataFrame:
    """Compare recomputed KPIs with the values reported in the manuscript.

    Tolerances: 2 % relative for savings and cost (rounding in the manuscript and
    MEASUR's internal precision), 0.6 months absolute for payback periods that the
    manuscript reports as approximate values (e.g. "~10 months").
    """
    out = df[["facility", "scenario"]].copy()

    def rel_ok(calc, rep):
        return [None if r is None or pd.isna(r) else abs(c - r) <= rel_tol * max(abs(r), 1e-9)
                for c, r in zip(calc, rep)]

    out["savings_pct_calc"] = df["savings_pct"].round(2)
    out["savings_pct_reported"] = df["reported_savings_pct"]
    # Values reported as whole percentages (e.g. "28%") are checked to +/-0.5 pp,
    # values reported to one decimal to +/-0.15 pp.
    out["savings_ok"] = [None if pd.isna(r) else
                         abs(c - r) <= (0.5 if float(r).is_integer() else 0.15)
                         for c, r in zip(df["savings_pct"], df["reported_savings_pct"])]
    out["cost_saving_calc"] = df["cost_saving_usd"].round(0)
    out["cost_saving_reported"] = df["reported_cost_saving_usd"]
    out["cost_ok"] = rel_ok(df["cost_saving_usd"], df["reported_cost_saving_usd"])
    out["payback_calc"] = df["payback_months"].round(2)
    out["payback_reported"] = df["reported_payback_months"]
    out["payback_ok"] = [None if pd.isna(r) else abs(c - r) <= payback_abs_tol
                         for c, r in zip(df["payback_months"], df["reported_payback_months"])]
    return out


def cross_system_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Facility-level ranges corresponding to Table 9."""
    g = df.groupby("facility")
    return pd.DataFrame({
        "savings_pct_min": g["savings_pct"].min().round(1),
        "savings_pct_max": g["savings_pct"].max().round(1),
        "energy_saving_kwh_min": g["delta_kwh"].min().round(0),
        "energy_saving_kwh_max": g["delta_kwh"].max().round(0),
        "co2_t_min": g["co2_reduction_t"].min().round(1),
        "co2_t_max": g["co2_reduction_t"].max().round(1),
        "payback_months_min": g["payback_months"].min().round(2),
        "payback_months_max": g["payback_months"].max().round(2),
    })
