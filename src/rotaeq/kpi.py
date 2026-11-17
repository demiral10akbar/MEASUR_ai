"""Key performance indicators - Equations (1) to (5) of the manuscript (Section 3.4).

All energies are annual electrical energies in kWh, powers in kW, costs in USD.
"""
from __future__ import annotations


def energy_saving(e_baseline_kwh: float, e_optimized_kwh: float) -> float:
    """Eq. (1a): annual energy saving dE = E_b - E_o (kWh/yr)."""
    return e_baseline_kwh - e_optimized_kwh


def energy_saving_pct(e_baseline_kwh: float, e_optimized_kwh: float) -> float:
    """Eq. (1b): percentage energy saving ES = 100 * dE / E_b."""
    if e_baseline_kwh <= 0:
        raise ValueError("baseline energy must be positive")
    return 100.0 * energy_saving(e_baseline_kwh, e_optimized_kwh) / e_baseline_kwh


def co2_reduction_t(delta_e_kwh: float, ef_kg_per_mwh: float) -> float:
    """Eq. (2): dCO2 = dE * EF_grid, returned in tonnes CO2 per year."""
    return delta_e_kwh / 1000.0 * ef_kg_per_mwh / 1000.0


def co2_emissions_t(energy_kwh: float, ef_kg_per_mwh: float) -> float:
    """Annual CO2 emissions (t/yr) for a given annual energy use."""
    return co2_reduction_t(energy_kwh, ef_kg_per_mwh)


def cost_saving(delta_e_kwh: float, price_usd_per_kwh: float,
                delta_peak_kw: float = 0.0, demand_usd_per_kw_yr: float = 0.0) -> float:
    """Eq. (3): dC = dE * c_e + dP_peak * c_d (USD/yr)."""
    return delta_e_kwh * price_usd_per_kwh + delta_peak_kw * demand_usd_per_kw_yr


def simple_payback_months(implementation_cost_usd: float, annual_saving_usd: float) -> float:
    """Eq. (4): SPP = 12 * C_impl / dC (months). Returns 0 for zero cost, inf for zero saving."""
    if implementation_cost_usd == 0:
        return 0.0
    if annual_saving_usd <= 0:
        return float("inf")
    return 12.0 * implementation_cost_usd / annual_saving_usd


def affinity_power_ratio(speed_ratio: float) -> float:
    """Eq. (5): P2/P1 = (n2/n1)**3 for fans, pumps and mixers."""
    return speed_ratio ** 3


def affinity_speed_ratio(power_ratio: float) -> float:
    """Inverse of Eq. (5): n2/n1 = (P2/P1)**(1/3)."""
    if power_ratio < 0:
        raise ValueError("power ratio must be non-negative")
    return power_ratio ** (1.0 / 3.0)
