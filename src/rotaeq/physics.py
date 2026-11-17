"""Physical relations used to interpret the MEASUR scenarios (Sections 3.3 and 4)."""
from __future__ import annotations

import math

from .kpi import affinity_power_ratio, affinity_speed_ratio

PSI_PER_BAR = 14.503773773


def psi_to_bar(psi: float) -> float:
    return psi / PSI_PER_BAR


def bar_to_psi(bar: float) -> float:
    return bar * PSI_PER_BAR


def airflow_ratio_after_leak_repair(leak_before: float, leak_after: float) -> float:
    """Ratio of required compressor airflow after/before leak repair.

    With constant useful demand Q_u, total flow is Q = Q_u / (1 - L).
    Hence Q_after / Q_before = (1 - L_before) / (1 - L_after).
    """
    for leak in (leak_before, leak_after):
        if not 0 <= leak < 1:
            raise ValueError("leakage fractions must be in [0, 1)")
    return (1.0 - leak_before) / (1.0 - leak_after)


def speed_ratio_for_power(p_baseline_kw: float, p_optimized_kw: float) -> float:
    """Speed ratio implied by an affinity-law power change (fans, pumps, mixers)."""
    return affinity_speed_ratio(p_optimized_kw / p_baseline_kw)


def power_for_speed_ratio(p_baseline_kw: float, speed_ratio: float) -> float:
    return p_baseline_kw * affinity_power_ratio(speed_ratio)


def pump_input_power_kw(hydraulic_power_kw: float, pump_efficiency: float,
                        motor_efficiency: float = 1.0) -> float:
    """Electrical input power for a given hydraulic output and efficiencies."""
    if not 0 < pump_efficiency <= 1 or not 0 < motor_efficiency <= 1:
        raise ValueError("efficiencies must be in (0, 1]")
    return hydraulic_power_kw / (pump_efficiency * motor_efficiency)


def mixer_reynolds(rho_kg_m3: float, speed_rps: float, impeller_d_m: float,
                   viscosity_pa_s: float) -> float:
    """Impeller Reynolds number Re = rho * N * D^2 / mu."""
    return rho_kg_m3 * speed_rps * impeller_d_m ** 2 / viscosity_pa_s


def mixer_min_speed_rps(rho_kg_m3: float, impeller_d_m: float, viscosity_pa_s: float,
                        re_min: float = 1.0e4) -> float:
    """Minimum impeller speed (rev/s) that keeps Re >= re_min (turbulent mixing)."""
    return re_min * viscosity_pa_s / (rho_kg_m3 * impeller_d_m ** 2)


def energy_ratio_from_load_factor(lf_baseline: float, lf_optimized: float) -> float:
    """Energy ratio for unchanged operating hours when input power scales with load factor."""
    return lf_optimized / lf_baseline


def shutdown_energy_kwh(power_kw: float, hours_per_day: float, days: float) -> float:
    """Energy avoided by switching a load off for a number of hours per day."""
    return power_kw * hours_per_day * days


def is_close(a: float, b: float, rel: float = 0.01) -> bool:
    return math.isclose(a, b, rel_tol=rel)
