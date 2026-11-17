"""Translation of model outputs into operating set-points (Table A1 'Outputs' column)."""
from __future__ import annotations

import numpy as np

from ..kpi import affinity_speed_ratio
from ..physics import mixer_min_speed_rps


def pressure_setpoint_bar(forecast_demand_m3_min, demand_capacity_m3_min: float,
                          p_min_bar: float, p_max_bar: float, margin_bar: float = 0.2):
    """Minimum safe compressor discharge pressure for a forecast air demand.

    The set-point rises linearly from p_min (zero demand) to p_max (full capacity),
    reflecting the larger distribution pressure drop at high flow, plus a safety
    margin, and is clipped to [p_min, p_max]. Calibrate p_min, p_max and the margin
    from the end-use pressure requirements and measured network pressure drops.
    """
    q = np.clip(np.asarray(forecast_demand_m3_min, float) / demand_capacity_m3_min, 0, 1)
    p = p_min_bar + q * (p_max_bar - p_min_bar) + margin_bar
    return np.clip(p, p_min_bar, p_max_bar)


def fan_speed_setpoint(forecast_flow, design_flow, min_speed_ratio: float = 0.3):
    """VFD speed ratio for a forecast airflow (flow is proportional to speed)."""
    r = np.asarray(forecast_flow, float) / design_flow
    return np.clip(r, min_speed_ratio, 1.0)


def fan_power_ratio(speed_ratio):
    """Shaft-power ratio for a fan speed ratio (affinity law, Eq. 5)."""
    return np.asarray(speed_ratio, float) ** 3


def mixer_speed_setpoint(viscosity_pa_s, rho_kg_m3: float, impeller_d_m: float,
                         rated_speed_rps: float, re_min: float = 1.0e4, margin: float = 1.10):
    """Lowest mixer speed (rev/s) that keeps turbulent mixing (Re >= re_min) with a
    safety margin, capped at the rated speed. The viscosity comes from the torque
    model (apparent viscosity estimate)."""
    mu = np.asarray(viscosity_pa_s, float)
    n = mixer_min_speed_rps(rho_kg_m3, impeller_d_m, mu, re_min) * margin
    return np.minimum(n, rated_speed_rps)


def speed_ratio_for_power_ratio(power_ratio: float) -> float:
    return affinity_speed_ratio(power_ratio)
