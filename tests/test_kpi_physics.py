import math

import numpy as np
import pytest

from rotaeq import kpi, physics


def test_energy_saving_eq1():
    assert kpi.energy_saving(254134, 204480) == 49654
    assert kpi.energy_saving_pct(254134, 204480) == pytest.approx(19.54, abs=0.01)


def test_co2_eq2_measur_default():
    # 1,087,477 kWh x 401.07 kg/MWh = 436.2 t (Facility B, Scenario 1)
    assert kpi.co2_reduction_t(1087477, 401.07) == pytest.approx(436.2, abs=0.1)


def test_cost_eq3_with_demand_charge():
    # Facility B, Scenario 1: energy 1,087,477 kWh x 0.102 + 137.31 kW x 120 USD/kW-yr
    assert kpi.cost_saving(1087477, 0.102, 137.31, 120) == pytest.approx(127399.6, abs=1)


def test_payback_eq4():
    assert kpi.simple_payback_months(5000, 5868.21) == pytest.approx(10.22, abs=0.01)
    assert kpi.simple_payback_months(0, 100) == 0
    assert math.isinf(kpi.simple_payback_months(100, 0))


def test_affinity_eq5_roundtrip():
    assert kpi.affinity_power_ratio(0.9) == pytest.approx(0.729)
    assert kpi.affinity_speed_ratio(kpi.affinity_power_ratio(0.65)) == pytest.approx(0.65)


def test_fan_speed_implied_by_shaft_power():
    # Facility C: 89.3 kW -> 63.8 kW shaft power implies ~10.6 % lower speed
    assert physics.speed_ratio_for_power(89.3, 63.8) == pytest.approx(0.894, abs=0.002)


def test_leak_repair_airflow():
    # 30 % -> 5 % leakage: required airflow falls to 0.70/0.95 = 73.7 %
    assert physics.airflow_ratio_after_leak_repair(0.30, 0.05) == pytest.approx(0.7368, abs=1e-4)
    with pytest.raises(ValueError):
        physics.airflow_ratio_after_leak_repair(1.0, 0.05)


def test_mixer_energy_scales_with_load_factor():
    # Facility E: 0.55/0.85 = 0.647 vs MEASUR 207,936/320,346 = 0.649
    assert physics.energy_ratio_from_load_factor(0.85, 0.55) == pytest.approx(207936 / 320346, abs=0.003)


def test_scheduling_energy_facility_b():
    # 560 kW x 4 h/day x 330 days = 739,200 kWh (Scenario 3)
    assert physics.shutdown_energy_kwh(560, 4, 330) == 739200


def test_mixer_reynolds_and_min_speed():
    n_min = physics.mixer_min_speed_rps(1200, 0.4, 0.1)
    assert physics.mixer_reynolds(1200, n_min, 0.4, 0.1) == pytest.approx(1e4)


def test_pressure_conversion():
    assert physics.psi_to_bar(89 - 72.6) == pytest.approx(1.13, abs=0.01)
    assert physics.bar_to_psi(physics.psi_to_bar(10.0)) == pytest.approx(10.0)
