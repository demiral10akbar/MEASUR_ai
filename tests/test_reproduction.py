"""The recomputed KPIs must match the values reported in the manuscript."""
import pytest

from rotaeq.analysis import consistency_report, cross_system_summary, scenario_kpis
from rotaeq.sensitivity import emission_factor_comparison, max_tolerable_change


@pytest.fixture(scope="module")
def kpis():
    return scenario_kpis()


def test_all_scenarios_within_tolerance(kpis):
    rep = consistency_report(kpis)
    for col in ("savings_ok", "cost_ok", "payback_ok"):
        bad = rep[rep[col] == False]  # noqa: E712
        assert bad.empty, f"{col} failed for {list(bad['scenario'])}"


def test_mean_saving_26_percent(kpis):
    # Section 5.1: mean of the facility maxima (A 19.5, B 28.5, C 28, D 19, E 35.1) = 26.0 %
    best = kpis.groupby("facility")["savings_pct"].max()
    assert best.mean() == pytest.approx(26.0, abs=0.1)


def test_table9_facility_b_range(kpis):
    s = cross_system_summary(kpis).loc["B"]
    assert s["savings_pct_min"] == pytest.approx(16.7, abs=0.05)
    assert s["savings_pct_max"] == pytest.approx(28.5, abs=0.05)


def test_emission_factor_scaling():
    ef = emission_factor_comparison()
    assert ef["factor_etkb_2022_distribution"].between(1.18, 1.185).all()


def test_robustness_thresholds_section_3_5():
    thr = max_tolerable_change().set_index("scenario")
    # Section 3.5: ~15 % price reduction (binding: A and D) and ~17 % cost increase (A)
    assert thr.loc["A-AI", "max_price_reduction_pct"] == pytest.approx(14.8, abs=0.2)
    assert thr.loc["D-AI", "max_price_reduction_pct"] == pytest.approx(15.6, abs=0.2)
    assert thr.loc["A-AI", "max_cost_increase_pct"] == pytest.approx(17.4, abs=0.2)
