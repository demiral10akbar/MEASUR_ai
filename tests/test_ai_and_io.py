import numpy as np
import pandas as pd
import pytest

from rotaeq.ai import (chronological_split, fan_speed_setpoint, fit_mlp_with_early_stopping, group_split,
                       make_demand_forecaster, mixer_speed_setpoint, pressure_setpoint_bar, vibration_features)
from rotaeq.io import SchemaError, leakage_fraction_from_log, load_field_csv


def test_chronological_split_no_leakage():
    tr, va, te = chronological_split(100)
    assert (len(tr), len(va), len(te)) == (70, 15, 15)
    assert tr.max() < va.min() and va.max() < te.min()


def test_group_split_keeps_groups_together():
    groups = np.repeat(np.arange(20), 5)
    tr, va, te = group_split(groups)
    sets = [set(groups[i]) for i in (tr, va, te)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(tr) + len(va) + len(te) == len(groups)


def test_vibration_features_recover_1x_amplitude():
    fs, rpm = 2048.0, 2950
    t = np.arange(int(fs)) / fs
    sig = 2.0 * np.sin(2 * np.pi * rpm / 60 * t) + 0.5 * np.sin(2 * np.pi * 2 * rpm / 60 * t)
    f = vibration_features(sig, fs, rpm)
    assert f["amp_1x"] == pytest.approx(2.0, rel=0.05)
    assert f["amp_2x"] == pytest.approx(0.5, rel=0.1)


def test_early_stopping_training_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=400)
    tr, va, te = chronological_split(len(y))
    model, hist = fit_mlp_with_early_stopping(make_demand_forecaster(), X[tr], y[tr], X[va], y[va],
                                              max_epochs=200, patience=10)
    assert len(hist) >= 1
    assert np.mean((model.predict(X[te]) - y[te]) ** 2) < 0.5


def test_setpoints_are_bounded():
    p = pressure_setpoint_bar([0, 6, 20], 12, 6.0, 7.5)
    assert p.min() >= 6.0 and p.max() <= 7.5
    assert fan_speed_setpoint([0.1, 2.0], 1.0).tolist() == [0.3, 1.0]
    assert mixer_speed_setpoint(10.0, 1200, 0.4, 3.0) == pytest.approx(3.0)


def test_field_csv_schema(tmp_path):
    good = tmp_path / "compressor_log.csv"
    pd.DataFrame({"timestamp": ["2026-01-01 00:01", "2026-01-01 00:00"], "power_kw": [50, 20],
                  "flow_m3_min": [8, 0], "pressure_bar": [7.4, 7.5], "state": ["load", "unload"]}).to_csv(good, index=False)
    df = load_field_csv(good)
    assert df["timestamp"].is_monotonic_increasing
    assert leakage_fraction_from_log(df) == pytest.approx(0.5)
    bad = tmp_path / "fan_log.csv"
    pd.DataFrame({"timestamp": ["2026-01-01"], "power_kw": [1]}).to_csv(bad, index=False)
    with pytest.raises(SchemaError):
        load_field_csv(bad)
