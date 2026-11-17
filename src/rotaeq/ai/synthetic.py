"""SYNTHETIC data generators used only to demonstrate and test the AI pipelines.

Nothing generated here represents the five facilities in the manuscript. Results
obtained on these data must not be reported as model performance for the case
studies; train and validate on field data (data/field/) for that purpose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compressed_air_demand(days: int = 365, step_min: int = 15, seed: int = 0) -> pd.DataFrame:
    """Shift-pattern air demand (m3/min) with weekly cycle, breaks and noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=days * 24 * 60 // step_min, freq=f"{step_min}min")
    h = idx.hour + idx.minute / 60
    working = (idx.dayofweek < 5) & (h >= 8) & (h < 18)
    lunch = (h >= 12) & (h < 13)
    base = np.where(working, 9.0, 1.5)              # 1.5 m3/min ~ leakage-like floor
    base = np.where(working & lunch, 4.0, base)
    demand = base + 1.2 * np.sin(2 * np.pi * idx.dayofyear / 365) + rng.normal(0, 0.6, len(idx))
    return pd.DataFrame({"timestamp": idx, "demand_m3_min": np.clip(demand, 0, None)})


def time_features(ts: pd.Series, y: pd.Series, lags=(1, 2, 4, 96)) -> pd.DataFrame:
    """Calendar features plus lagged demand for one-step-ahead forecasting."""
    ts = pd.to_datetime(ts)
    df = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * ts.dt.hour / 24), "hour_cos": np.cos(2 * np.pi * ts.dt.hour / 24),
        "dow": ts.dt.dayofweek, "weekend": (ts.dt.dayofweek >= 5).astype(int),
    })
    for lag in lags:
        df[f"lag_{lag}"] = y.shift(lag)
    df["target"] = y
    return df.dropna().reset_index(drop=True)


def pump_vibration_windows(n_per_class: int = 150, fs_hz: float = 2048.0, rpm: float = 2950,
                           seed: int = 0):
    """1-s vibration windows for four health states with different spectral signatures."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs_hz)) / fs_hz
    f1 = rpm / 60
    states = {"normal": (1.0, 0.2, 0.10), "fouling": (1.4, 0.25, 0.10),
              "bearing_wear": (1.05, 0.35, 0.35), "cavitation": (1.15, 0.25, 0.55)}
    X, y = [], []
    for label, (a1, a2, broad) in states.items():
        for _ in range(n_per_class):
            sig = (a1 * (1 + 0.20 * rng.standard_normal()) * np.sin(2 * np.pi * f1 * t + rng.uniform(0, 6.28))
                   + a2 * np.sin(2 * np.pi * 2 * f1 * t)
                   + broad * rng.standard_normal(t.size) * np.sin(2 * np.pi * 400 * t) ** 2
                   + 0.4 * rng.standard_normal(t.size))
            X.append(sig); y.append(label)
    return np.array(X), np.array(y), fs_hz, rpm


def mixer_batches(n_batches: int = 80, seed: int = 0) -> pd.DataFrame:
    """Batch mixer data: viscosity falls through the batch; torque ~ mu * N (laminar-transitional proxy)."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_batches):
        mu0 = rng.uniform(0.02, 0.06)
        for k in range(60):
            phase = k / 59
            mu = mu0 * (1 - 0.8 * phase) + 0.005
            speed = rng.uniform(0.8, 3.0)               # rev/s
            torque = 800 * mu * speed + 5 * speed ** 2 + rng.normal(0, 0.8)
            current = 20 + 0.35 * torque * speed + rng.normal(0, 0.5)
            rows.append((b, phase, speed, current, torque, mu))
    return pd.DataFrame(rows, columns=["batch_id", "phase", "speed_rps", "current_a", "torque_nm", "viscosity_pa_s"])
