"""Demonstrate the AI reference pipelines of Table A1 on SYNTHETIC data.

Usage:  python scripts/demo_ai_modules.py
Writes: results/demo_ai_metrics_SYNTHETIC.csv

The metrics printed here only show that the pipelines run end to end (feature
extraction, chronological / group splits, Adam training with early stopping,
set-point translation). They are NOT model-performance results for the five
facilities of the manuscript and must not be reported as such. To obtain real
metrics, place field data in data/field/ and adapt the loaders below.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

from rotaeq.ai import (chronological_split, fit_mlp_with_early_stopping, group_split,
                       make_demand_forecaster, make_pump_health_classifier, make_torque_regressor,
                       mixer_speed_setpoint, pressure_setpoint_bar, vibration_features)
from rotaeq.ai import synthetic

OUT = Path(__file__).resolve().parents[1] / "results"


def demand_forecasting():
    raw = synthetic.compressed_air_demand(days=365)
    df = synthetic.time_features(raw["timestamp"], raw["demand_m3_min"])
    X, y = df.drop(columns="target").to_numpy(), df["target"].to_numpy()
    tr, va, te = chronological_split(len(df))
    model, hist = fit_mlp_with_early_stopping(make_demand_forecaster(), X[tr], y[tr], X[va], y[va])
    pred = model.predict(X[te])
    setp = pressure_setpoint_bar(pred, demand_capacity_m3_min=12, p_min_bar=6.0, p_max_bar=7.5)
    return {"module": "Demand forecasting -> pressure set-point", "test_MAE": mean_absolute_error(y[te], pred),
            "test_R2": r2_score(y[te], pred), "epochs": len(hist),
            "mean_setpoint_bar": float(setp.mean()), "n_train/val/test": f"{len(tr)}/{len(va)}/{len(te)}"}


def pump_health():
    windows, labels, fs, rpm = synthetic.pump_vibration_windows()
    feats = pd.DataFrame([vibration_features(w, fs, rpm) for w in windows])
    X, y = feats.to_numpy(), labels
    blocks = np.arange(len(y)) // 10                     # time blocks of 10 windows
    tr, va, te = group_split(blocks)
    model, hist = fit_mlp_with_early_stopping(make_pump_health_classifier(), X[tr], y[tr], X[va], y[va],
                                              balance_classes=True)
    pred = model.predict(X[te])
    return {"module": "Pump predictive maintenance (4 states)", "test_accuracy": accuracy_score(y[te], pred),
            "test_macro_F1": f1_score(y[te], pred, average="macro"), "epochs": len(hist),
            "n_train/val/test": f"{len(tr)}/{len(va)}/{len(te)}"}


def mixer_torque():
    df = synthetic.mixer_batches()
    X = df[["phase", "speed_rps", "current_a"]].to_numpy()
    y = df["torque_nm"].to_numpy()
    tr, va, te = group_split(df["batch_id"])
    model, hist = fit_mlp_with_early_stopping(make_torque_regressor(), X[tr], y[tr], X[va], y[va])
    pred = model.predict(X[te])
    n_set = mixer_speed_setpoint(df["viscosity_pa_s"].iloc[te], rho_kg_m3=1200, impeller_d_m=0.4,
                                 rated_speed_rps=3.0)
    return {"module": "Mixer torque prediction -> speed set-point", "test_MAE": mean_absolute_error(y[te], pred),
            "test_R2": r2_score(y[te], pred), "epochs": len(hist),
            "mean_speed_setpoint_rps": float(np.mean(n_set)), "n_train/val/test": f"{len(tr)}/{len(va)}/{len(te)}"}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = [demand_forecasting(), pump_health(), mixer_torque()]
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "demo_ai_metrics_SYNTHETIC.csv", index=False)
    print("SYNTHETIC-DATA DEMONSTRATION - not results for the case-study facilities\n")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
