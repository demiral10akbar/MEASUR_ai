"""Schema-checked loading of field-measurement CSV files (see data/field/README.md)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SCHEMAS: dict[str, list[str]] = {
    "compressor_log": ["timestamp", "power_kw", "flow_m3_min", "pressure_bar", "state"],
    "fan_log": ["timestamp", "power_kw", "speed_rpm", "flow_m3_s", "dp_pa", "outdoor_temp_c", "production_on"],
    "pump_condition": ["timestamp", "pump_id", "vib_rms_mm_s", "vib_1x_mm_s", "vib_2x_mm_s", "current_a", "dp_bar", "label"],
    "mixer_batch": ["timestamp", "batch_id", "phase", "current_a", "speed_rpm", "torque_nm"],
    "leak_survey": ["survey_date", "leak_id", "location", "flow_l_min", "repaired"],
}
OPTIONAL = {"pump_condition": {"label"}, "mixer_batch": {"torque_nm"}}
TIME_COLUMNS = {"timestamp", "survey_date"}


class SchemaError(ValueError):
    pass


def load_field_csv(path: str | Path, kind: str | None = None) -> pd.DataFrame:
    """Load a field CSV and check that it contains the columns of its schema.

    kind defaults to the file stem (e.g. 'compressor_log.csv' -> 'compressor_log').
    Time columns are parsed and the frame is sorted chronologically.
    """
    path = Path(path)
    kind = kind or path.stem
    if kind not in SCHEMAS:
        raise SchemaError(f"unknown data kind '{kind}'; expected one of {sorted(SCHEMAS)}")
    df = pd.read_csv(path)
    required = set(SCHEMAS[kind]) - OPTIONAL.get(kind, set())
    missing = required - set(df.columns)
    if missing:
        raise SchemaError(f"{path.name}: missing columns {sorted(missing)}")
    for col in TIME_COLUMNS & set(df.columns):
        df[col] = pd.to_datetime(df[col], errors="raise")
        df = df.sort_values(col).reset_index(drop=True)
    return df


def leakage_fraction_from_log(df: pd.DataFrame) -> float:
    """Estimate the leakage fraction from a compressor log recorded during a
    non-production period (all end uses off), using the load/unload method:
    L = t_load / (t_load + t_unload). Only rows with state in {'load', 'unload'} are used.
    """
    states = df["state"].str.lower()
    t_load = (states == "load").sum()
    t_unload = (states == "unload").sum()
    if t_load + t_unload == 0:
        raise ValueError("log contains no load/unload records")
    return t_load / (t_load + t_unload)
