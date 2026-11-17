"""Loading of the facility data and the economic/environmental assumptions."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
# Override with the ROTAEQ_DATA environment variable if the package is not
# installed in editable mode from a clone of the repository.
DATA_DIR = Path(os.environ.get("ROTAEQ_DATA", REPO_ROOT / "data"))


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_facilities(path: Path | None = None) -> dict[str, Any]:
    return _load_yaml(path or DATA_DIR / "facilities.yaml")["facilities"]


def load_assumptions(path: Path | None = None) -> dict[str, Any]:
    return _load_yaml(path or DATA_DIR / "assumptions.yaml")


def value(entry: Any) -> Any:
    """Return the numeric value of a flagged entry ({value: x, flag: M}) or a plain number."""
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def flag(entry: Any) -> str | None:
    return entry.get("flag") if isinstance(entry, dict) else None
