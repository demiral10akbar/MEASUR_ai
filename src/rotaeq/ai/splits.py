"""Data-splitting strategies of Table A1 (no shuffling of time-series data)."""
from __future__ import annotations

import numpy as np


def chronological_split(n: int, train: float = 0.70, val: float = 0.15):
    """Index arrays for a chronological train/validation/test split (default 70/15/15).

    The data must already be sorted by time. Later samples never leak into earlier
    sets, which avoids the optimistic bias of random splits on time series.
    """
    if not 0 < train < 1 or not 0 <= val < 1 or train + val >= 1:
        raise ValueError("invalid split fractions")
    i_tr = int(round(n * train))
    i_va = int(round(n * (train + val)))
    idx = np.arange(n)
    return idx[:i_tr], idx[i_tr:i_va], idx[i_va:]


def group_split(groups, train: float = 0.70, val: float = 0.15, seed: int = 0):
    """Split by whole groups (e.g. batches or pump/time blocks), 70/15/15 by default.

    All samples of one group end up in the same set, so a model is always tested
    on batches or blocks it has never seen.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_tr = int(round(len(uniq) * train))
    n_va = int(round(len(uniq) * val))
    g_tr, g_va, g_te = uniq[:n_tr], uniq[n_tr:n_tr + n_va], uniq[n_tr + n_va:]
    pick = lambda g: np.flatnonzero(np.isin(groups, g))
    return pick(g_tr), pick(g_va), pick(g_te)
