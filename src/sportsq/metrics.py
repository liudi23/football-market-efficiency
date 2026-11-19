"""Proper scoring rules and calibration for 1X2 forecasts.

Probabilities are arrays of shape (n, 3) ordered home/draw/away; outcomes are
integer codes 0=H, 1=D, 2=A (see `encode`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CODES = {"H": 0, "D": 1, "A": 2}


def encode(results) -> np.ndarray:
    return np.array([CODES[r] for r in results])


def one_hot(y, k=3) -> np.ndarray:
    return np.eye(k)[np.asarray(y)]


def rps(p, y) -> np.ndarray:
    """Ranked probability score per match (lower is better). Respects H<D<A ordering."""
    p = np.asarray(p, float)
    o = one_hot(y, p.shape[1])
    cp, co = np.cumsum(p, 1)[:, :-1], np.cumsum(o, 1)[:, :-1]
    return ((cp - co) ** 2).sum(1) / (p.shape[1] - 1)


def log_loss(p, y, eps=1e-12) -> np.ndarray:
    p = np.asarray(p, float)
    return -np.log(np.clip(p[np.arange(len(p)), np.asarray(y)], eps, 1))


def brier(p, y) -> np.ndarray:
    p = np.asarray(p, float)
    return ((p - one_hot(y, p.shape[1])) ** 2).sum(1)


def summary(p, y) -> dict:
    return {"rps": rps(p, y).mean(), "log_loss": log_loss(p, y).mean(),
            "brier": brier(p, y).mean(), "n": len(y)}


def paired_bootstrap(loss_a, loss_b, groups=None, n_boot=2000, seed=0) -> dict:
    """Mean of (loss_a - loss_b) with a block bootstrap CI.

    `groups` (e.g. matchweek/date) are resampled as blocks because matches on
    the same weekend share information. Negative mean => model A is better.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(d))
    g = pd.Series(d).groupby(np.asarray(groups))
    sums, counts = g.sum().values, g.size().values
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": d.mean(), "ci_low": lo, "ci_high": hi,
            "p_a_better": (boots < 0).mean()}


def calibration_table(p, y, bins=10) -> pd.DataFrame:
    """Reliability table pooled over all three outcomes."""
    p = np.asarray(p, float).ravel()
    o = one_hot(y, 3).ravel()
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "o": o})
    t = df.groupby("bin").agg(pred=("p", "mean"), obs=("o", "mean"), n=("o", "size"))
    return t


def ece(p, y, bins=10) -> float:
    t = calibration_table(p, y, bins)
    return float((t.n * (t.pred - t.obs).abs()).sum() / t.n.sum())
