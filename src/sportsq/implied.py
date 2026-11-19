"""Turn bookmaker odds into probabilities by removing the margin (overround).

All functions take an array of decimal odds with outcomes on the last axis
(e.g. shape (n_matches, 3) for home/draw/away) and return probabilities that
sum to 1 along that axis.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def overround(odds) -> np.ndarray:
    """Bookmaker margin: sum of 1/odds minus 1."""
    return (1.0 / np.asarray(odds, float)).sum(-1) - 1.0


def multiplicative(odds) -> np.ndarray:
    q = 1.0 / np.asarray(odds, float)
    return q / q.sum(-1, keepdims=True)


def _rowwise(fn, odds):
    odds = np.asarray(odds, float)
    if odds.ndim == 1:
        return fn(odds)
    out = np.full_like(odds, np.nan)
    ok = np.isfinite(odds).all(-1) & (odds > 1).all(-1)
    out[ok] = np.apply_along_axis(fn, -1, odds[ok])
    return out


def _power_row(o):
    q = 1.0 / o
    if q.sum() <= 1.0:
        return q / q.sum()
    k = brentq(lambda k: (q ** k).sum() - 1.0, 1.0, 10.0)
    return q ** k


def power(odds) -> np.ndarray:
    """p_i = q_i^k with k chosen so probabilities sum to 1 (shrinks longshots more)."""
    return _rowwise(_power_row, odds)


def _shin_row(o):
    q = 1.0 / o
    b = q.sum()
    if b <= 1.0:
        return q / b

    def probs(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / b) - z) / (2 * (1 - z))

    z = brentq(lambda z: probs(z).sum() - 1.0, 0.0, 0.4)
    return probs(z)


def shin(odds) -> np.ndarray:
    """Shin (1992/93) insider-trading model; standard for the favourite-longshot bias."""
    return _rowwise(_shin_row, odds)


METHODS = {"multiplicative": multiplicative, "power": power, "shin": shin}


def implied(odds, method: str = "shin") -> np.ndarray:
    return METHODS[method](odds)
