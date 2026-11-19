"""Sequential team ratings (Elo, pi-ratings) and an ordered-logit map to 1X2.

Ratings are computed in one chronological pass. The feature for each match is
the *pre-match* rating difference, so there is no lookahead by construction.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


class Elo:
    """Football Elo with home advantage and a goal-difference multiplier."""

    def __init__(self, k=20.0, home_adv=60.0, init=1500.0, regress=0.2):
        self.k, self.home_adv, self.init, self.regress = k, home_adv, init, regress

    def features(self, df: pd.DataFrame) -> pd.Series:
        r = defaultdict(lambda: self.init)
        season = None
        out = np.empty(len(df))
        for i, row in enumerate(df.itertuples(index=False)):
            if row.season != season:  # pull ratings toward the mean between seasons
                season = row.season
                if r:
                    m = np.mean(list(r.values()))
                    for t in r:
                        r[t] = m + (1 - self.regress) * (r[t] - m)
            diff = r[row.home] + self.home_adv - r[row.away]
            out[i] = diff
            exp_h = 1.0 / (1.0 + 10 ** (-diff / 400))
            act = 1.0 if row.hg > row.ag else 0.5 if row.hg == row.ag else 0.0
            mult = np.log1p(abs(row.hg - row.ag)) + 1.0
            delta = self.k * mult * (act - exp_h)
            r[row.home] += delta
            r[row.away] -= delta
        return pd.Series(out, index=df.index, name="elo_diff")


class PiRatings:
    """Constantinou & Fenton (2013) pi-ratings: separate home/away ratings in goal units."""

    def __init__(self, lam=0.035, gamma=0.7, b=10.0, c=3.0):
        self.lam, self.gamma, self.b, self.c = lam, gamma, b, c

    def _g(self, r):
        return np.sign(r) * (self.b ** (abs(r) / self.c) - 1)

    def features(self, df: pd.DataFrame) -> pd.Series:
        home_r = defaultdict(float)
        away_r = defaultdict(float)
        out = np.empty(len(df))
        for i, row in enumerate(df.itertuples(index=False)):
            pred = self._g(home_r[row.home]) - self._g(away_r[row.away])
            out[i] = pred
            err = (row.hg - row.ag) - pred
            psi = self.c * np.log10(1 + abs(err)) * np.sign(err)
            dh, da = psi * self.lam, -psi * self.lam
            home_r[row.home] += dh
            away_r[row.home] += dh * self.gamma
            away_r[row.away] += da
            home_r[row.away] += da * self.gamma
        return pd.Series(out, index=df.index, name="pi_gd")


class OrderedLogit:
    """P(A) = s(c1 - b x), P(H) = 1 - s(c2 - b x), with c1 < c2 (A < D < H on x)."""

    def fit(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y)  # 0=H,1=D,2=A

        def nll(theta):
            b, c1, log_gap = theta
            c2 = c1 + np.exp(log_gap)
            p = self._probs(x, b, c1, c2)
            return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).sum()

        scale = np.std(x) or 1.0
        res = minimize(nll, x0=[1.0 / scale, -0.5, 0.0], method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
        b, c1, lg = res.x
        self.b_, self.c1_, self.c2_ = b, c1, c1 + np.exp(lg)
        return self

    @staticmethod
    def _probs(x, b, c1, c2):
        pa = expit(c1 - b * x)
        pa_or_d = expit(c2 - b * x)
        return np.column_stack([1 - pa_or_d, pa_or_d - pa, pa])

    def predict_proba(self, x):
        return self._probs(np.asarray(x, float), self.b_, self.c1_, self.c2_)
