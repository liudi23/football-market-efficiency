"""Dixon & Coles (1997) bivariate goals model with time-decay weights.

log(lambda_home) = mu + home + att[h] - def[a]
log(lambda_away) = mu        + att[a] - def[h]
with a low-score correction tau(rho) and sum-to-zero constraints on att/def.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson


def decay_weights(dates, ref_date, xi=0.0019) -> np.ndarray:
    """exp(-xi * days_before_ref). xi=0.0019/day is roughly a 1-year half-life."""
    days = (pd.Timestamp(ref_date) - pd.to_datetime(dates)).dt.days.to_numpy()
    return np.exp(-xi * np.clip(days, 0, None))


def _log_tau(x, y, lh, la, rho):
    t = np.ones_like(lh)
    t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return np.log(np.clip(t, 1e-10, None))


class DixonColes:
    def __init__(self, xi=0.0019, max_goals=10):
        self.xi, self.max_goals = xi, max_goals
        self._x0 = None

    # -- parameter packing: att/def for n-1 teams, last = -sum(others)
    def _unpack(self, th):
        n = self.n_
        att = np.append(th[: n - 1], -th[: n - 1].sum())
        dfn = np.append(th[n - 1: 2 * n - 2], -th[n - 1: 2 * n - 2].sum())
        mu, home, rho = th[2 * n - 2:]
        return att, dfn, mu, home, rho

    def fit(self, df: pd.DataFrame, ref_date=None):
        teams = sorted(set(df.home) | set(df.away))
        self.teams_ = {t: i for i, t in enumerate(teams)}
        self.n_ = n = len(teams)
        h = df.home.map(self.teams_).to_numpy()
        a = df.away.map(self.teams_).to_numpy()
        x, y = df.hg.to_numpy(), df.ag.to_numpy()
        ref = ref_date if ref_date is not None else df.date.max()
        w = decay_weights(df.date, ref, self.xi) if self.xi else np.ones(len(df))
        const = gammaln(x + 1) + gammaln(y + 1)

        def nll(th):
            att, dfn, mu, home, rho = self._unpack(th)
            llh = mu + home + att[h] - dfn[a]
            lla = mu + att[a] - dfn[h]
            lh, la = np.exp(llh), np.exp(lla)
            ll = x * llh - lh + y * lla - la - const + _log_tau(x, y, lh, la, rho)
            return -(w * ll).sum()

        x0 = np.zeros(2 * n + 1)
        x0[-3:] = [0.1, 0.25, -0.05]
        if self._x0 is not None and len(self._x0) == len(x0):
            x0 = self._x0
        bounds = [(None, None)] * (2 * n) + [(-0.3, 0.3)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        self._x0 = res.x
        self.att_, self.def_, self.mu_, self.home_, self.rho_ = self._unpack(res.x)
        self.converged_ = res.success
        return self

    def rates(self, home, away):
        """Expected goals for each fixture; unseen teams get average strength (0)."""
        ih = np.array([self.teams_.get(t, -1) for t in home])
        ia = np.array([self.teams_.get(t, -1) for t in away])
        att = np.append(self.att_, 0.0)  # index -1 -> neutral
        dfn = np.append(self.def_, 0.0)
        lh = np.exp(self.mu_ + self.home_ + att[ih] - dfn[ia])
        la = np.exp(self.mu_ + att[ia] - dfn[ih])
        return lh, la

    def score_grid(self, lh, la):
        return score_grid(lh, la, self.rho_, self.max_goals)

    def predict_proba(self, home, away) -> np.ndarray:
        return grid_to_1x2(self.score_grid(*self.rates(home, away)))

    def predict_over(self, home, away, line=2.5) -> np.ndarray:
        return grid_to_over(self.score_grid(*self.rates(home, away)), line)


def score_grid(lh, la, rho, max_goals=10) -> np.ndarray:
    """Array (n, G, G) of P(home=i, away=j) including the Dixon-Coles tau correction."""
    lh, la = np.atleast_1d(np.asarray(lh, float)), np.atleast_1d(np.asarray(la, float))
    g = np.arange(max_goals + 1)
    grid = poisson.pmf(g[None, :, None], lh[:, None, None]) * \
        poisson.pmf(g[None, None, :], la[:, None, None])
    grid[:, 0, 0] *= 1 - lh * la * rho
    grid[:, 0, 1] *= 1 + lh * rho
    grid[:, 1, 0] *= 1 + la * rho
    grid[:, 1, 1] *= 1 - rho
    return grid / grid.sum((1, 2), keepdims=True)


def grid_to_1x2(grid) -> np.ndarray:
    tri = np.subtract.outer(np.arange(grid.shape[1]), np.arange(grid.shape[2]))
    return np.column_stack([grid[:, tri > 0].sum(1), grid[:, tri == 0].sum(1),
                            grid[:, tri < 0].sum(1)])


def grid_to_over(grid, line=2.5) -> np.ndarray:
    g = np.arange(grid.shape[1])
    return grid[:, (g[:, None] + g[None, :]) > line].sum(1)
