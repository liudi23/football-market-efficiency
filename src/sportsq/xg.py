"""Expected-goals (xG) team ratings for pre-match pricing.

Goals are a noisy measure of how well a team played; xG (the summed scoring
chance of its shots) is less noisy.  `RateModel` fits the same attack/defence
structure as Dixon-Coles to a continuous target

    target = (1 - alpha) * goals + alpha * xG,

with a time-decayed Poisson quasi-likelihood (valid for non-integer targets)
and an analytic gradient.  alpha = 0 is the goals model, alpha = 1 pure xG.
Prices use the Dixon-Coles score grid with rho taken from the goals fit (rho
describes final scores, which xG cannot inform).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .dixon_coles import DixonColes, decay_weights, grid_to_1x2, grid_to_over, score_grid


class RateModel:
    def __init__(self, xi=0.0019):
        self.xi = xi
        self._x0 = None

    def _unpack(self, th):
        n = self.n_
        att = np.append(th[: n - 1], -th[: n - 1].sum())
        dfn = np.append(th[n - 1: 2 * n - 2], -th[n - 1: 2 * n - 2].sum())
        return att, dfn, th[-2], th[-1]

    def fit(self, home, away, th_target, ta_target, dates, ref_date):
        teams = sorted(set(home) | set(away))
        self.teams_ = {t: i for i, t in enumerate(teams)}
        n = self.n_ = len(teams)
        h = pd.Series(home).map(self.teams_).to_numpy()
        a = pd.Series(away).map(self.teams_).to_numpy()
        x, y = np.asarray(th_target, float), np.asarray(ta_target, float)
        w = decay_weights(pd.Series(dates), ref_date, self.xi) if self.xi else np.ones(len(x))
        f = self._objective(h, a, x, y, w)
        x0 = np.zeros(2 * n)
        x0[-2:] = [0.1, 0.25]
        if self._x0 is not None and len(self._x0) == len(x0):
            x0 = self._x0
        res = minimize(f, x0, jac=True, method="L-BFGS-B")
        self._x0 = res.x
        self.att_, self.def_, self.mu_, self.home_ = self._unpack(res.x)
        return self

    def _objective(self, h, a, x, y, w):
        """Weighted Poisson quasi-likelihood and its analytic gradient."""
        n = self.n_
        P = np.vstack([np.eye(n - 1), -np.ones((1, n - 1))])   # free -> sum-to-zero

        def f(th):
            att, dfn, mu, hm = self._unpack(th)
            llh = mu + hm + att[h] - dfn[a]
            lla = mu + att[a] - dfn[h]
            lh, la = np.exp(llh), np.exp(lla)
            nll = -(w * (x * llh - lh + y * lla - la)).sum()
            rh, ra = w * (x - lh), w * (y - la)          # d ll / d log-rate
            g_att = np.bincount(h, rh, n) + np.bincount(a, ra, n)
            g_def = -np.bincount(a, rh, n) - np.bincount(h, ra, n)
            grad = np.concatenate([P.T @ g_att, P.T @ g_def, [rh.sum() + ra.sum(), rh.sum()]])
            return nll, -grad
        return f

    def rates(self, home, away):
        ih = np.array([self.teams_.get(t, -1) for t in home])
        ia = np.array([self.teams_.get(t, -1) for t in away])
        att, dfn = np.append(self.att_, 0.0), np.append(self.def_, 0.0)
        return (np.exp(self.mu_ + self.home_ + att[ih] - dfn[ia]),
                np.exp(self.mu_ + att[ia] - dfn[ih]))


def walk_forward_xg(df: pd.DataFrame, start_date, alphas=(0.5, 0.75, 1.0),
                    refit_days=7, window_days=730, xi=0.0019) -> pd.DataFrame:
    """Out-of-sample probabilities for one league.  Columns dc_* (goals
    Dixon-Coles) and a{alpha}_* for each xG blend.  df needs date, home, away,
    hg, ag, hxg, axg."""
    d = df.sort_values("date").reset_index(drop=True)
    dc = DixonColes(xi=xi)
    rms = {al: RateModel(xi=xi) for al in alphas}
    out = []
    t, end = pd.Timestamp(start_date), d.date.max()
    while t <= end:
        t_next = t + pd.Timedelta(days=refit_days)
        test = d[(d.date >= t) & (d.date < t_next)]
        if len(test):
            tr = d[(d.date < t) & (d.date >= t - pd.Timedelta(days=window_days))]
            dc.fit(tr, ref_date=t)
            rho = dc.rho_
            g = dc.score_grid(*dc.rates(test.home, test.away))
            p = grid_to_1x2(g)
            row = {"idx": test.index, "dc_h": p[:, 0], "dc_d": p[:, 1], "dc_a": p[:, 2],
                   "dc_over": grid_to_over(g, 2.5), "rho": rho}
            for al, m in rms.items():
                m.fit(tr.home, tr.away, (1 - al) * tr.hg + al * tr.hxg,
                      (1 - al) * tr.ag + al * tr.axg, tr.date, t)
                lh, la = m.rates(test.home, test.away)
                g = score_grid(lh, la, rho)
                p = grid_to_1x2(g)
                k = f"a{al:g}"
                row.update({f"{k}_h": p[:, 0], f"{k}_d": p[:, 1], f"{k}_a": p[:, 2],
                            f"{k}_over": grid_to_over(g, 2.5), f"{k}_lh": lh, f"{k}_la": la})
            out.append(pd.DataFrame(row))
        t = t_next
    return d.join(pd.concat(out).set_index("idx"), how="inner")


def match_understat(fd: pd.DataFrame, us: pd.DataFrame, min_votes=3):
    """Attach Understat xG to football-data rows without a hand-made name table.
    Candidate pairs share league, date (+/- 1 day) and the exact score; each
    Understat name is mapped to the football-data name it co-occurs with most.
    Returns (fd with hxg/axg, name map, n unmatched fd rows)."""
    us = us.copy()
    us["date"] = pd.to_datetime(us.date).dt.normalize()
    cand = []
    for shift in (-1, 0, 1):
        u = us.assign(date=us.date + pd.Timedelta(days=shift))
        cand.append(fd.merge(u, left_on=["league", "date", "hg", "ag"],
                             right_on=["league", "date", "hg_us", "ag_us"]))
    c = pd.concat(cand)
    votes = pd.concat([c[["league", "home_us", "home"]].set_axis(["league", "us", "fd"], axis=1),
                       c[["league", "away_us", "away"]].set_axis(["league", "us", "fd"], axis=1)])
    v = votes.value_counts().reset_index(name="n")
    v = v[v.n >= min_votes].sort_values("n", ascending=False).drop_duplicates(["league", "us"])
    name = {(r.league, r.us): r.fd for r in v.itertuples()}
    us["home"] = [name.get((l, t)) for l, t in zip(us.league, us.home_us)]
    us["away"] = [name.get((l, t)) for l, t in zip(us.league, us.away_us)]
    us = us.dropna(subset=["home", "away"])
    out = []
    for shift in (0, -1, 1):
        u = us.assign(date=us.date + pd.Timedelta(days=shift))[["league", "date", "home", "away", "hxg", "axg"]]
        out.append(fd.merge(u, on=["league", "date", "home", "away"], how="left"))
    m = out[0]
    for o in out[1:]:
        m["hxg"] = m.hxg.fillna(o.hxg); m["axg"] = m.axg.fillna(o.axg)
    return m, name, int(m.hxg.isna().sum())
