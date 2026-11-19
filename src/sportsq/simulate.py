"""Synthetic leagues and betting markets where the truth is known.

Used for injection-recovery tests: generate a league from a dynamic
Dixon-Coles model, build a noisy market around the true probabilities,
optionally inject a known inefficiency, then check whether the pipeline
recovers the parameters and detects (only) the edge that was injected.

The output frame has the same schema as `data.load`, plus the true
probabilities `true_h, true_d, true_a, true_over`.

Market model
------------
closing : softmax(log p_true + N(0, close_noise))              (sharp, near truth)
opening : softmax(log p_lagged + N(0, open_noise))             (noisier)
    p_lagged uses team strengths from `market_lag` matchweeks ago, so a
    market_lag > 0 injects a "slow market" edge that a model tracking recent
    form can exploit. market_lag=0 and small open_noise is the null world.
flb     : p_market ∝ p^flb (flb < 1 overprices longshots)
margin  : odds = 1 / (p_market * (1 + margin))
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dixon_coles import grid_to_1x2, grid_to_over, score_grid


@dataclass
class SimConfig:
    n_teams: int = 20
    n_seasons: int = 6
    first_season: int = 2015
    mu: float = 0.18            # log baseline away rate (~1.2 goals)
    home: float = 0.25
    rho: float = -0.08
    strength_sd: float = 0.30   # initial spread of att/def
    drift_sd: float = 0.03      # weekly random-walk sd of att/def
    mean_revert: float = 0.01   # weekly pull back toward 0
    close_noise: float = 0.03
    open_noise: float = 0.08
    market_lag: int = 0         # matchweeks of staleness in opening prices
    flb: float = 1.0
    margin: float = 0.025
    seed: int = 0


def _round_robin(n):
    teams = list(range(n))
    rounds = []
    for r in range(n - 1):
        pairs = [(teams[i], teams[n - 1 - i]) for i in range(n // 2)]
        rounds.append([(a, b) if r % 2 == 0 else (b, a) for a, b in pairs])
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds + [[(b, a) for a, b in rd] for rd in rounds]


def _softmax_log(logp, rng, sd):
    z = logp + rng.normal(0, sd, logp.shape) if sd else logp
    z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def simulate(cfg: SimConfig | None = None, **kw) -> pd.DataFrame:
    cfg = cfg or SimConfig(**kw)
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_teams
    names = [f"Team{i:02d}" for i in range(n)]
    att = rng.normal(0, cfg.strength_sd, n)
    dfn = rng.normal(0, cfg.strength_sd, n)
    history = []  # (att, def) snapshots per matchweek for the lagged market
    rows = []
    sched = _round_robin(n)
    for s in range(cfg.n_seasons):
        year = cfg.first_season + s
        start = pd.Timestamp(f"{year}-08-10")
        for wk, fixtures in enumerate(sched):
            date = start + pd.Timedelta(weeks=wk)
            history.append((att.copy(), dfn.copy()))
            lag_att, lag_def = history[max(0, len(history) - 1 - cfg.market_lag)]
            h = np.array([f[0] for f in fixtures])
            a = np.array([f[1] for f in fixtures])

            def rates(at, de):
                ac, dc = at - at.mean(), de - de.mean()
                return (np.exp(cfg.mu + cfg.home + ac[h] - dc[a]),
                        np.exp(cfg.mu + ac[a] - dc[h]))

            lh, la = rates(att, dfn)
            grid = score_grid(lh, la, cfg.rho)
            p_true = grid_to_1x2(grid)
            p_over = grid_to_over(grid)
            # sample scores from the true grid
            flat = grid.reshape(len(h), -1)
            idx = np.array([rng.choice(flat.shape[1], p=f) for f in flat])
            hg, ag = np.divmod(idx, grid.shape[2])

            p_lag = grid_to_1x2(score_grid(*rates(lag_att, lag_def), cfg.rho))
            p_close = _softmax_log(np.log(p_true), rng, cfg.close_noise)
            p_open = _softmax_log(np.log(p_lag), rng, cfg.open_noise)
            if cfg.flb != 1.0:
                p_close = p_close ** cfg.flb / (p_close ** cfg.flb).sum(1, keepdims=True)
                p_open = p_open ** cfg.flb / (p_open ** cfg.flb).sum(1, keepdims=True)
            po = np.column_stack([p_over, 1 - p_over])
            po_close = _softmax_log(np.log(po), rng, cfg.close_noise)
            po_open = _softmax_log(np.log(po), rng, cfg.open_noise)

            for i in range(len(h)):
                rows.append({
                    "date": date, "season": f"{year % 100:02d}{(year + 1) % 100:02d}",
                    "league": "SIM", "home": names[h[i]], "away": names[a[i]],
                    "hg": int(hg[i]), "ag": int(ag[i]),
                    "hs": rng.poisson(lh[i] * 9), "as_": rng.poisson(la[i] * 9),
                    "hst": rng.poisson(lh[i] * 3.5), "ast": rng.poisson(la[i] * 3.5),
                    **{f"open_{k}": 1 / (p_open[i, j] * (1 + cfg.margin)) for j, k in enumerate("hda")},
                    **{f"close_{k}": 1 / (p_close[i, j] * (1 + cfg.margin)) for j, k in enumerate("hda")},
                    "ou_over": 1 / (po_open[i, 0] * (1 + cfg.margin)),
                    "ou_under": 1 / (po_open[i, 1] * (1 + cfg.margin)),
                    "ou_close_over": 1 / (po_close[i, 0] * (1 + cfg.margin)),
                    "ou_close_under": 1 / (po_close[i, 1] * (1 + cfg.margin)),
                    "true_h": p_true[i, 0], "true_d": p_true[i, 1], "true_a": p_true[i, 2],
                    "true_over": p_over[i],
                    "true_att_h": att[h[i]] - att.mean(), "true_def_h": dfn[h[i]] - dfn.mean(),
                })
            # strengths evolve after the matchweek
            att = (1 - cfg.mean_revert) * att + rng.normal(0, cfg.drift_sd, n)
            dfn = (1 - cfg.mean_revert) * dfn + rng.normal(0, cfg.drift_sd, n)
    df = pd.DataFrame(rows)
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], ["H", "D"], "A")
    return df
