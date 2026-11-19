"""Bet selection, Kelly staking with shrinkage, settlement and closing line value."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .implied import implied


def kelly_fraction(p, odds) -> np.ndarray:
    """Full-Kelly stake as a fraction of bankroll for a single binary bet."""
    p, odds = np.asarray(p, float), np.asarray(odds, float)
    b = odds - 1.0
    return np.clip((p * b - (1 - p)) / b, 0, None)


def shrunk_kelly(p, odds, p_market, shrink=0.5) -> np.ndarray:
    """Kelly on a probability shrunk toward the market: p* = s*p + (1-s)*p_market.

    Shrinking toward the market is equivalent to fractional Kelly and is the
    practical version of Baker & McHale (2013): estimated edges are noisy, so
    full Kelly over-bets. Choose `shrink` from the simulation study.
    """
    p_star = shrink * np.asarray(p) + (1 - shrink) * np.asarray(p_market)
    return kelly_fraction(p_star, odds)


def select_bets(df, p_model, odds_cols=("open_h", "open_d", "open_a"),
                close_cols=("close_h", "close_d", "close_a"),
                threshold=0.03, shrink=0.5, max_stake=0.02, method="shin") -> pd.DataFrame:
    """One row per (match, outcome) bet with EV above `threshold`.

    Bets are priced at the opening odds; closing odds are used only to
    measure closing line value (CLV) afterwards.
    """
    odds = df[list(odds_cols)].to_numpy(float)
    p_mkt = implied(odds, method)
    p_model = np.asarray(p_model, float)
    ev = p_model * odds - 1
    stake = np.minimum(shrunk_kelly(p_model, odds, p_mkt, shrink), max_stake)
    close = df[list(close_cols)].to_numpy(float)
    fair_close = 1 / implied(close, method)
    res_idx = df["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()

    rows = []
    for j, k in enumerate("HDA"):
        m = (ev[:, j] > threshold) & np.isfinite(ev[:, j]) & (stake[:, j] > 0)
        sub = pd.DataFrame({
            "date": df["date"].to_numpy()[m], "league": df["league"].to_numpy()[m],
            "home": df["home"].to_numpy()[m], "away": df["away"].to_numpy()[m],
            "pick": k, "odds": odds[m, j], "p_model": p_model[m, j], "p_market": p_mkt[m, j],
            "ev": ev[m, j], "stake_frac": stake[m, j],
            "clv": odds[m, j] / fair_close[m, j] - 1,
            "won": res_idx[m] == j,
        })
        rows.append(sub)
    bets = pd.concat(rows, ignore_index=True).sort_values("date").reset_index(drop=True)
    bets["ret_per_unit"] = np.where(bets.won, bets.odds - 1, -1.0)
    return bets


def run_bankroll(bets: pd.DataFrame, start=1.0) -> pd.DataFrame:
    """Compound stakes day by day (stakes sized on the bankroll at start of day)."""
    bank = start
    out = []
    for date, day in bets.groupby("date", sort=True):
        pnl = (day.stake_frac * bank * day.ret_per_unit).sum()
        bank += pnl
        out.append({"date": date, "bankroll": bank, "n_bets": len(day)})
    return pd.DataFrame(out)


def betting_summary(bets: pd.DataFrame, n_boot=2000, seed=0) -> dict:
    if bets.empty:
        return {"n_bets": 0}
    rng = np.random.default_rng(seed)
    r, c = bets.ret_per_unit.to_numpy(), bets.clv.to_numpy()
    boot = rng.choice(r, (n_boot, len(r))).mean(1)
    cboot = rng.choice(c, (n_boot, len(c))).mean(1)
    return {
        "n_bets": len(bets), "roi_flat": r.mean(),
        "roi_ci": tuple(np.percentile(boot, [2.5, 97.5])),
        "mean_clv": c.mean(), "clv_ci": tuple(np.percentile(cboot, [2.5, 97.5])),
        "share_beating_close": (c > 0).mean(),
    }
