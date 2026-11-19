"""Stages 1-3 baseline on real data (or --sim to exercise the pipeline).

1. Margin removal: which method gives the best-calibrated Pinnacle closing probabilities?
2. Ratings: Elo and pi-ratings mapped to 1X2 with an ordered logit (fit on development seasons).
3. Walk-forward Dixon-Coles, a market blend, and value betting vs opening odds with CLV.

Splits follow docs/methodology.md. The locked holdout is NOT touched here.
Writes reports/baseline_real.md (or reports/baseline_sim.md).

Usage:
    python scripts/baseline_real.py --leagues E0 E1 D1 SP1 I1 F1
    python scripts/baseline_real.py --sim
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sportsq import data, metrics
from sportsq.backtest import MarketBlend, walk_forward_dc
from sportsq.implied import METHODS, implied
from sportsq.ratings import Elo, OrderedLogit, PiRatings
from sportsq.simulate import SimConfig, simulate
from sportsq.staking import betting_summary, select_bets

# season start dates for the splits (real data)
DEV_END = "2019-07-01"       # development: 2012/13 - 2018/19
VAL_END = "2023-07-01"       # validation:  2019/20 - 2022/23
#                              holdout:     2023/24 onwards (locked)

OPEN = ["open_h", "open_d", "open_a"]
CLOSE = ["close_h", "close_d", "close_a"]


def table(d: dict) -> str:
    return pd.DataFrame(d).T.round(4).to_markdown()


def main(args):
    if args.sim:
        df = simulate(SimConfig(n_seasons=9, first_season=2014, market_lag=6, seed=11))
        dev_end, val_end, name = "2019-07-01", "2023-07-01", "sim"
    else:
        df = data.load(leagues=args.leagues)
        dev_end, val_end, name = DEV_END, VAL_END, "real"
    df = df[df.date < val_end].reset_index(drop=True)   # never touch the holdout
    out = [f"# Baseline ({name} data)", "",
           f"Leagues: {sorted(df.league.unique())}; matches before {val_end}: {len(df)}", ""]

    # ---- 1. margin removal on closing odds
    c = df.dropna(subset=CLOSE)
    y = metrics.encode(c.result)
    res = {}
    for m in METHODS:
        p = implied(c[CLOSE].to_numpy(), m)
        res[m] = {**metrics.summary(p, y), "ece": metrics.ece(p, y)}
    out += ["## 1. Margin removal (Pinnacle closing odds)", "", table(res), "",
            f"Mean closing overround: {(1 / c[CLOSE]).sum(axis=1).sub(1).mean():.4f}", ""]

    # ---- 2. ratings -> ordered logit, fit on dev, scored on validation
    feats = {"elo": Elo().features(df), "pi": PiRatings().features(df)}
    dev = (df.date < dev_end) & (df.date >= df.date.min() + pd.Timedelta(days=365))  # warm-up
    val = (df.date >= dev_end) & (df.date < val_end)
    yv = metrics.encode(df.result[val])
    res = {}
    for k, x in feats.items():
        ol = OrderedLogit().fit(x[dev], metrics.encode(df.result[dev]))
        res[k] = metrics.summary(ol.predict_proba(x[val]), yv)
    vo = df[val].dropna(subset=OPEN)
    res["market_open"] = metrics.summary(implied(vo[OPEN].to_numpy(), "shin"),
                                         metrics.encode(vo.result))
    vc = df[val].dropna(subset=CLOSE)
    res["market_close"] = metrics.summary(implied(vc[CLOSE].to_numpy(), "shin"),
                                          metrics.encode(vc.result))
    out += ["## 2. Ratings vs market (validation seasons)", "", table(res), ""]

    # ---- 3. walk-forward Dixon-Coles + blend + betting on validation
    wf = walk_forward_dc(df, dev_end, refit_days=7).dropna(subset=OPEN + CLOSE)
    wf = wf[wf.date < val_end]
    y = metrics.encode(wf.result)
    p_open = implied(wf[OPEN].to_numpy(), "shin")
    p_dc = wf[["dc_h", "dc_d", "dc_a"]].to_numpy()
    first = (wf.date < wf.date.quantile(0.5)).to_numpy()
    blend = MarketBlend().fit(p_open[first], p_dc[first], y[first])
    p_bl = blend.predict_proba(p_open, p_dc)
    t = ~first
    res = {"dixon_coles": metrics.summary(p_dc[t], y[t]),
           "market_open": metrics.summary(p_open[t], y[t]),
           "blend": metrics.summary(p_bl[t], y[t])}
    cmp = metrics.paired_bootstrap(metrics.rps(p_bl[t], y[t]), metrics.rps(p_open[t], y[t]),
                                   groups=wf.date[t].to_numpy())
    bets = select_bets(wf[t], p_bl[t], threshold=0.02)
    bs = betting_summary(bets)
    out += ["## 3. Walk-forward Dixon-Coles and market blend (second half of validation)", "",
            table(res), "",
            f"Blend weights: market {blend.w_[0]:.3f}, model {blend.w_[1]:.3f}", "",
            f"Blend minus opening RPS: {cmp['mean_diff']:.5f} "
            f"(95% CI {cmp['ci_low']:.5f} to {cmp['ci_high']:.5f})", "",
            "### Value betting at opening odds", "",
            pd.Series({k: (tuple(round(float(x), 4) for x in v) if isinstance(v, tuple)
                           else round(float(v), 4)) for k, v in bs.items()}
                      ).to_frame("value").to_markdown(), ""]
    if not bets.empty:
        out += ["By league:", "", bets.groupby("league").agg(
            n=("clv", "size"), mean_clv=("clv", "mean"), roi=("ret_per_unit", "mean")
        ).round(4).to_markdown(), ""]
    Path("reports").mkdir(exist_ok=True)
    path = Path(f"reports/baseline_{name}.md")
    path.write_text("\n".join(out))
    print("\n".join(out))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--leagues", nargs="+", default=["E0", "E1", "D1", "SP1", "I1", "F1"])
    main(ap.parse_args())
