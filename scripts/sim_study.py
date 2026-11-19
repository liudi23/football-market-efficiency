"""Stage 0: injection-recovery study on synthetic markets.

Two worlds, several seeds each:
  null  - opening prices are unbiased and nearly as sharp as closing (no edge)
  slow  - opening prices use team strengths 4 matchweeks stale (small injected edge)
  very_slow - 10 matchweeks stale plus noisier opening prices (large injected edge)

For each run: walk-forward Dixon-Coles, a market blend fitted on the first
half of the out-of-sample period and scored on the second half, then
value betting at opening odds. Writes reports/sim_study.md.

Usage:  python scripts/sim_study.py [--seeds 5]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sportsq import metrics
from sportsq.backtest import MarketBlend, walk_forward_dc
from sportsq.implied import implied
from sportsq.simulate import SimConfig, simulate
from sportsq.staking import betting_summary, select_bets

WORLDS = {
    "null": dict(market_lag=0, open_noise=0.03),
    "slow": dict(market_lag=4, open_noise=0.05),
    "very_slow": dict(market_lag=10, open_noise=0.08),
}
START = "2017-08-01"


def run_one(world, seed):
    df = simulate(SimConfig(seed=seed, **WORLDS[world]))
    wf = walk_forward_dc(df, START)
    y = metrics.encode(wf.result)
    p = {
        "truth": wf[["true_h", "true_d", "true_a"]].to_numpy(),
        "close": implied(wf[["close_h", "close_d", "close_a"]].to_numpy(), "shin"),
        "open": implied(wf[["open_h", "open_d", "open_a"]].to_numpy(), "shin"),
        "dixon_coles": wf[["dc_h", "dc_d", "dc_a"]].to_numpy(),
    }
    half = wf.date < wf.date.quantile(0.5)
    blend = MarketBlend().fit(p["open"][half], p["dixon_coles"][half], y[half])
    p["blend"] = blend.predict_proba(p["open"], p["dixon_coles"])

    test = ~half.to_numpy()
    rows = {k: metrics.summary(v[test], y[test]) for k, v in p.items()}
    cmp = metrics.paired_bootstrap(metrics.rps(p["blend"][test], y[test]),
                                   metrics.rps(p["open"][test], y[test]),
                                   groups=wf.date[test].to_numpy())
    bets = select_bets(wf[test], p["blend"][test], threshold=0.02)
    return rows, cmp, betting_summary(bets), blend.w_


def main(seeds):
    lines = ["# Stage 0: simulation study", "",
             f"Worlds: {WORLDS}. Walk-forward Dixon-Coles from {START}; blend fitted on the "
             "first half of out-of-sample matches, everything below scored on the second half.", ""]
    for world in WORLDS:
        recs = []
        for s in range(seeds):
            rows, cmp, bet, w = run_one(world, s)
            rec = {f"rps_{k}": v["rps"] for k, v in rows.items()}
            rec.update({"blend_minus_open": cmp["mean_diff"], "ci_low": cmp["ci_low"],
                        "ci_high": cmp["ci_high"], "w_market": w[0], "w_model": w[1],
                        "n_bets": bet.get("n_bets", 0), "roi": bet.get("roi_flat", np.nan),
                        "mean_clv": bet.get("mean_clv", np.nan),
                        "clv_ci_low": bet.get("clv_ci", (np.nan,))[0]})
            recs.append(rec)
            print(world, s, {k: round(v, 4) for k, v in rec.items()})
        t = pd.DataFrame(recs)
        det_rps = (t.ci_high < 0).mean()
        det_clv = (t.clv_ci_low > 0).mean()
        det_roi = (t.roi > 0).mean()
        lines += [f"## World: {world}", "", t.round(4).to_markdown(index=True), "",
                  f"- Edge detected by forecast score (blend beats opening RPS, 95% CI below 0): "
                  f"**{det_rps:.0%}** of {seeds} seeds",
                  f"- Edge detected by closing line value (CLV 95% CI above 0): **{det_clv:.0%}**",
                  f"- Seeds with positive flat ROI (not a test, shown for contrast): {det_roi:.0%}", ""]
    lines += ["Reading: in the null world a detection rate is the false-positive rate; in the "
              "other worlds it is the power. Ranked probability score: lower is better. "
              "Seeds with 0 bets mean the blend gave the model too little weight to clear "
              "the EV threshold, which is the correct behaviour when there is no edge."]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/sim_study.md").write_text("\n".join(lines))
    print("\nwrote reports/sim_study.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    main(ap.parse_args().seeds)
