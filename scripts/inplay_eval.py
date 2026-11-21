"""In-play Step 6: evaluate the in-play pricers by minute, against results and the market.

Models (all start from the same pre-match expected goals, implied by the Betfair
market at kick-off, so they differ only in how they update in play):
  M1  constant goal rates over 90 minutes (Step 1)
  M3  time-varying goal rates + random stoppage time (Step 3)
  M4  M3 + predictive score effects (Step 4)
  MKT the Betfair Exchange market (last traded prices, normalised)

At fixed minutes the 1X2 probabilities are scored against final results (RPS,
log loss; over 2.5 by Brier score), with paired block-bootstrap CIs by match,
and compared with the market (mean absolute difference).

Decision rule (docs/inplay-plan.md, revised before any real evaluation data was
seen): keep an added effect if its RPS difference pooled over the in-play minutes
(15'-90') has a 95% match-bootstrap CI below 0 and no single minute is significantly
worse. The original rule ("CI below 0 at most minutes") rejected the *true* model in
the synthetic check, since time effects only matter late in matches.

Data rules: Betfair 2024/25 is the locked holdout and is refused here; validate on
an earlier season (e.g. 2022/23). States within 3 minutes after a goal are skipped
because last-traded prices can still be pre-goal.

Usage:
    python scripts/inplay_eval.py --season 2223
    python scripts/inplay_eval.py --synthetic            # pipeline check on simulated matches
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from sportsq import inplay, metrics

EVAL_POINTS = [(0, 1), (15, 1), (30, 1), (45, 2), (60, 2), (75, 2), (85, 2), (90, 2)]
LOCKED_SEASONS = {"2324", "2425", "2526"}  # 2023/24 onwards: holdout
MODELS = ["M1", "M3", "M4"]


def label(m, p):
    return "HT" if (m, p) == (45, 2) else f"{m}'"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_models(processed="data/processed"):
    d = Path(processed)
    prof = inplay.GoalIntensity.from_dict(json.loads((d / "goal_intensity.json").read_text()))
    eff = inplay.ScoreEffects.from_dict(json.loads((d / "score_effects.json").read_text())["predictive"])
    return prof, eff


def synthetic(n_matches=200, seed=0, noise=0.05, processed="data/processed"):
    """Simulated season in the same schema as betfair_epl_<season>_inplay.csv.

    Truth: Step 3 time shape with random stoppage (no score effects). The "market"
    is the true in-play probability plus logit noise, so a correct pipeline should
    rank M3 close to the market and ahead of M1.
    """
    rng = np.random.default_rng(seed)
    prof, _ = load_models(processed)
    lam_h = np.exp(rng.normal(np.log(1.45), 0.3, n_matches))
    lam_a = np.exp(rng.normal(np.log(1.15), 0.3, n_matches))
    sc = rng.choice(len(prof.probs), n_matches, p=prof.probs)
    bounds = prof.bounds(prof.s1[sc], prof.s2[sc])
    wb = prof.weights

    def rate_fn(u, h, a, idx):
        b = (u[:, None] >= bounds[idx, 1:-1]).sum(1)
        return lam_h[idx] * wb[b], lam_a[idx] * wb[b]

    hg, ag, log = np.zeros(n_matches, int), np.zeros(n_matches, int), []
    inplay._thinning(np.zeros(n_matches), bounds[:, -1], hg, ag, rate_fn,
                     (lam_h + lam_a).max() * wb.max(), rng, log)
    end1 = bounds[:, len(prof.edges1)]
    goals = {i: [] for i in range(n_matches)}
    for i, u, side in log:
        per, minute = (1, u) if u < end1[i] else (2, 45 + (u - end1[i]))
        goals[i].append((per, minute, side))
    rows = []
    for i in range(n_matches):
        s1, s2 = prof.s1[sc[i]], prof.s2[sc[i]]
        grid = [(m, 1) for m in range(0, int(45 + s1) + 1)] + \
               [(m, 2) for m in range(45, int(90 + s2) + 1)]
        for m, p in grid:
            h = sum(1 for g in goals[i] if (g[0], g[1]) <= (p, m) and g[2] == "H")
            a = sum(1 for g in goals[i] if (g[0], g[1]) <= (p, m) and g[2] == "A")
            last = max([g[1] for g in goals[i] if (g[0], g[1]) <= (p, m)], default=None)
            tv = inplay.price_tv(prof, lam_h[i], lam_a[i], h, a, m, p)
            q = np.log(np.clip(tv["1x2"][0], 1e-6, 1)) + rng.normal(0, noise, 3)
            q = np.exp(q - q.max()); q /= q.sum()
            o = np.clip(tv["over"][0][2], 1e-6, 1 - 1e-6)
            o = 1 / (1 + np.exp(-(np.log(o / (1 - o)) + rng.normal(0, noise))))
            rows.append({"date": f"sim{i:03d}", "home": f"H{i}", "away": f"A{i}", "minute": m,
                         "period": p, "hg": h, "ag": a, "mkt_h": q[0], "mkt_d": q[1],
                         "mkt_a": q[2], "mkt_over25": o,
                         "since_goal": (m - last) if last is not None else np.nan,
                         "FTHG": hg[i], "FTAG": ag[i]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------
def _price_match(args):
    """All evaluation states of one match: returns list of dict rows.

    Each model gets its own pre-match rates, fitted so that *its* kick-off 1X2 and
    over-2.5 prices match the market. They then differ only in in-play updating.
    """
    key, states, target, prof, eff = args
    ph, pa, po = target
    lam = {}
    lam["M1"], _ = inplay.fit_prematch_rates(ph, pa, po)

    def tv_fn(lh, la):
        r = inplay.price_tv(prof, lh, la, 0, 0, 0.0, 1)
        return r["1x2"][0][0], r["1x2"][0][2], r["over"][0][2]
    lam["M3"], _ = inplay.fit_rates(tv_fn, ph, pa, po, x0=lam["M1"])

    eff_mean = inplay.ScoreEffects(eff.multipliers, eff.profile.with_mean_stoppage())

    def se_fn(lh, la):
        c = inplay.calibrate(eff_mean, lh, la)
        r = inplay.price_se(eff_mean, lh, la, 0, 0, 0.0, 1, scale=c)
        return r["1x2"][0], r["1x2"][2], r["over"][2]
    lam["M4"], _ = inplay.fit_rates(se_fn, ph, pa, po, x0=lam["M1"])
    scale = inplay.calibrate(eff, *lam["M4"])
    out = []
    for st in states:
        h, a, m, p = st["hg"], st["ag"], st["minute"], st["period"]
        m1 = inplay.price(*lam["M1"], h, a, min(m, 90))
        m3 = inplay.price_tv(prof, *lam["M3"], h, a, m, p)
        m4 = inplay.price_se(eff, *lam["M4"], h, a, m, p, scale=scale)
        out.append({**st, "lam_h": lam["M1"][0], "lam_a": lam["M1"][1],
                    "M1": m1["1x2"][0], "M1_over": m1["over"][0][2],
                    "M3": m3["1x2"][0], "M3_over": m3["over"][0][2],
                    "M4": m4["1x2"], "M4_over": m4["over"][2]})
    return out


def price_states(df, prof, eff, workers=4):
    df = df.copy()
    df["key"] = df.date.astype(str) + "|" + df.home + "|" + df.away
    jobs, skipped = [], {"no kick-off price": 0, "after goal": 0, "missing price": 0}
    fits = []
    for key, g in df.groupby("key", sort=False):
        k0 = g[(g.minute == 0) & (g.period == 1)]
        if not len(k0) or k0[["mkt_h", "mkt_a"]].isna().any(axis=None):
            skipped["no kick-off price"] += 1
            continue
        r0 = k0.iloc[0]
        _, err = inplay.fit_prematch_rates(r0.mkt_h, r0.mkt_a, r0.mkt_over25)
        fits.append(err)
        states = []
        for m, p in EVAL_POINTS:
            row = g[(g.minute == m) & (g.period == p)]
            if not len(row):
                continue
            row = row.iloc[0]
            if row[["mkt_h", "mkt_d", "mkt_a"]].isna().any():
                skipped["missing price"] += 1
                continue
            if pd.notna(row.since_goal) and row.since_goal < 3:
                skipped["after goal"] += 1
                continue
            states.append({"key": key, "minute": int(m), "period": int(p), "hg": int(row.hg),
                           "ag": int(row.ag), "FTHG": int(row.FTHG), "FTAG": int(row.FTAG),
                           "MKT": np.array([row.mkt_h, row.mkt_d, row.mkt_a]),
                           "MKT_over": row.mkt_over25})
        jobs.append((key, states, (r0.mkt_h, r0.mkt_a, r0.mkt_over25), prof, eff))
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for chunk in ex.map(_price_match, jobs) for r in chunk]
    return pd.DataFrame(rows), skipped, np.array(fits)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score(res):
    y = np.where(res.FTHG > res.FTAG, 0, np.where(res.FTHG == res.FTAG, 1, 2))
    over = ((res.FTHG + res.FTAG) > 2.5).astype(float).to_numpy()
    s = {}
    for name in ["MKT"] + MODELS:
        P = np.vstack(res[name].to_numpy())
        P = P / P.sum(1, keepdims=True)
        s[f"rps_{name}"] = metrics.rps(P, y)
        s[f"ll_{name}"] = metrics.log_loss(np.clip(P, 1e-6, 1), y)
        p_over = res[f"{name}_over"].to_numpy(dtype=float)
        if name == "MKT":
            # once the third goal is in, the over-2.5 market is settled and closed: its
            # last traded price is a stale pre-goal price, so the market's view is 1
            p_over = np.where((res.hg + res.ag).to_numpy() >= 3, 1.0, p_over)
        s[f"brier_over_{name}"] = (p_over - over) ** 2
        if name != "MKT":
            s[f"dist_{name}"] = np.abs(P - np.vstack(res["MKT"].to_numpy())).sum(1) / 2
    return pd.DataFrame(s, index=res.index)


def compare(res, sc, a, b, metric="rps"):
    """Per evaluation point: mean (a - b) with a match-block bootstrap CI."""
    rows = []
    for m, p in EVAL_POINTS:
        sel = (res.minute == m) & (res.period == p)
        if sel.sum() < 10:
            continue
        d = sc.loc[sel, f"{metric}_{a}"].to_numpy() - sc.loc[sel, f"{metric}_{b}"].to_numpy()
        r = metrics.paired_bootstrap(d, np.zeros_like(d), groups=res.loc[sel, "key"].to_numpy())
        rows.append({"minute": label(m, p), "n": int(sel.sum()), "mean": r["mean_diff"],
                     "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    return pd.DataFrame(rows)


def pooled(res, sc, a, b, metric="rps", n_boot=4000, seed=0):
    """RPS difference pooled over the in-play evaluation minutes (excluding kick-off):
    per-match mean of (a - b), bootstrapped over matches."""
    sel = ~((res.minute == 0) & (res.period == 1))
    d = (sc.loc[sel, f"{metric}_{a}"] - sc.loc[sel, f"{metric}_{b}"]).groupby(res.loc[sel, "key"]).mean()
    rng = np.random.default_rng(seed)
    boot = rng.choice(d.to_numpy(), (n_boot, len(d))).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return d.mean(), lo, hi, len(d)


def fmt(df):
    out = df.copy()
    for c in ("mean", "ci_low", "ci_high"):
        out[c] = out[c].map(lambda x: f"{x:+.4f}")
    out["better?"] = np.where(df.ci_high < 0, "yes (CI < 0)",
                              np.where(df.ci_low > 0, "worse (CI > 0)", "no clear difference"))
    return out


def main(season=None, synthetic_run=False, n_matches=200, workers=4, seed=0):
    prof, eff = load_models()
    if synthetic_run:
        df = synthetic(n_matches, seed=seed)
        title, tag = f"synthetic season (pipeline check, seed {seed})", f"synthetic_s{seed}"
    else:
        if season in LOCKED_SEASONS:
            raise SystemExit(f"Season {season} is in the locked holdout (2023/24+). "
                             "Validate on an earlier season.")
        df = pd.read_csv(f"data/processed/betfair_epl_{season}_inplay.csv")
        title, tag = f"Premier League {season[:2]}/{season[2:]} (Betfair Exchange)", season
    res, skipped, fit_err = price_states(df, prof, eff, workers)
    sc = score(res)
    lines = [f"# In-play Step 6: evaluation by minute — {title}", "",
             f"Matches: {res.key.nunique()}; evaluation states: {len(res)}. "
             f"Skipped: {skipped}. Kick-off market fit (constant-rate model): RMS "
             f"probability error {np.mean(fit_err):.4f}.", "",
             "Models: **M1** constant rates (Step 1); **M3** time-varying rates + stoppage "
             "(Step 3); **M4** M3 + predictive score effects (Step 4); **MKT** Betfair "
             "market. Red cards (Step 5) are not modelled.", "",
             "Each model's pre-match rates are fitted so that its own kick-off prices match "
             "the market.", ""]
    # 1. RPS table
    tab = []
    for m, p in EVAL_POINTS:
        sel = (res.minute == m) & (res.period == p)
        if not sel.any():
            continue
        tab.append({"minute": label(m, p), "n": int(sel.sum()),
                    **{f"RPS {k}": round(sc.loc[sel, f"rps_{k}"].mean(), 4) for k in ["MKT"] + MODELS},
                    **{f"Brier o2.5 {k}": round(sc.loc[sel, f"brier_over_{k}"].mean(), 4)
                       for k in ["MKT"] + MODELS}})
    lines += ["## 1. Accuracy by minute (lower is better)", "",
              pd.DataFrame(tab).to_markdown(index=False), ""]
    # 2. criterion comparisons
    verdicts = {}
    lines += ["## 2. Does each added effect help? (paired, bootstrap by match)", "",
              "**Decision rule** (revised before any real evaluation data was seen; see "
              "docs/inplay-plan.md): keep an effect if the RPS difference pooled over the "
              "in-play minutes (15'–90') has a 95% CI below 0 **and** no single minute is "
              "significantly worse. The original rule (CI below 0 at *most* minutes) rejected "
              "the true model in the synthetic check, because time effects matter only late in "
              "matches.", ""]
    for a, b, desc in (("M3", "M1", "Step 3: time-varying rates + stoppage (M3 − M1)"),
                       ("M4", "M3", "Step 4: score effects (M4 − M3)")):
        c = compare(res, sc, a, b)
        c_ex0 = c[c.minute != "0'"]
        mean, lo, hi, n = pooled(res, sc, a, b)
        worse = int((c_ex0.ci_low > 0).sum())
        keep = hi < 0 and worse == 0
        verdicts[desc] = {"pooled": (mean, lo, hi), "worse_minutes": worse, "keep": keep,
                          "old_rule_wins": int((c_ex0.ci_high < 0).sum()), "minutes": len(c_ex0)}
        lines += [f"### {desc}", "", fmt(c).to_markdown(index=False), "",
                  f"Pooled over in-play minutes ({n} matches): {mean:+.5f} "
                  f"(95% CI {lo:+.5f} to {hi:+.5f}); minutes significantly worse: {worse}. "
                  f"→ **{'KEEP' if keep else 'DROP (not shown to help)'}**", "",
                  f"(Original rule, for reference: CI below 0 at {int((c_ex0.ci_high < 0).sum())} "
                  f"of {len(c_ex0)} minutes.)", ""]
    # 3. versus the market
    lines += ["## 3. Model versus market", "",
              "RPS difference model − market (negative = model more accurate than the market).", ""]
    for k in MODELS:
        lines += [f"**{k} − MKT**", "", fmt(compare(res, sc, k, "MKT")).to_markdown(index=False), ""]
    dist = []
    for m, p in EVAL_POINTS:
        sel = (res.minute == m) & (res.period == p)
        if sel.any():
            dist.append({"minute": label(m, p), **{k: round(sc.loc[sel, f"dist_{k}"].mean(), 4)
                                                   for k in MODELS}})
    lines += ["Average distance from the market (half the sum of |p_model − p_market| over "
              "H/D/A; 0.05 = 5 points of probability moved):", "",
              pd.DataFrame(dist).to_markdown(index=False), "",
              "## Caveats", "",
              "- The market is last traded prices (not mid-prices), normalised; they can be "
              "stale for up to a minute and right after goals (those 3 minutes are skipped).",
              "- Over 2.5: once three goals are scored the market is settled and closed, so "
              "its probability is set to 1 (its last traded price is stale).",
              "- Pre-match rates come from the kick-off market, so minute 0 matches it by "
              "construction; differences are purely about in-play updating.",
              "- Goal times are rebuilt from market suspensions (±1 minute); red cards are "
              "not observed or modelled.",
              "- Time shape and score effects were fitted on 2015/16 StatsBomb data; this "
              "evaluation fits nothing on the evaluation season."]
    Path("reports").mkdir(exist_ok=True)
    out = Path(f"reports/inplay_eval_{tag}.md")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return verdicts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--matches", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="synthetic season seed")
    a = ap.parse_args()
    if not a.synthetic and a.season is None:
        ap.error("give --season (e.g. 2223) or --synthetic")
    main(a.season, a.synthetic, a.matches, a.workers, a.seed)
