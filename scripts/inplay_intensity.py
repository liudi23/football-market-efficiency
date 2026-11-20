"""In-play Step 3: fit and check the time-varying goal intensity.

Uses the StatsBomb files from scripts/download_statsbomb.py and writes
reports/inplay_intensity.md plus the fitted profile to
data/processed/goal_intensity.json (load it with GoalIntensity.from_dict).

Sections:
1. Fitted intensity by time bin (with exact Poisson 95% CIs) and stoppage time.
2. Is the shape the same for home and away goals? (needed for the in-play
   Dixon-Coles correction): chi-square test of homogeneity.
3. Does time variation predict goal timing out of sample? 5-fold
   cross-validated log-likelihood of goal times: constant rate vs 15-minute
   bins vs 5-minute bins, with a bootstrap CI over matches.
4. Price impact versus the constant-rate pricer (Step 1).

Usage:  python scripts/inplay_intensity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from sportsq import inplay


def match_loglik(prof: inplay.GoalIntensity, goals_per_match, m, g):
    """Poisson-process log-likelihood of each match's goal times (both teams pooled).

    Intensity at played time u is G * w_b(u); each match uses its own stoppage
    lengths. Constant terms that don't depend on the shape cancel in differences.
    """
    s1 = (m.end_p1 - 45).clip(lower=0).to_numpy()
    s2 = (m.end_p2 - 90).clip(lower=0).to_numpy()
    b = prof.bounds(s1, s2)
    integ = (prof.weights * np.diff(b, axis=1)).sum(1) * goals_per_match
    pos = {mid: i for i, mid in enumerate(m.match_id)}
    ll = -integ
    for row in g.itertuples(index=False):
        i = pos[row.match_id]
        u = row.t if row.period == 1 else b[i, len(prof.edges1)] + (row.t - 45)
        u = min(u, b[i, -1] - 1e-9)
        k = int((u >= b[i, 1:-1]).sum())
        ll[i] += np.log(goals_per_match * prof.weights[k])
    return ll


def constant_like(prof: inplay.GoalIntensity) -> inplay.GoalIntensity:
    """Same bins and stoppage, but one rate per played minute everywhere."""
    c = inplay.GoalIntensity(prof.edges1, prof.edges2, np.ones_like(prof.weights),
                             prof.s1, prof.s2, prof.probs)
    c.weights = c.weights / c.total()
    return c


def cv_compare(m, e, folds=5, seed=0, n_boot=4000):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, len(m))
    g_all = e[e.event == "goal"]
    out = {"constant": [], "15-min bins": [], "5-min bins": []}
    for f in range(folds):
        tr, te = m[fold != f], m[fold == f]
        g_te = g_all[g_all.match_id.isin(te.match_id)]
        p15 = inplay.fit_goal_intensity(tr, e, 15)
        p5 = inplay.fit_goal_intensity(tr, e, 5)
        G = p15.table.attrs["goals_per_match"]  # same mean goals for every model
        out["constant"].append(match_loglik(constant_like(p15), G, te, g_te))
        out["15-min bins"].append(match_loglik(p15, G, te, g_te))
        out["5-min bins"].append(match_loglik(p5, G, te, g_te))
    ll = {k: np.concatenate(v) for k, v in out.items()}
    rows = []
    for k in ("15-min bins", "5-min bins"):
        d = ll[k] - ll["constant"]
        boot = rng.choice(d, (n_boot, len(d))).mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"model vs constant": k, "Δ log-lik per match": round(d.mean(), 4),
                     "95% CI": f"{lo:.4f} to {hi:.4f}", "total Δ log-lik": round(d.sum(), 1)})
    d = ll["5-min bins"] - ll["15-min bins"]
    boot = rng.choice(d, (n_boot, len(d))).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rows.append({"model vs constant": "5-min minus 15-min", "Δ log-lik per match": round(d.mean(), 4),
                 "95% CI": f"{lo:.4f} to {hi:.4f}", "total Δ log-lik": round(d.sum(), 1)})
    return pd.DataFrame(rows)


def main():
    m, e = inplay.load_statsbomb()
    prof = inplay.fit_goal_intensity(m, e)
    t = prof.table
    lines = ["# In-play Step 3: time-varying goal intensity", "",
             f"Data: StatsBomb open data, {t.attrs['n_matches']} matches (2015/16 Premier League, "
             f"La Liga, Serie A, Ligue 1); {int(t.goals.sum())} goals; "
             f"{t.attrs['goals_per_match']:.3f} goals per match.", "",
             "## 1. Goal intensity by time bin", "",
             "Relative rate = goals per minute in the bin ÷ goals per minute over all play "
             "(incl. stoppage). CIs are exact Poisson intervals.", "",
             t.to_markdown(index=False), "",
             f"Stoppage time: first half mean {t.attrs['mean_s1']:.2f} min, second half mean "
             f"{t.attrs['mean_s2']:.2f} min; modelled as {len(prof.s1)} joint scenarios "
             "(0.5-minute resolution).", ""]

    # 2. home vs away shape
    g = e[e.event == "goal"].merge(m[["match_id"]])
    ph, pa = (inplay.fit_goal_intensity(m, e[(e.event != "goal") | (e.side == s)])
              for s in ("H", "A"))
    cont = np.vstack([ph.table.goals, pa.table.goals])
    chi, pval, dof, _ = chi2_contingency(cont)
    comp = pd.DataFrame({"bin": t.bin, "home goals": ph.table.goals,
                         "home relative rate": ph.table.relative_rate,
                         "away goals": pa.table.goals,
                         "away relative rate": pa.table.relative_rate})
    lines += ["## 2. Same shape for home and away goals?", "",
              "The in-play Dixon–Coles correction assumes both teams share one time shape.", "",
              comp.to_markdown(index=False), "",
              f"Chi-square test that the home share of goals is the same in every bin: "
              f"χ² = {chi:.1f} on {dof} df, p = {pval:.3f}.", ""]

    # 3. out-of-sample comparison
    cv = cv_compare(m, e)
    lines += ["## 3. Does time variation help out of sample?", "",
              "5-fold cross-validated log-likelihood of goal times (Poisson process, both teams "
              "pooled, same goals-per-match for every model so only the time shape differs). "
              "Positive Δ = better than a constant rate per played minute. Bootstrap CI over "
              "matches.", "", cv.to_markdown(index=False), ""]

    # 4. price impact
    states = [(0, 0, 30, 1), (0, 0, 45, 1), (0, 0, 47, 1), (0, 0, 60, 2), (0, 0, 80, 2),
              (1, 0, 85, 2), (0, 0, 90, 2), (1, 1, 92, 2), (0, 0, 95, 2)]
    rows = []
    for h, a, mi, p in states:
        c_min = min(mi, 45) if p == 1 else min(mi, 90)
        c = inplay.price(1.5, 1.1, h, a, c_min, rho=-0.08)
        v = inplay.price_tv(prof, 1.5, 1.1, h, a, mi, p, rho=-0.08)
        clock = f"{mi if mi <= (45 if p == 1 else 90) else ('45+' if p == 1 else '90+') + str(mi - (45 if p == 1 else 90))}'"
        rows.append({"state": f"{h}-{a}, {clock}" + (" (1st half)" if p == 1 and mi >= 45 else ""),
                     "Step 1 H/D/A": " / ".join(f"{x:.3f}" for x in c["1x2"][0]),
                     "Step 3 H/D/A": " / ".join(f"{x:.3f}" for x in v["1x2"][0]),
                     "Step 1 over 2.5": f"{c['over'][0][2]:.3f}",
                     "Step 3 over 2.5": f"{v['over'][0][2]:.3f}"})
    lines += ["## 4. Price impact (pre-match expected goals 1.5 v 1.1, ρ = −0.08)", "",
              "Step 1 assumes a constant rate over exactly 90 minutes; Step 3 uses the fitted "
              "shape and random stoppage time.", "", pd.DataFrame(rows).to_markdown(index=False), "",
              "## Caveats", "",
              "- One season of four leagues (2015/16), inside the development split.",
              "- The late rise in goals partly reflects score effects (trailing teams attack, "
              "leaders defend). Step 4 models those explicitly; until then this shape absorbs them.",
              "- Data: StatsBomb open data (attribution required when publishing)."]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/inplay_intensity.md").write_text("\n".join(lines) + "\n")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/processed/goal_intensity.json").write_text(json.dumps(prof.to_dict(), indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
