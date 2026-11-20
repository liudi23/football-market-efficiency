"""In-play Step 4: estimate and check score effects.

Uses the StatsBomb files from scripts/download_statsbomb.py and writes
reports/inplay_score_effects.md plus the fitted model to
data/processed/score_effects.json ({"predictive": ..., "tactical": ...}; load
each with ScoreEffects.from_dict; use "predictive" for pricing).

Sections:
1. Score-state multipliers: tactical (in-sample team strengths, with a
   likelihood-ratio test) and predictive (out-of-fold team strengths).
2. Why the estimation method matters: the same data fitted three ways.
3. Time shape before (Step 3) and after (Step 4) removing score effects.
4. Out of sample: 5-fold cross-validated log-likelihood of goals with and
   without score effects, for both variants.
5. Price impact versus Step 3.

Usage:  python scripts/inplay_score_effects.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from sportsq import inplay

LABELS = ["≤ −2", "−1", "0 (level)", "+1", "≥ +2"]


def _boot_ci(d, rng, n_boot=4000):
    boot = rng.choice(d, (n_boot, len(d))).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return d.mean(), lo, hi, d.sum()


def cv_score_effects(m, e, folds=5, seed=0):
    """Held-out Poisson log-likelihood per match, with minus without score effects.

    "tactical": team strengths, time bins and score effects all refitted on the
    training folds. "predictive": every segment gets an out-of-fold team-strength
    offset (fitted without that match), then time bins and score effects are
    refitted on the training folds on top of the offsets. Returns
    {variant: (mean, ci_low, ci_high, total)}.
    """
    from scipy.sparse import csr_matrix
    rng = np.random.default_rng(seed)
    base = inplay.fit_goal_intensity(m, e)
    seg = inplay.build_segments(m, e, tuple(base.edges1), tuple(base.edges2))
    n_bins = len(base.weights)
    teams = pd.Index(sorted(set(seg.team) | set(seg.opponent)))
    fold_of = dict(zip(m.match_id, rng.integers(0, folds, len(m))))
    f = seg.match_id.map(fold_of).to_numpy()
    L, Y = seg.length.to_numpy(float), seg.goals.to_numpy(float)
    by_match = seg.match_id.to_numpy()

    def heldout(X, offset, pen):
        ll = np.zeros(len(seg))
        for k in range(folds):
            tr, te = f != k, f == k
            th, _, _ = inplay._poisson_newton(X[tr], offset[tr], Y[tr], pen)
            eta = offset[te] + X[te] @ th
            ll[te] = Y[te] * eta - np.exp(eta)
        return pd.Series(ll).groupby(by_match).sum().to_numpy()

    out = {}
    # tactical: everything refitted per fold
    lls = {}
    for name, states in (("with", [-2, -1, 1, 2]), ("without", [])):
        X, pen, _, _ = inplay.score_design(seg, n_bins, states, teams)
        lls[name] = heldout(X, np.log(L), pen)
    out["tactical (in-sample team strengths)"] = _boot_ci(lls["with"] - lls["without"], rng)
    # predictive: out-of-fold strength offsets
    X0, pen0, _, nf0 = inplay.score_design(seg, n_bins, [], teams)
    cols = np.r_[[0, 1], np.arange(nf0, X0.shape[1])]
    off = np.zeros(len(seg))
    for k in range(folds):
        th, _, _ = inplay._poisson_newton(X0[f != k], np.log(L[f != k]), Y[f != k], pen0)
        off[f == k] = X0[f == k][:, cols] @ th[cols]
    lls = {}
    for name, states in (("with", [-2, -1, 1, 2]), ("without", [])):
        X = csr_matrix(np.column_stack(
            [np.ones(len(seg))] + [(seg.bin.to_numpy() == b).astype(float) for b in range(1, n_bins)] +
            [(seg["diff"].to_numpy() == d).astype(float) for d in states]))
        lls[name] = heldout(X, np.log(L) + off, np.zeros(X.shape[1]))
    out["predictive (out-of-fold team strengths)"] = _boot_ci(lls["with"] - lls["without"], rng)
    return out


def main():
    m, e = inplay.load_statsbomb()
    eff = inplay.fit_score_effects(m, e)
    pred = inplay.fit_score_effects(m, e, method="predictive")
    null = inplay.fit_score_effects(m, e, score_effects=False)
    lr = 2 * (eff.table.attrs["loglik"] - null.table.attrs["loglik"])
    p_lr = chi2.sf(lr, 4)
    t = eff.table
    lines = ["# In-play Step 4: score effects", "",
             f"Data: StatsBomb open data, {len(m)} matches (2015/16 Premier League, La Liga, "
             "Serie A, Ligue 1). Model: each team's goals follow a Poisson process with log-rate "
             "= intercept + home + attack[team] − defence[opponent] + time bin + score state, "
             "fitted by maximum likelihood on segments of constant time bin and score.", "",
             "## 1. Score-state multipliers", "",
             "Multiplier on a team's scoring rate relative to a level score, by goal difference "
             "from that team's point of view (95% CIs).", "",
             "**Tactical** (team strengths estimated in-sample, i.e. using these same matches):", "",
             t.to_markdown(index=False), "",
             f"Likelihood-ratio test against no score effects: LR = {lr:.1f} on 4 df, "
             f"p = {p_lr:.3g}.", "",
             "**Predictive** (team strengths cross-fitted on other matches, so the effects are "
             "measured relative to an *estimated* strength, as a live pricer must use them):", "",
             pred.table.to_markdown(index=False), "",
             "The score is also evidence about strength: a team that leads is probably better "
             "than estimated, which offsets the tactical slowdown of leaders. Trailing teams "
             "score more under both.", ""]

    # 2. method comparison
    rows = []
    for meth, desc in (("poisson", "**Tactical:** full likelihood + team strengths"),
                       ("predictive", "**Predictive:** + out-of-fold team strengths"),
                       ("naive", "Naive: full likelihood, no team strengths"),
                       ("conditional", "Invalid: conditional on each team's goals")):
        f = {"poisson": eff, "predictive": pred}.get(meth) or \
            inplay.fit_score_effects(m, e, method=meth)
        rows.append({"method": desc, **{lab: f"{x:.2f}" for lab, x in
                                        zip(LABELS, f.multipliers) if lab != "0 (level)"}})
    lines += ["## 2. Why the estimation method matters", "",
              pd.DataFrame(rows).to_markdown(index=False), "",
              "- **Naive:** stronger teams both lead more often and score more, so without "
              "team strengths 'leading' looks like it *raises* scoring. Pure confounding.",
              "- **Conditional:** conditioning on each team's goal total in the match is "
              "invalid here because the score states are created by the team's own goals. "
              "A team that scored once scored it while not yet ahead and, by construction, "
              "never scores again while leading. The effects come out wildly overstated.",
              "- The simulation tests in `tests/test_inplay.py` reproduce both biases on "
              "synthetic leagues with known effects, and show the correct method recovers the "
              "truth.", ""]

    # 3. time shape
    gross = inplay.fit_goal_intensity(m, e)
    bins = gross.table.bin.tolist()
    lines += ["## 3. Time shape before and after removing score effects", "",
              "Rate in each bin relative to minutes 0–15.", "",
              pd.DataFrame({"bin": bins,
                            "Step 3 (gross)": (gross.weights / gross.weights[0]).round(3),
                            "Step 4 (net of score effects)":
                                (eff.profile.weights / eff.profile.weights[0]).round(3)}
                           ).to_markdown(index=False), "",
              "Most of the late rise in goals survives: it is a time effect, not a score effect.",
              ""]

    # 4. out of sample
    cv = cv_score_effects(m, e)
    lines += ["## 4. Do score effects help out of sample?", "",
              "5-fold cross-validation by match. Held-out Poisson log-likelihood of each team's "
              "goals, with minus without score effects (positive = score effects help). "
              "Bootstrap 95% CI over matches.", "",
              pd.DataFrame([{"variant": k, "Δ log-lik per match": f"{v[0]:+.4f}",
                             "95% CI": f"{v[1]:+.4f} to {v[2]:+.4f}", "total": f"{v[3]:+.1f}"}
                            for k, v in cv.items()]).to_markdown(index=False), "",
              "- **Tactical** effects are significant in-sample but make held-out predictions "
              "worse: with team strengths estimated from other matches, the score carries "
              "information about strength that the tactical multipliers push the wrong way.",
              "- **Predictive** effects neither help nor hurt goal-rate prediction here. The "
              "effects are small relative to one season's noise; Step 6 tests whether they "
              "improve in-play forecasts of final results.", ""]

    # 5. price impact
    c = inplay.calibrate(pred, 1.5, 1.1)
    states = [(1, 0, 30, 1), (1, 0, 70, 2), (0, 1, 70, 2), (2, 0, 60, 2), (0, 2, 60, 2),
              (0, 0, 80, 2), (1, 1, 85, 2)]
    rows = []
    for h, a, mi, p in states:
        s3 = inplay.price_tv(pred.profile, 1.5, 1.1, h, a, mi, p)
        s4 = inplay.price_se(pred, 1.5, 1.1, h, a, mi, p, scale=c)
        rows.append({"state": f"{h}-{a}, {mi}'",
                     "Step 3 H/D/A": " / ".join(f"{x:.3f}" for x in s3["1x2"][0]),
                     "Step 4 H/D/A": " / ".join(f"{x:.3f}" for x in s4["1x2"]),
                     "Step 3 over 2.5": f"{s3['over'][0][2]:.3f}",
                     "Step 4 over 2.5": f"{s4['over'][2]:.3f}"})
    lines += ["## 5. Price impact (pre-match expected goals 1.5 v 1.1, ρ = 0)", "",
              "Both use the predictive model's time shape; Step 4 adds its multipliers and "
              f"recalibrates so pre-match expected goals stay 1.5 and 1.1 (scale factors "
              f"{c[0]:.4f}, {c[1]:.4f}).", "", pd.DataFrame(rows).to_markdown(index=False), "",
              "## Caveats", "",
              "- One season of four leagues (2015/16); team strength is one attack and one "
              "defence parameter per team for the season.",
              "- Effects are assumed the same for home and away teams and at every minute "
              "(trailing late may matter more than trailing early).",
              "- Red cards are not yet modelled (Step 5) and are partly mixed into these "
              "estimates.",
              "- Whether score effects improve in-play forecasts of final results is decided "
              "in Step 6.",
              "- Data: StatsBomb open data (attribution required when publishing)."]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/inplay_score_effects.md").write_text("\n".join(lines) + "\n")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/processed/score_effects.json").write_text(json.dumps(
        {"predictive": pred.to_dict(), "tactical": eff.to_dict()}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
