"""Are Betfair's in-play match-odds and over/under 2.5 markets consistent?

For every in-play snapshot (EPL 2022/23, validation season) fit the remaining
goal rates to the match-odds prices with the Step 3 pricer (price_tv), then
read off the over-2.5 probability those rates imply.  gap = OU market - implied.

  1. Which predicts the over/under outcome better: OU market, the match-odds
     implied price, or a blend?  (logistic, standard errors clustered by match)
  2. Who corrects?  Next-5-minute change of each price regressed on the gap,
     on intervals with no goal.
Caveat: Betfair BASIC files give last-traded prices, which can be stale.

  PYTHONPATH=src python3 scripts/inplay_consistency.py fit [season]    # resumable, ~3 min
  PYTHONPATH=src python3 scripts/inplay_consistency.py report [season] # default 2223

The 2021/22 run is a confirmation test: the code and thresholds are frozen as
they were for the 2022/23 run; do not tune them on 2021/22.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sportsq import inplay

ROOT = Path(__file__).resolve().parents[1]
SEASON = sys.argv[2] if len(sys.argv) > 2 else "2223"
assert SEASON not in ("2324", "2425", "2526", "2627"), "locked holdout"
SRC = ROOT / f"data/processed/betfair_epl_{SEASON}_inplay.csv"
OUT = ROOT / f"data/processed/inplay_consistency_{SEASON}.csv"


def fit(budget=160.0):
    import time
    t0 = time.time()
    prof = inplay.GoalIntensity.from_dict(
        json.loads((ROOT / "data/processed/goal_intensity.json").read_text()))
    d = pd.read_csv(SRC)
    d = d[(d.hg + d.ag <= 2) & (d.minute <= 92)].reset_index(drop=True)
    d["key"] = d.date + "|" + d.home
    done = set(pd.read_csv(OUT).key) if OUT.exists() else set()
    rows = []
    for key, g in d.groupby("key", sort=False):
        if key in done:
            continue
        for r in g.itertuples():
            def fn(lh, la):
                p = inplay.price_tv(prof, lh, la, r.hg, r.ag, r.minute, r.period)["1x2"][0]
                return p[0], p[2], np.nan
            (lh, la), rms = inplay.fit_rates(fn, r.mkt_h, r.mkt_a, None)
            x = inplay.price_tv(prof, lh, la, r.hg, r.ag, r.minute, r.period)
            k = list(np.round(x["lines"], 1)).index(2.5)
            rows.append({"key": key, "minute": r.minute, "period": r.period,
                         "lam_h": lh, "lam_a": la, "rms": rms,
                         "impl_over": x["over"][0][k]})
        if time.time() - t0 > budget:
            break
    if rows:
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    n = pd.read_csv(OUT).key.nunique()
    print(f"fitted matches: {n}/{d.key.nunique()}")


def cluster_logit(X, y, g):
    from scipy.optimize import minimize
    def nll(b):
        e = X @ b
        return np.sum(np.logaddexp(0, e) - y * e)
    b = minimize(nll, np.zeros(X.shape[1]), method="BFGS").x
    p = 1 / (1 + np.exp(-(X @ b)))
    H = X.T @ (X * (p * (1 - p))[:, None])
    s = X * (y - p)[:, None]
    S = pd.DataFrame(s).groupby(g).sum().to_numpy()
    Hi = np.linalg.inv(H)
    return b, np.sqrt(np.diag(Hi @ (S.T @ S) @ Hi))


def cluster_ols(x, y, g):
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    S = pd.DataFrame(X * e[:, None]).groupby(g).sum().to_numpy()
    Hi = np.linalg.inv(X.T @ X)
    return b, np.sqrt(np.diag(Hi @ (S.T @ S) @ Hi))


def report():
    d = pd.read_csv(SRC)
    d["key"] = d.date + "|" + d.home
    f = pd.read_csv(OUT)
    d = d.merge(f, on=["key", "minute", "period"])
    d = d[(d.rms < 1e-4) & d.mkt_over25.between(0.01, 0.99)].copy()
    d["over"] = ((d.FTHG + d.FTAG) >= 3).astype(int)
    d["gap"] = d.mkt_over25 - d.impl_over
    d = d.sort_values(["key", "period", "minute"])
    first = d.groupby("key").gap.transform("first")         # kick-off disagreement
    d["gap_dm"] = d.gap - first
    d["ou_stale"] = d.groupby("key").mkt_over25.diff().fillna(1).eq(0)
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    print(f"snapshots: {len(d)}  matches: {d.key.nunique()}  "
          f"OU price unchanged from previous minute: {d.ou_stale.mean():.0%}")

    print("\n## 0. Size of the disagreement (OU market - match-odds implied), prob points")
    for lo, hi in [(0, 1), (1, 30), (30, 46), (46, 75), (75, 95)]:
        x = d[(d.minute >= lo) & (d.minute < hi)]
        print(f"   minute {lo:2d}-{hi:2d}: mean {x.gap.mean()*100:+.2f}  "
              f"mean |gap| {x.gap.abs().mean()*100:.2f}  n={len(x)}")

    print("\n## 1. Outcome: logit P(over) on logit(OU market) and logit(implied)")
    for name, sub in (("all snapshots", d), ("OU price fresh (changed in last minute)", d[~d.ou_stale]),
                      ("minute >= 1", d[d.minute >= 1])):
        X = np.column_stack([np.ones(len(sub)), lg(sub.mkt_over25), lg(sub.impl_over)])
        b, se = cluster_logit(X, sub.over.to_numpy(), sub.key.to_numpy())
        ll = lambda p: -np.mean(sub.over * np.log(p) + (1 - sub.over) * np.log(1 - p))
        print(f"   [{name}] n={len(sub)}  weight OU {b[1]:+.3f} ({se[1]:.3f})  "
              f"weight implied {b[2]:+.3f} ({se[2]:.3f})   "
              f"log loss OU {ll(np.clip(sub.mkt_over25,1e-4,1-1e-4)):.4f} implied {ll(np.clip(sub.impl_over,1e-4,1-1e-4)):.4f}")

    print("\n## 2. Who corrects?  5-minute-ahead change regressed on the gap (no goal in between)")
    d["t"] = np.where(d.period == 2, d.minute + 1000, d.minute)
    nxt = d[["key", "t", "mkt_over25", "impl_over", "hg", "ag"]].copy()
    nxt["t"] -= 5
    m = d.merge(nxt, on=["key", "t"], suffixes=("", "_n"))
    m = m[(m.hg == m.hg_n) & (m.ag == m.ag_n)]
    for name, sub in (("all", m), ("OU price fresh", m[~m.ou_stale])):
        g = sub.key.to_numpy()
        b1, s1 = cluster_ols(sub.gap.to_numpy(), (sub.mkt_over25_n - sub.mkt_over25).to_numpy(), g)
        b2, s2 = cluster_ols(sub.gap.to_numpy(), (sub.impl_over_n - sub.impl_over).to_numpy(), g)
        print(f"   [{name}] n={len(sub)}  OU market moves {b1[1]:+.3f} ({s1[1]:.3f}) x gap;  "
              f"implied moves {b2[1]:+.3f} ({s2[1]:.3f}) x gap")
    print("   (negative OU slope = OU moves toward match odds; positive implied slope = match odds move toward OU)")

    print("\n## 3. Trading the gap: back/lay OU 2.5 at the market price when |gap - kick-off gap| > thr")
    print("   (no commission; last-traded price, so an upper bound)")
    for thr in (0.02, 0.04, 0.06):
        for name, sub in (("all", d[d.minute >= 1]), ("fresh", d[(d.minute >= 1) & ~d.ou_stale])):
            x = sub[sub.gap_dm.abs() > thr]
            # one trade per match: first signal
            x = x.groupby("key").head(1)
            side = np.sign(-x.gap_dm)                   # OU too high -> lay (sell) over
            pnl = side * (x.over - x.mkt_over25)        # stake to win 1 unit, in prob units
            if len(x) >= 20:
                print(f"   thr {thr:.2f} [{name:5s}] trades {len(x):3d}  mean edge {pnl.mean():+.4f} "
                      f"({pnl.std()/np.sqrt(len(x)):.4f}) per unit")


if __name__ == "__main__":
    {"fit": fit, "report": report}[sys.argv[1]]()
