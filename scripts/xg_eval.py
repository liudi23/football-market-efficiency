"""Do xG ratings bring a goals model closer to the market?

Big-5 leagues, walk-forward from 2016/17 (two seasons of xG history first).
alpha (share of xG in the rating target) is chosen on 2016/17-2018/19 and
reported on 2019/20-2022/23.  The 2023/24+ holdout is never loaded.

  PYTHONPATH=src python3 scripts/xg_eval.py fit E0 [understat|shots]   # one league (cached)
  PYTHONPATH=src python3 scripts/xg_eval.py report [understat|shots]

'shots' uses a shot-based xG proxy from football-data (no download needed):
xG_proxy = b1*shots + b2*shots_on_target, fitted by Poisson regression of goals
on the development seasons only.
  PYTHONPATH=src python3 scripts/xg_eval.py sim           # synthetic check
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

from sportsq import data, metrics
from sportsq.backtest import MarketBlend
from sportsq.implied import implied
from sportsq.xg import match_understat, walk_forward_xg

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ["E0", "SP1", "D1", "I1", "F1"]
ALPHAS = (0.25, 0.5, 0.75, 1.0)
START, DEV_END, VAL_END = "2016-07-01", "2019-07-01", "2023-07-01"
CACHE = ROOT / "data/processed"
CLOSE = ["close_h", "close_d", "close_a"]


def shots_proxy(fd):
    """Fit goals ~ Poisson(b1*shots + b2*on_target) on development seasons, apply to all."""
    from scipy.optimize import minimize
    c = ["hs", "as_", "hst", "ast"]
    fd[c] = fd[c].apply(pd.to_numeric, errors="coerce")
    fd = fd.dropna(subset=c).reset_index(drop=True)
    dev = fd[fd.date < DEV_END]
    S = np.concatenate([dev.hs, dev["as_"]]); T = np.concatenate([dev.hst, dev.ast])
    G = np.concatenate([dev.hg, dev.ag])
    T = np.minimum(T, S)                               # data errors: on target > shots
    def nll(b):
        lam = np.clip(b[0] * (S - T) + b[1] * T, 1e-6, None)   # off-target and on-target shots
        return (lam - G * np.log(lam)).sum()
    b = minimize(nll, [0.03, 0.3], bounds=[(0, 1), (0, 1)]).x
    fd["hxg"] = b[0] * (fd.hs - np.minimum(fd.hst, fd.hs)) + b[1] * np.minimum(fd.hst, fd.hs)
    fd["axg"] = b[0] * (fd["as_"] - np.minimum(fd.ast, fd["as_"])) + b[1] * np.minimum(fd.ast, fd["as_"])
    print(f"   shot proxy: {b[0]:.3f} per off-target shot, {b[1]:.3f} per shot on target")
    return fd


def load_league(lg, source="understat"):
    fd = data.load(raw_dir=str(ROOT / "data/raw"), leagues=[lg])
    fd = fd[(fd.date >= "2014-07-01") & (fd.date < VAL_END)].reset_index(drop=True)
    if source == "shots2":                     # proxy calibrated to StatsBomb xG
        import json
        b = json.loads((ROOT / "data/processed/xg_proxy.json").read_text())
        c = ["hs", "as_", "hst", "ast"]
        fd[c] = fd[c].apply(pd.to_numeric, errors="coerce")
        fd = fd.dropna(subset=c).reset_index(drop=True)
        for side, s_, t_ in (("h", "hs", "hst"), ("a", "as_", "ast")):
            on = np.minimum(fd[t_], fd[s_])
            fd[f"{side}xg"] = b["sot"] * on + b["off"] * (fd[s_] - on)
        print(f"{lg}: proxy {b['sot']:.3f} x on target + {b['off']:.3f} x off target; {len(fd)} matches")
        return fd
    if source == "shots":
        m = shots_proxy(fd)
        print(f"{lg}: {len(fd)} matches, shots available for {len(m)}")
        return m
    us = pd.read_csv(ROOT / "data/external/understat_matches.csv")
    us = us[us.league == lg]
    m, names, missing = match_understat(fd, us)
    print(f"{lg}: {len(fd)} matches, xG attached to {len(fd) - missing} "
          f"({1 - missing / len(fd):.1%}); {len(names)} team names mapped")
    return m.dropna(subset=["hxg", "axg"]).reset_index(drop=True)


def fit(lg, source):
    t0 = time.time()
    d = load_league(lg, source)
    wf = walk_forward_xg(d, START, ALPHAS)
    wf.to_csv(CACHE / f"xg_wf_{source}_{lg}.csv", index=False)
    print(f"{lg}: {len(wf)} predictions in {time.time() - t0:.0f}s")


def tv(p, q):
    """Total-variation distance in probability points."""
    return 50 * np.abs(p - q).sum(1)


def report(wf, title):
    wf = wf.dropna(subset=CLOSE).reset_index(drop=True)
    wf["date"] = pd.to_datetime(wf.date)
    y = metrics.encode(wf.result)
    mk = implied(wf[CLOSE].to_numpy(float), "shin")
    models = {"goals DC": wf[["dc_h", "dc_d", "dc_a"]].to_numpy()}
    for al in ALPHAS:
        models[f"xG alpha={al:g}"] = wf[[f"a{al:g}_h", f"a{al:g}_d", f"a{al:g}_a"]].to_numpy()
    dev = (wf.date < DEV_END).to_numpy()
    val = ~dev
    week = wf.date.dt.to_period("W").astype(str).to_numpy()
    lines = [f"# {title}", "", f"Matches: development {dev.sum()}, validation {val.sum()}", ""]

    dev_rps = {k: metrics.rps(p[dev], y[dev]).mean() for k, p in models.items()}
    best = min((k for k in models if k != "goals DC"), key=dev_rps.get)
    lines += ["## 1. Choose alpha on development seasons (RPS)", ""]
    lines += [f"- {k}: {v:.5f}" for k, v in dev_rps.items()]
    lines += [f"- chosen: **{best}**", ""]

    lines += ["## 2. Validation seasons", "",
              "| model | RPS | log loss | distance to market (pts) |", "|---|---|---|---|",
              f"| market (Pinnacle close) | {metrics.rps(mk[val], y[val]).mean():.5f} | "
              f"{metrics.log_loss(mk[val], y[val]).mean():.4f} | 0 |"]
    for k, p in models.items():
        lines.append(f"| {k} | {metrics.rps(p[val], y[val]).mean():.5f} | "
                     f"{metrics.log_loss(p[val], y[val]).mean():.4f} | {tv(p[val], mk[val]).mean():.2f} |")
    lines.append("")

    rp = lambda p: metrics.rps(p[val], y[val])
    for a, b in ((best, "goals DC"), (best, "market")):
        pb = mk if b == "market" else models[b]
        r = metrics.paired_bootstrap(rp(models[a]), rp(pb), week[val])
        lines.append(f"- RPS {a} minus {b}: {r['mean_diff']:+.5f} "
                     f"(95% CI {r['ci_low']:+.5f} to {r['ci_high']:+.5f})")
    r = metrics.paired_bootstrap(rp(models["goals DC"]), rp(mk), week[val])
    lines.append(f"- RPS goals DC minus market: {r['mean_diff']:+.5f} "
                 f"(95% CI {r['ci_low']:+.5f} to {r['ci_high']:+.5f})")
    gap_dc, gap_x = tv(models["goals DC"][val], mk[val]).mean(), tv(models[best][val], mk[val]).mean()
    lines += [f"- distance to market: goals DC {gap_dc:.2f} pts -> {best} {gap_x:.2f} pts "
              f"({(gap_dc - gap_x) / gap_dc:.0%} closer)", ""]

    lines += ["## 3. Does the xG model add anything to the market? (log-linear blend)", ""]
    for k in ("goals DC", best):
        b = MarketBlend().fit(mk[dev], models[k][dev], y[dev])
        pb = b.predict_proba(mk[val], models[k][val])
        bv = MarketBlend().fit(mk[val], models[k][val], y[val])
        r = metrics.paired_bootstrap(metrics.rps(pb, y[val]), rp(mk), week[val])
        lines.append(f"- {k}: weights fitted on dev market {b.w_[0]:.3f} / model {b.w_[1]:+.3f}; "
                     f"on validation {bv.w_[0]:.3f} / {bv.w_[1]:+.3f}. Blend (dev weights) minus market "
                     f"RPS on validation {r['mean_diff']:+.5f} (CI {r['ci_low']:+.5f} to {r['ci_high']:+.5f})")
    lines.append("")
    if "league" in wf:
        lines += ["## 4. By league (validation): RPS and distance to market", "",
                  "| league | market | goals DC | " + best + " | dist DC | dist xG |", "|---|---|---|---|---|---|"]
        for lg in sorted(wf.league.unique()):
            k = (wf.league == lg).to_numpy() & val
            lines.append(f"| {lg} | {metrics.rps(mk[k], y[k]).mean():.5f} | "
                         f"{metrics.rps(models['goals DC'][k], y[k]).mean():.5f} | "
                         f"{metrics.rps(models[best][k], y[k]).mean():.5f} | "
                         f"{tv(models['goals DC'][k], mk[k]).mean():.2f} | {tv(models[best][k], mk[k]).mean():.2f} |")
    return "\n".join(lines) + "\n"


def sim():
    from sportsq.simulate import SimConfig, simulate
    df = simulate(SimConfig(n_seasons=9, first_season=2014, seed=3))
    df["date"] = pd.to_datetime(df.date)
    # synthetic xG: shots x 0.11 -> unbiased, noisy measure of the true rate,
    # independent of the goals actually scored
    df["hxg"], df["axg"] = 0.11 * df.hs, 0.11 * df["as_"]
    wf = walk_forward_xg(df, START, ALPHAS)
    txt = report(wf, "xG model on SIMULATED data (synthetic xG = 0.11 x shots)")
    (ROOT / "reports/xg_eval_sim.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    cmd = sys.argv[1]
    src = sys.argv[3] if cmd == "fit" and len(sys.argv) > 3 else \
        (sys.argv[2] if cmd == "report" and len(sys.argv) > 2 else "understat")
    if cmd == "fit":
        fit(sys.argv[2], src)
    elif cmd == "sim":
        sim()
    else:
        wf = pd.concat([pd.read_csv(CACHE / f"xg_wf_{src}_{lg}.csv") for lg in LEAGUES
                        if (CACHE / f"xg_wf_{src}_{lg}.csv").exists()])
        label = {"understat": "Understat xG", "shots": "shot-based xG proxy (fitted to goals)",
                 "shots2": "shot-based xG proxy (calibrated to StatsBomb xG)"}[src]
        txt = report(wf, f"{label} ratings vs goals Dixon-Coles vs the market (big-5 leagues)")
        (ROOT / f"reports/xg_eval_{src}.md").write_text(txt)
        print(txt)
