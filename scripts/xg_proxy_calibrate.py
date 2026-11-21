"""Calibrate a shot-count xG proxy against StatsBomb 2015/16 xG.

StatsBomb open data has shot-level xG for every 2015/16 match in the Premier
League, La Liga, Serie A and Ligue 1 (summed per team in
data/external/statsbomb_team_xg_1516.csv).  football-data has only shot counts.
Fit xG ~ b1*shots_on_target + b2*shots_off_target (+ corners), non-negative
least squares, leave-one-league-out.  2015/16 is before every evaluation
season, so using it for calibration leaks nothing.
Writes data/processed/xg_proxy.json for scripts/xg_eval.py (source 'shots2').
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import nnls
from sportsq.xg import match_understat

ROOT = Path(__file__).resolve().parents[1]
sb = pd.read_csv(ROOT / "data/external/statsbomb_team_xg_1516.csv")
LG = {"Premier League": "E0", "La Liga": "SP1", "Serie A": "I1", "Ligue 1": "F1"}
us = pd.DataFrame({"league": sb.competition.map(LG), "date": sb.date, "home_us": sb.home, "away_us": sb.away,
                   "hg_us": sb.hg, "ag_us": sb.ag, "hxg": sb.hxg, "axg": sb.axg})
fd = []
for code in LG.values():
    r = pd.read_csv(ROOT / f"data/raw/{code}_1516.csv", encoding="latin1")
    fd.append(pd.DataFrame({"league": code, "date": pd.to_datetime(r.Date, dayfirst=True, format="mixed"),
                            "home": r.HomeTeam, "away": r.AwayTeam, "hg": r.FTHG, "ag": r.FTAG,
                            "hs": r.HS, "as_": r.AS, "hst": r.HST, "ast": r.AST, "hc": r.HC, "ac": r.AC}))
fd = pd.concat(fd, ignore_index=True)
m, names, miss = match_understat(fd, us)
m = m.dropna(subset=["hxg"])
print(f"StatsBomb 2015/16 xG matched to football-data: {len(m)}/{len(fd)} matches")
T = pd.DataFrame({"lg": np.r_[m.league, m.league], "xg": np.r_[m.hxg, m.axg], "g": np.r_[m.hg, m.ag],
                  "s": np.r_[m.hs, m["as_"]], "sot": np.r_[m.hst, m.ast], "c": np.r_[m.hc, m.ac]})
T["sot"] = np.minimum(T.sot, T.s)
T["off"] = T.s - T.sot
feats = {"A: 0.31 x on-target (first proxy)": None, "B: on-target + off-target": ["sot", "off"],
         "C: B + corners": ["sot", "off", "c"]}
print("\nleave-one-league-out, team-match rows")
print(f"{'proxy':36s} corr with xG   RMSE   coefficients (all four leagues)")
best = None
for name, f in feats.items():
    pred = np.zeros(len(T))
    for l in T.lg.unique():
        tr, te = (T.lg != l).to_numpy(), (T.lg == l).to_numpy()
        if f is None:
            pred[te] = 0.31 * T.sot[te]
        else:
            b, _ = nnls(T.loc[tr, f].to_numpy(float), T.xg[tr].to_numpy(float))
            pred[te] = T.loc[te, f].to_numpy(float) @ b
    coef = {} if f is None else dict(zip(f, np.round(nnls(T[f].to_numpy(float), T.xg.to_numpy(float))[0], 4)))
    print(f"{name:36s} {np.corrcoef(pred, T.xg)[0, 1]:.3f}          {np.sqrt(np.mean((pred - T.xg) ** 2)):.3f}  {coef}")
    if f == ["sot", "off"]:
        best = coef
print(f"\nreference: corr(goals, xG) = {np.corrcoef(T.g, T.xg)[0, 1]:.3f}")
(ROOT / "data/processed/xg_proxy.json").write_text(json.dumps({k: float(v) for k, v in best.items()}))
print("saved proxy B coefficients:", best)
