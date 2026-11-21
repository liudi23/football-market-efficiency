"""Popular-team bias: do soft bookmakers shade the prices of big clubs (because
the public backs them), and does Pinnacle?  Seasons 2016/17-2022/23 (2023/24+
locked).  'Popular' sides: the big clubs below playing a non-big club."""
import glob, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEASONS = ["1617", "1718", "1819", "1920", "2021", "2122", "2223"]
LEAGUES = ["E0", "SP1", "D1", "I1", "F1", "SC0"]
BIG = {"Man United", "Liverpool", "Arsenal", "Chelsea", "Man City", "Tottenham",
       "Barcelona", "Real Madrid", "Bayern Munich", "Dortmund", "Juventus", "Milan",
       "Inter", "Paris SG", "Celtic", "Rangers"}
SOFT = ["B365", "BW", "IW", "WH", "VC"]


def devig(df, b):
    c = [f"{b}{x}" for x in "HDA"]
    o = df[c].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    o[o <= 1] = np.nan
    p = 1 / o
    return p / p.sum(1, keepdims=True)


def load():
    out = []
    for lg in LEAGUES:
        for s in SEASONS:
            f = ROOT / f"data/raw/{lg}_{s}.csv"
            d = pd.read_csv(f, encoding="latin1", on_bad_lines="skip")
            d.columns = [c.replace("﻿", "").replace("ï»¿", "") for c in d.columns]
            d["lg"], d["season"] = lg, s
            out.append(d)
    d = pd.concat(out, ignore_index=True).dropna(subset=["FTR"])
    pin = devig(d, "PSC")
    soft = np.nanmean(np.stack([devig(d, b) if f"{b}H" in d else np.full((len(d), 3), np.nan)
                                for b in SOFT]), 0)
    psc = d[["PSCH", "PSCD", "PSCA"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    mx = d[["MaxCH", "MaxCD", "MaxCA"]].apply(pd.to_numeric, errors="coerce").to_numpy() \
        if "MaxCH" in d else np.full((len(d), 3), np.nan)
    rows = []
    for j, (side, opp, res) in enumerate((("HomeTeam", "AwayTeam", "H"), ("AwayTeam", "HomeTeam", "A"))):
        k = 0 if res == "H" else 2
        rows.append(pd.DataFrame({
            "lg": d.lg, "season": d.season, "team": d[side], "opp": d[opp],
            "home": res == "H", "win": (d.FTR == res).astype(int),
            "pin": pin[:, k], "soft": soft[:, k], "odds_pin": psc[:, k], "odds_max": mx[:, k],
            # opponent "not lose" is messy; record opponent-win odds instead
            "opp_win": (d.FTR == ("A" if res == "H" else "H")).astype(int),
            "opp_pin": pin[:, 2 - k], "opp_odds_pin": psc[:, 2 - k], "opp_odds_max": mx[:, 2 - k]}))
    s = pd.concat(rows, ignore_index=True)
    s["big"] = s.team.isin(BIG) & ~s.opp.isin(BIG)
    s["vsbig"] = ~s.team.isin(BIG) & s.opp.isin(BIG)
    return s.dropna(subset=["pin", "soft", "odds_pin"]).reset_index(drop=True)


def se(x):
    x = x[np.isfinite(x)]
    return x.mean(), x.std() / np.sqrt(len(x)), len(x)


def main():
    s = load()
    s["bin"] = pd.cut(s.pin, [0, .2, .35, .5, .65, .8, 1])
    print(f"team-sides: {len(s)}  big-club sides (vs non-big): {s.big.sum()}")

    print("\n## 1. Soft-book shading: soft consensus prob - Pinnacle prob (pts), same Pinnacle price band")
    print("Pinnacle prob   big clubs (se)          other sides (se)        difference")
    for b, x in s[~s.vsbig].groupby("bin", observed=True):
        a = se((x.soft - x.pin)[x.big].to_numpy() * 100)
        o = se((x.soft - x.pin)[~x.big].to_numpy() * 100)
        if a[2] >= 30:
            print(f"{str(b):12s}   {a[0]:+.2f} ({a[1]:.2f}) n={a[2]:4d}   {o[0]:+.2f} ({o[1]:.2f}) n={o[2]:5d}   {a[0]-o[0]:+.2f}")

    print("\n## 2. Does Pinnacle misprice big clubs?  win - Pinnacle prob, and ROI at Pinnacle close")
    for name, m in (("big clubs", s.big), ("their opponents", s.vsbig), ("all other sides", ~s.big & ~s.vsbig)):
        x = s[m]
        r = se((x.win - x.pin).to_numpy())
        roi = se((x.win * x.odds_pin - 1).to_numpy())
        roim = se((x.win * x.odds_max - 1).to_numpy())
        print(f"   {name:16s} n={r[2]:5d}  residual {r[0]:+.4f} ({r[1]:.4f})  "
              f"ROI Pinnacle {roi[0]:+.4f} ({roi[1]:.4f})  ROI best price {roim[0]:+.4f} ({roim[1]:.4f})")

    print("\n## 3. By season: back big clubs' opponents at the best price")
    for ss, x in s[s.vsbig].groupby("season"):
        roim = se((x.win * x.odds_max - 1).to_numpy())
        r = se((x.win - x.pin).to_numpy())
        print(f"   {ss}  n={roim[2]:4d}  residual {r[0]:+.4f} ({r[1]:.4f})  ROI best price {roim[0]:+.4f} ({roim[1]:.4f})")


if __name__ == "__main__":
    main()
