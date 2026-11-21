"""Crowd-free football (COVID): how fast did the market price the loss of home
advantage?  All 18 football-data leagues, 2016/17-2022/23 (2023/24+ locked).
Market = Pinnacle closing 1X2, margin removed.  Residual = outcome - market.
Ghost period approximated as 2020-03-15 .. 2021-05-31 (partial crowds in some
countries in autumn 2020 and May 2021 are ignored)."""
import glob, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEASONS = ["1617", "1718", "1819", "1920", "2021", "2122", "2223"]
G0, G1 = pd.Timestamp("2020-03-15"), pd.Timestamp("2021-05-31")


def load():
    out = []
    for f in sorted(glob.glob(str(ROOT / "data/raw/*.csv"))):
        s = re.search(r"_(\d{4})\.csv", f).group(1)
        if s not in SEASONS:
            continue
        d = pd.read_csv(f, encoding="latin1", on_bad_lines="skip")
        d.columns = [c.replace("﻿", "").replace("ï»¿", "") for c in d.columns]
        d["season"], d["lg"] = s, Path(f).name.split("_")[0]
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    c = ["PSCH", "PSCD", "PSCA", "FTHG", "FTAG"]
    d[c] = d[c].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=c + ["Date", "FTR"])
    d = d[(d[["PSCH", "PSCD", "PSCA"]] > 1).all(1)].copy()
    p = 1 / d[["PSCH", "PSCD", "PSCA"]].to_numpy()
    p /= p.sum(1, keepdims=True)
    d["pH"], d["pD"], d["pA"] = p.T
    d["date"] = pd.to_datetime(d.Date, dayfirst=True, format="mixed")
    d["H"], d["A"] = (d.FTR == "H").astype(int), (d.FTR == "A").astype(int)
    d["rH"] = d.H - d.pH                       # home-win residual
    d["gd"] = d.FTHG - d.FTAG
    d["period"] = np.where(d.date < G0, "before",
                           np.where(d.date <= G1, "ghost", "after"))
    return d.reset_index(drop=True)


def se(x):
    return x.std() / np.sqrt(len(x))


def main():
    d = load()
    print(f"matches: {len(d)}  ({d.date.min():%Y-%m} to {d.date.max():%Y-%m})")

    print("\n## 1. Home-win rate vs Pinnacle close, by period")
    print("period    n      home win  market   residual (se)   home goal diff   ROI back away (se)")
    for k in ("before", "ghost", "after"):
        x = d[d.period == k]
        roi = x.A * x.PSCA - 1
        print(f"{k:7s} {len(x):6d}   {x.H.mean():.3f}    {x.pH.mean():.3f}   "
              f"{x.rH.mean():+.4f} ({se(x.rH):.4f})   {x.gd.mean():+.3f}"
              f"            {roi.mean():+.4f} ({se(roi):.4f})")

    print("\n## 2. Ghost period by month: did the residual shrink as the market learned?")
    print("month      n     home win  market   residual (se)   ROI back away (se)")
    g = d[d.period == "ghost"]
    for mth, x in g.groupby(g.date.dt.to_period("M")):
        if len(x) < 100:
            continue
        roi = x.A * x.PSCA - 1
        print(f"{mth}  {len(x):5d}    {x.H.mean():.3f}    {x.pH.mean():.3f}   "
              f"{x.rH.mean():+.4f} ({se(x.rH):.4f})   {roi.mean():+.4f} ({se(roi):.4f})")

    print("\n## 3. Ghost period in thirds (by date)")
    q = pd.qcut(g.date.rank(method="first"), 3, labels=["first", "middle", "last"])
    for k, x in g.groupby(q, observed=True):
        roi = x.A * x.PSCA - 1
        print(f"{k:6s} {x.date.min():%Y-%m-%d}..{x.date.max():%Y-%m-%d}  n={len(x):5d}  "
              f"residual {x.rH.mean():+.4f} ({se(x.rH):.4f})  ROI away {roi.mean():+.4f} ({se(roi):.4f})")

    print("\n## 4. Market-implied home edge (pH - pA) by period: how far did prices move?")
    for k in ("before", "ghost", "after"):
        x = d[d.period == k]
        print(f"{k:7s} pH-pA {(x.pH - x.pA).mean():+.4f}   actual H-A {(x.H - x.A).mean():+.4f}")


if __name__ == "__main__":
    main()
