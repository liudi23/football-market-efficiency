"""Can the bookmaker consensus predict where a SHARP price (Pinnacle, Betfair
exchange) will move before kick-off?  Kaunitz-style test on the sharp book's
early ('opening') price, scored by closing line value against its own close.
Seasons up to 2022/23 only (2023/24+ is the locked holdout)."""
import glob, re, sys
import numpy as np, pandas as pd

LOCKED = ("2324", "2425", "2526", "2627")

def devig(df, cols):
    p = 1.0 / df[cols].to_numpy(float)
    return p / p.sum(1, keepdims=True)

def load():
    out = []
    for f in sorted(glob.glob("data/raw/*.csv")):
        s = re.search(r"_(\d{4})\.csv", f).group(1)
        if s in LOCKED or int(s[:2]) > 30:      # skip 1990s files
            continue
        d = pd.read_csv(f, encoding="latin1", on_bad_lines="skip")
        d.columns = [c.lstrip("﻿").lstrip("ï»¿") for c in d.columns]
        d["season"], d["file"] = s, f
        out.append(d)
    return pd.concat(out, ignore_index=True)

def study(d, book, label):
    o = [f"{book}{x}" for x in "HDA"]; c = [f"{book}C{x}" for x in "HDA"]
    a = ["AvgH", "AvgD", "AvgA"]
    need = o + c + a + ["FTR"]
    if not all(k in d for k in need):
        return None
    x = d.dropna(subset=need)
    x = x[(x[o + c + a].astype(float) > 1.0).all(1)]
    po, pc, pa = devig(x, o), devig(x, c), devig(x, a)
    oo = x[o].to_numpy(float)
    res = np.stack([(x.FTR == k).to_numpy() for k in "HDA"], 1)
    gap = (pa - po).ravel()                   # consensus minus sharp, at 'open'
    move = (pc - po).ravel()                  # sharp close minus sharp open
    b = np.polyfit(gap, move, 1)[0]
    r = np.corrcoef(gap, move)[0, 1]
    # rule: back the sharp book at its opening price when consensus says it is
    # too long by > thr; CLV = open odds * close fair prob - 1
    rows = []
    for thr in (0.0, 0.01, 0.02, 0.03):
        m = (pa - po) > thr
        clv = (oo * pc - 1)[m]
        pnl = (oo * res - 1)[m]
        n = m.sum()
        rows.append((thr, n, clv.mean(), clv.std() / np.sqrt(n),
                     pnl.mean(), pnl.std() / np.sqrt(n)))
    print(f"\n## {label}  ({len(x)} matches, {x.season.min()}-{x.season.max()})")
    print(f"slope of sharp move on consensus gap: {b:.3f}  corr {r:.3f}")
    print("thr   bets   meanCLV   (se)     ROI      (se)")
    for t, n, cm, cs, pm, ps in rows:
        print(f"{t:.2f} {n:6d}  {cm:+.4f} ({cs:.4f})  {pm:+.4f} ({ps:.4f})")

d = load()
study(d, "PS", "Pinnacle")
study(d, "BFE", "Betfair exchange")
