"""Which bookmakers lead Pinnacle?  For each book, test whether its early price
predicts Pinnacle's early->close move, and the reverse (does the book move to
Pinnacle?).  Pick books on 2019/20-2020/21, test on 2021/22-2022/23.
Seasons 2023/24+ are the locked holdout and are never read."""
import glob, re
import numpy as np, pandas as pd

TRAIN, TEST = ("1920", "2021"), ("2122", "2223")
BOOKS = ["B365", "BW", "IW", "VC", "WH", "Avg", "Max"]

def devig(x, cols):
    p = 1.0 / x[cols].to_numpy(float)
    return p / p.sum(1, keepdims=True)

def load():
    out = []
    for f in sorted(glob.glob("data/raw/*.csv")):
        s = re.search(r"_(\d{4})\.csv", f).group(1)
        if s not in TRAIN + TEST:
            continue
        d = pd.read_csv(f, encoding="latin1", on_bad_lines="skip")
        d.columns = [c.replace("﻿", "").replace("ï»¿", "") for c in d.columns]
        d["season"] = s
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    cols = [f"{b}{c}{x}" for b in BOOKS + ["PS"] for c in ("", "C") for x in "HDA"]
    d[cols] = d[cols].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=cols + ["FTR"])
    return d[(d[cols].astype(float) > 1.0).all(1)].reset_index(drop=True)

d = load()
P = {(b, c): devig(d, [f"{b}{c}{x}" for x in "HDA"]) for b in BOOKS + ["PS"] for c in ("", "C")}
odds_ps = d[["PSH", "PSD", "PSA"]].to_numpy(float)
res = np.stack([(d.FTR == k).to_numpy() for k in "HDA"], 1)
tr, te = d.season.isin(TRAIN).to_numpy(), d.season.isin(TEST).to_numpy()
move = P["PS", "C"] - P["PS", ""]
print(f"matches: train {tr.sum()}, test {te.sum()}")

print("\n## 1. Who leads?  gap = book early - Pinnacle early (prob points)")
print("book   PS closes gap  (corr)   book closes gap   mean|gap|")
for b in BOOKS:
    gap = P[b, ""] - P["PS", ""]
    bmove = P[b, "C"] - P[b, ""]
    m = tr
    s1 = np.polyfit(gap[m].ravel(), move[m].ravel(), 1)[0]
    r1 = np.corrcoef(gap[m].ravel(), move[m].ravel())[0, 1]
    s2 = np.polyfit(-gap[m].ravel(), bmove[m].ravel(), 1)[0]
    print(f"{b:5s}  {s1:6.3f}       ({r1:.3f})   {s2:6.3f}          {np.abs(gap[m]).mean()*100:.2f}")

print("\n## 2. Multivariate: Pinnacle move ~ all book gaps (fit on train)")
named = ["B365", "BW", "IW", "VC", "WH", "Avg"]
X = lambda m: np.stack([(P[b, ""] - P["PS", ""])[m].ravel() for b in named], 1)
y = lambda m: move[m].ravel()
beta, *_ = np.linalg.lstsq(X(tr), y(tr), rcond=None)
for b, w in zip(named, beta):
    print(f"  {b:5s} {w:+.3f}")
pred_te = X(te) @ beta
r2 = 1 - ((y(te) - pred_te) ** 2).sum() / ((y(te) - y(te).mean()) ** 2).sum()
r2_avg = None
g = (P["Avg", ""] - P["PS", ""])
ba = np.polyfit(g[tr].ravel(), move[tr].ravel(), 1)
pa = np.polyval(ba, g[te].ravel())
r2_avg = 1 - ((y(te) - pa) ** 2).sum() / ((y(te) - y(te).mean()) ** 2).sum()
print(f"test R^2 of Pinnacle move: all books {r2:.4f}  vs Avg only {r2_avg:.4f}")

print("\n## 3. Betting Pinnacle early price on predicted move (test seasons only)")
print("signal     thr   bets   meanCLV   (se)      ROI      (se)")
clv_all = (odds_ps * P["PS", "C"] - 1)[te].ravel()
pnl_all = (odds_ps * res - 1)[te].ravel()
for name, pr in (("Avg only", pa), ("all books", pred_te)):
    for thr in (0.0, 0.005, 0.01, 0.015, 0.02):
        m = pr > thr
        n = m.sum()
        if n < 20:
            print(f"{name:9s} {thr:.3f} {n:6d}  (too few)")
            continue
        c, p = clv_all[m], pnl_all[m]
        print(f"{name:9s} {thr:.3f} {n:6d}  {c.mean():+.4f} ({c.std()/np.sqrt(n):.4f})  "
              f"{p.mean():+.4f} ({p.std()/np.sqrt(n):.4f})")
print(f"baseline (all outcomes): CLV {clv_all.mean():+.4f}, ROI {pnl_all.mean():+.4f}")
