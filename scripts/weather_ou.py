"""Does weather predict goals beyond the closing over/under 2.5 price?

English divisions E0-E3, 2019/20-2022/23 (2023/24+ is the locked holdout).
Market = Pinnacle closing over/under 2.5, margin removed (Avg close if missing).
Weather = Open-Meteo hourly reanalysis at each ground (scripts/fetch_weather.py).

Two feature sets:
  known : kick-off hour wind/gust/temperature, rain in the 3 hours up to kick-off
          (a short-range forecast at the close would know roughly this)
  match : wind and rain over the first two match hours (upper bound; partly
          unknowable at the close)

Fit on 2019/20-2020/21, test on 2021/22-2022/23.
  python3 scripts/weather_ou.py              # real weather
  python3 scripts/weather_ou.py --synthetic  # fake weather with an injected
                                              # effect, to check the pipeline
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
DIVS = ["E0", "E1", "E2", "E3"]
TRAIN, TEST = ("1920", "2021"), ("2122", "2223")
SETS = {"known": ["wind_ko", "rain_pre", "temp_ko"],   # gusts ~ collinear with wind
        "match": ["wind_m", "rain_m", "temp_ko"]}
SCALE = {"wind_ko": 10, "gust_ko": 10, "wind_m": 10, "temp_ko": 10,
         "rain_pre": 1, "rain_m": 1}


def load_matches():
    rows = []
    for d in DIVS:
        for s in TRAIN + TEST:
            x = pd.read_csv(ROOT / f"data/raw/{d}_{s}.csv", encoding="latin1",
                            on_bad_lines="skip")
            x.columns = [c.replace("﻿", "").replace("ï»¿", "") for c in x.columns]
            x["div"], x["season"] = d, s
            rows.append(x)
    m = pd.concat(rows, ignore_index=True)
    num = ["PC>2.5", "PC<2.5", "AvgC>2.5", "AvgC<2.5", "MaxC>2.5", "MaxC<2.5",
           "FTHG", "FTAG"]
    m[num] = m[num].apply(pd.to_numeric, errors="coerce")
    for side in (">", "<"):        # fall back to the average close if no Pinnacle
        m[f"PC{side}2.5"] = m[f"PC{side}2.5"].fillna(m[f"AvgC{side}2.5"])
    m = m.dropna(subset=["PC>2.5", "PC<2.5", "FTHG", "FTAG", "Time", "Date"])
    po, pu = 1 / m["PC>2.5"], 1 / m["PC<2.5"]
    m["p_mkt"] = po / (po + pu)
    m["over"] = ((m.FTHG + m.FTAG) >= 3).astype(int)
    m["goals"] = m.FTHG + m.FTAG
    ko = pd.to_datetime(m.Date + " " + m.Time, dayfirst=True, format="mixed")
    m["ko"] = ko.dt.round("h")
    return m.reset_index(drop=True)


def attach_weather(m):
    wdir = ROOT / "data/external/weather"
    feats = []
    missing = set()
    cache = {}
    for i, r in m.iterrows():
        key = r.HomeTeam.replace("'", "").replace(" ", "_")
        if key not in cache:
            f = wdir / f"{key}.csv"
            if not f.exists():
                missing.add(r.HomeTeam); cache[key] = None
            else:
                w = pd.read_csv(f, parse_dates=["time"]).set_index("time")
                cache[key] = w[~w.index.duplicated()]   # clock-change hour
        w = cache[key]
        if w is None or r.ko not in w.index:
            feats.append([np.nan] * 6); continue
        j = w.index.get_loc(r.ko)
        pre, mat = w.iloc[max(j - 2, 0): j + 1], w.iloc[j: j + 2]
        feats.append([w.wind_speed_10m.iloc[j], w.wind_gusts_10m.iloc[j],
                      w.temperature_2m.iloc[j], pre.precipitation.sum(),
                      mat.wind_speed_10m.mean(), mat.precipitation.sum()])
    f = pd.DataFrame(feats, columns=["wind_ko", "gust_ko", "temp_ko", "rain_pre",
                                     "wind_m", "rain_m"])
    if missing:
        print(f"WARNING no weather file for: {sorted(missing)}")
    return pd.concat([m, f], axis=1)


def synthetic(m, seed=0, effect=-0.08):
    """Fake weather; over/under drawn from the market price plus `effect`
    per 10 km/h of kick-off wind above 15 km/h."""
    rng = np.random.default_rng(seed)
    n = len(m)
    m = m.copy()
    m["wind_ko"] = rng.gamma(4, 4, n)
    m["gust_ko"] = m.wind_ko * 1.6 + rng.normal(0, 3, n)
    m["wind_m"] = m.wind_ko + rng.normal(0, 2, n)
    m["temp_ko"] = rng.normal(10, 5, n)
    m["rain_pre"] = rng.exponential(0.5, n) * (rng.random(n) < 0.3)
    m["rain_m"] = m.rain_pre * rng.uniform(0, 1.5, n)
    lo = np.log(m.p_mkt / (1 - m.p_mkt)) + effect * (m.wind_ko - 15) / 10
    m["over"] = (rng.random(n) < 1 / (1 + np.exp(-lo))).astype(int)
    print(f"SYNTHETIC: injected {effect:+.2f} log-odds per 10 km/h wind")
    return m


def design(m, cols):
    z = np.log(m.p_mkt / (1 - m.p_mkt)).to_numpy()
    X = [np.ones(len(m)), z] + [(m[c] / SCALE[c]).to_numpy() for c in cols]
    return np.column_stack(X)


def fit_logit(X, y, ridge=1e-6):
    def nll(b):
        e = X @ b
        return np.sum(np.logaddexp(0, e) - y * e) + ridge * b @ b
    def grad(b):
        p = 1 / (1 + np.exp(-(X @ b)))
        return X.T @ (p - y) + 2 * ridge * b
    b0 = np.zeros(X.shape[1]); b0[1] = 1.0
    b = minimize(nll, b0, jac=grad, method="BFGS").x
    p = 1 / (1 + np.exp(-(X @ b)))
    H = X.T @ (X * (p * (1 - p))[:, None])
    return b, np.sqrt(np.diag(np.linalg.inv(H))), nll(b)


def logloss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    syn = "--synthetic" in sys.argv
    m = load_matches()
    m = synthetic(m) if syn else attach_weather(m)
    m = m.dropna(subset=SETS["known"] + SETS["match"]).reset_index(drop=True)
    assert not m.season.isin(["2324", "2425", "2526", "2627"]).any()  # holdout
    tr, te = m.season.isin(TRAIN).to_numpy(), m.season.isin(TEST).to_numpy()
    y = m.over.to_numpy()
    print(f"matches with weather: {len(m)} (train {tr.sum()}, test {te.sum()})")

    print("\n## 1. Over-2.5 rate vs closing price, by weather bucket (all seasons)")
    print("bucket                    n   over   market   diff    (se)   goals")
    buckets = {
        "wind_ko < 15 km/h": m.wind_ko < 15,
        "wind_ko 15-25": (m.wind_ko >= 15) & (m.wind_ko < 25),
        "wind_ko 25-35": (m.wind_ko >= 25) & (m.wind_ko < 35),
        "wind_ko >= 35": m.wind_ko >= 35,
        "rain_pre = 0": m.rain_pre == 0,
        "rain_pre 0-2 mm": (m.rain_pre > 0) & (m.rain_pre < 2),
        "rain_pre >= 2 mm": m.rain_pre >= 2,
        "temp_ko < 5 C": m.temp_ko < 5,
        "temp_ko >= 20 C": m.temp_ko >= 20,
    }
    for name, b in buckets.items():
        b = b.to_numpy(); n = b.sum()
        if n == 0:
            continue
        d = y[b] - m.p_mkt.to_numpy()[b]
        print(f"{name:22s} {n:5d}  {y[b].mean():.3f}   {m.p_mkt[b].mean():.3f}  "
              f"{d.mean():+.3f}  ({d.std()/np.sqrt(n):.3f})  {m.goals[b].mean():.2f}")

    print("\n## 2. Logistic: logit P(over) = c + a*logit(market) + weather (train)")
    base_X = design(m, [])
    b0, _, nll0 = fit_logit(base_X[tr], y[tr])
    for name, cols in SETS.items():
        X = design(m, cols)
        b, se, nll1 = fit_logit(X[tr], y[tr])
        lr = 2 * (nll0 - nll1)
        print(f"[{name}]  LR stat {lr:.2f} on {len(cols)} df")
        for c, bi, si in zip(["const", "market"] + cols, b, se):
            unit = f"per {SCALE[c]}" if c in SCALE else ""
            print(f"   {c:9s} {bi:+.3f} ({si:.3f}) {unit}")
        # out of sample
        p_base = 1 / (1 + np.exp(-(base_X[te] @ b0)))
        p_w = 1 / (1 + np.exp(-(X[te] @ b)))
        p_m = m.p_mkt.to_numpy()[te]
        dl = logloss(p_w, y[te]) - logloss(p_base, y[te])
        dm = logloss(p_w, y[te]) - logloss(p_m, y[te])
        print(f"   test log loss  weather-recalibrated minus recalibrated market: "
              f"{dl.mean():+.5f} ({dl.std()/np.sqrt(te.sum()):.5f})")
        print(f"   test log loss  weather model minus raw market:                 "
              f"{dm.mean():+.5f} ({dm.std()/np.sqrt(te.sum()):.5f})")

        # betting on the test seasons at closing odds
        mt = m[te]
        for book in ("PC", "MaxC"):
            oo, ou = mt[f"{book}>2.5"].to_numpy(), mt[f"{book}<2.5"].to_numpy()
            ok = ~(np.isnan(oo) | np.isnan(ou))
            ev_o, ev_u = p_w * oo - 1, (1 - p_w) * ou - 1
            for thr in (0.0, 0.02, 0.04):
                bo, bu = ok & (ev_o > thr), ok & (ev_u > thr) & ~(ev_o > thr)
                pnl = np.concatenate([(y[te][bo] * oo[bo]) - 1,
                                      ((1 - y[te][bu]) * ou[bu]) - 1])
                n = len(pnl)
                if n >= 30:
                    print(f"   bet at {book:4s} close, EV>{thr:.2f}: {n:5d} bets "
                          f"({bo.sum()} over / {bu.sum()} under)  ROI {pnl.mean():+.4f} "
                          f"({pnl.std()/np.sqrt(n):.4f})")

    print("\n## 3. Wind effect by division (all seasons, 'known' wind only)")
    for d in DIVS:
        k = (m["div"] == d).to_numpy()
        X = design(m[k], ["wind_ko"])
        b, se, _ = fit_logit(X, y[k])
        print(f"   {d}: {b[2]:+.3f} ({se[2]:.3f}) log-odds per 10 km/h   n={k.sum()}")


if __name__ == "__main__":
    main()
