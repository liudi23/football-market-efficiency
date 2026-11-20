"""Build Premier League in-play market data from the Betfair archive.

Needs: data/BASIC (Betfair BASIC files), the index from scripts/betfair_index.py,
and football-data's E0 CSV for the season. Steps:
1. match Betfair events to Premier League fixtures (date + team names);
2. rebuild each match timeline (kick-off, half-time, goals, scorer, full time);
3. check the rebuilt scores against football-data's full-time and half-time scores;
4. write per-minute market probabilities with the match state.

Outputs (data/processed/, gitignored):
  betfair_epl_<season>_matches.csv   one row per match: timeline + checks
  betfair_epl_<season>_inplay.csv    one row per match-minute: score, market 1X2 and over-2.5 probabilities

Usage:  python scripts/betfair_epl.py --season 2425
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from sportsq import betfair as bf

NAME_MAP = {"Man United": "Man Utd", "Nott'm Forest": "Nottm Forest"}
TYPES = ["MATCH_ODDS"] + [f"OVER_UNDER_{k}5" for k in range(5)]


def load_index(index_dir):
    frames = [pd.read_csv(f) for f in glob.glob(f"{index_dir}/*.csv") if open(f).readline()]
    ix = pd.concat(frames, ignore_index=True)
    ix = ix[ix.country == "GB"].copy()
    ix["kick"] = pd.to_datetime(ix.market_time, utc=True)
    ix["kdate"] = ix.kick.dt.tz_convert("Europe/London").dt.date
    return ix


def pick(markets_rows, root):
    """One market file per type.

    Prefer single-market files (named 1.<id>.bz2) over event-level bundles; among
    several market files of the same type keep the one with the most price updates.
    """
    chosen = {}
    for mt, grp in markets_rows.groupby("market_type"):
        single = grp[grp.path.map(lambda p: Path(p).name.startswith("1."))]
        grp = single if len(single) else grp
        if len(grp) == 1:
            chosen[mt] = grp.path.iloc[0]
        else:
            chosen[mt] = max(grp.path, key=lambda p: len(bf.read_market(Path(root) / p).prices))
    return chosen


def main(season="2425", root="data/BASIC", index_dir="data/processed/betfair_index"):
    e0 = pd.read_csv(f"data/raw/E0_{season}.csv", encoding="latin-1")
    e0["date"] = pd.to_datetime(e0.Date, dayfirst=True)
    e0["kdate"] = e0.date.dt.date
    e0["bf_name"] = (e0.HomeTeam.map(lambda x: NAME_MAP.get(x, x)) + " v " +
                     e0.AwayTeam.map(lambda x: NAME_MAP.get(x, x)))
    ix = load_index(index_dir)
    ix = ix[ix.market_type.isin(TYPES) & ix.event_name.isin(set(e0.bf_name))]
    rows, minute_rows = [], []
    for fx in e0.itertuples(index=False):
        mk = ix[(ix.event_name == fx.bf_name) & (ix.kdate == fx.kdate)]
        if not (mk.market_type == "MATCH_ODDS").any():
            continue
        home, away = fx.bf_name.split(" v ")
        try:
            markets = bf.load_event(pick(mk, root), root)
            tl = bf.build_timeline(markets, home, away)
            tl_fit = bf.build_timeline(markets, home, away, (fx.FTHG, fx.FTAG))
        except Exception as exc:  # keep going; record the failure
            rows.append({"date": fx.date.date(), "home": fx.HomeTeam, "away": fx.AwayTeam,
                         "error": repr(exc)})
            continue
        gh = sum(g[3] == "H" for g in tl.goals)
        ga = sum(g[3] == "A" for g in tl.goals)
        # half-time check uses the timeline with sides fitted to the final score
        hth = sum(g[3] == "H" and g[2] == 1 for g in tl_fit.goals)
        hta = sum(g[3] == "A" and g[2] == 1 for g in tl_fit.goals)
        rows.append({
            "date": fx.date.date(), "home": fx.HomeTeam, "away": fx.AwayTeam,
            "kickoff": tl.kickoff, "second_half": tl.second_half, "full_time": tl.full_time,
            "ht_source": tl.ht_source, "ft_minute": tl.minute(tl.full_time)[0] if tl.full_time is not None else np.nan,
            "FTHG": fx.FTHG, "FTAG": fx.FTAG, "HTHG": fx.HTHG, "HTAG": fx.HTAG,
            "bf_hg": gh, "bf_ag": ga, "bf_hthg": hth, "bf_htag": hta,
            "total_ok": gh + ga == fx.FTHG + fx.FTAG or fx.FTHG + fx.FTAG > 5,
            "score_ok": (gh, ga) == (fx.FTHG, fx.FTAG),     # sides from price moves alone
            "ht_ok": (hth, hta) == (fx.HTHG, fx.HTAG),      # timing check (sides fitted to FT)
            "goals": ";".join(f"{g[1]}{g[3]}(p{g[2]})" for g in tl_fit.goals),
            "notes": "; ".join(tl_fit.notes),
            "PSCH": fx.PSCH, "PSCD": fx.PSCD, "PSCA": fx.PSCA,
            "PC>2.5": getattr(fx, "_asdict")().get("PC>2.5", np.nan),
            "PC<2.5": getattr(fx, "_asdict")().get("PC<2.5", np.nan),
        })
        # per-minute market probabilities (regulation minutes + stoppage up to full time)
        mo = markets["MATCH_ODDS"]
        ou = markets.get("OVER_UNDER_25")
        end2 = tl.minute(tl.full_time)[0] if tl.full_time is not None else 95
        grid = [(m, 1) for m in range(0, 46)] + [(m, 2) for m in range(45, int(end2) + 1)]
        for minute, period in grid:
            t = (tl.kickoff + pd.Timedelta(minutes=minute)) if period == 1 else \
                (tl.second_half + pd.Timedelta(minutes=minute - 45))
            if period == 1 and t >= tl.second_half - pd.Timedelta(minutes=13):
                continue  # beyond the first half (half-time interval)
            p = bf.match_probs(mo, t, home, away)
            over = np.nan
            if ou is not None:
                qo, qu = (1 / bf.price_at(ou, t, ou.runner_id("Over")),
                          1 / bf.price_at(ou, t, ou.runner_id("Under")))
                over = qo / (qo + qu)
            h, a = tl_fit.score_at(minute, period)
            last_goal = max([g[1] for g in tl_fit.goals if (g[2], g[1]) <= (period, minute)], default=None)
            minute_rows.append({
                "date": fx.date.date(), "home": fx.HomeTeam, "away": fx.AwayTeam,
                "minute": minute, "period": period, "hg": h, "ag": a,
                "mkt_h": p[0], "mkt_d": p[1], "mkt_a": p[2], "mkt_over25": over,
                "since_goal": (minute - last_goal) if last_goal is not None else np.nan,
                "FTHG": fx.FTHG, "FTAG": fx.FTAG})
    out = Path("data/processed")
    m = pd.DataFrame(rows)
    m.to_csv(out / f"betfair_epl_{season}_matches.csv", index=False)
    pd.DataFrame(minute_rows).to_csv(out / f"betfair_epl_{season}_inplay.csv", index=False)
    ok = m.dropna(subset=["score_ok"]) if "score_ok" in m else m
    print(f"matches: {len(m)}; errors: {m.error.notna().sum() if 'error' in m else 0}")
    print(f"total goals match: {ok.total_ok.mean():.1%}; full-time score from price moves alone: "
          f"{ok.score_ok.mean():.1%}; half-time score (sides fitted to FT): {ok.ht_ok.mean():.1%}")
    print("half-time source:", ok.ht_source.value_counts().to_dict())
    tot = ok[ok.total_ok]
    print(f"among matches with the right goal count: FT from price moves {tot.score_ok.mean():.1%}, "
          f"HT {tot.ht_ok.mean():.1%}")
    print(f"per-minute rows: {len(minute_rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2425")
    a = ap.parse_args()
    main(a.season)
