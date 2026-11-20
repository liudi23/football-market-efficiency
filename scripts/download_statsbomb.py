"""Download goal and red-card minutes from StatsBomb open data.

Source: https://github.com/statsbomb/open-data (free; attribute StatsBomb when
publishing anything built on it, per their user agreement).

Only the compact information the in-play model needs is kept, so the ~3 MB
raw event file per match is never written to disk:

  data/processed/statsbomb_matches.csv   one row per match: teams, final score,
                                          length of each half incl. stoppage time
  data/processed/statsbomb_events.csv    one row per goal or red card

StatsBomb clock: `minute` runs continuously, and the second half starts again
at 45:00. So first-half stoppage time (45+2 is minute 46 in period 1) overlaps
with early second-half minutes; always use `period` together with `minute`.
`t` below is match time in minutes counted from the start of each period's
nominal window (period 1: 0-45+, period 2: 45-90+).

Default: the four complete 2015/16 leagues (Premier League, La Liga, Serie A,
Ligue 1; 1,517 matches). Other seasons in the open data cover one club or a
tournament, which would bias goal-timing estimates; add them with --all.

Usage:
    python scripts/download_statsbomb.py            # 4 full leagues, ~4.6 GB streamed
    python scripts/download_statsbomb.py --workers 16
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
FULL_LEAGUES_2015 = [(2, 27), (11, 27), (12, 27), (7, 27)]  # (competition_id, season_id)
RED = {"Red Card", "Second Yellow"}


def get_json(path: str, retries: int = 3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(f"{BASE}/{path}", timeout=60) as r:
                return json.load(r)
        except Exception:
            if i == retries - 1:
                raise


def parse_match(m: dict) -> tuple[dict, list[dict]]:
    ev = get_json(f"events/{m['match_id']}.json")
    home = m["home_team"]["home_team_name"]
    side = lambda team: "H" if team == home else "A"
    rows, half_end = [], {}
    for e in ev:
        typ, per = e["type"]["name"], e["period"]
        t = e["minute"] + e["second"] / 60
        if typ == "Half End":
            half_end[per] = max(half_end.get(per, 0), t)
        if per > 4:  # penalty shoot-out
            continue
        team = e.get("team", {}).get("name")
        kind = None
        if typ == "Shot" and e["shot"]["outcome"]["name"] == "Goal":
            kind = "goal"
        elif typ == "Own Goal For":
            kind = "goal"
        elif typ in ("Foul Committed", "Bad Behaviour"):
            card = e.get(typ.lower().replace(" ", "_"), {}).get("card", {}).get("name")
            if card in RED:
                kind = "red"
        if kind:
            rows.append({"match_id": m["match_id"], "period": per, "minute": e["minute"],
                         "second": e["second"], "t": round(t, 3), "event": kind,
                         "side": side(team), "own_goal": typ == "Own Goal For"})
    info = {
        "match_id": m["match_id"], "date": m["match_date"],
        "competition": m["competition"]["competition_name"],
        "season": m["season"]["season_name"], "home": home,
        "away": m["away_team"]["away_team_name"],
        "hg": m["home_score"], "ag": m["away_score"],
        "end_p1": half_end.get(1), "end_p2": half_end.get(2),
    }
    return info, rows


def main(comp_seasons, out_dir="data/processed", workers=12):
    matches = []
    for cid, sid in comp_seasons:
        matches += get_json(f"matches/{cid}/{sid}.json")
    print(f"{len(matches)} matches to fetch")
    infos, events = [], []
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(parse_match, m): m["match_id"] for m in matches}
        for i, f in enumerate(as_completed(futs), 1):
            info, rows = f.result()
            infos.append(info)
            events += rows
            if i % 100 == 0:
                print(f"  {i}/{len(matches)}")
    mdf = pd.DataFrame(infos).sort_values(["date", "match_id"])
    edf = pd.DataFrame(events).sort_values(["match_id", "period", "t"])
    # consistency check: goal events must reproduce the final score
    g = edf[edf.event == "goal"].groupby(["match_id", "side"]).size().unstack(fill_value=0)
    chk = mdf.set_index("match_id")[["hg", "ag"]].join(g, how="left").fillna(0)
    bad = chk[(chk.hg != chk.get("H", 0)) | (chk.ag != chk.get("A", 0))]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mdf.to_csv(out / "statsbomb_matches.csv", index=False)
    edf.to_csv(out / "statsbomb_events.csv", index=False)
    print(f"wrote {len(mdf)} matches, {len(edf)} events to {out}/")
    print(f"goals: {(edf.event == 'goal').sum()}, red cards: {(edf.event == 'red').sum()}")
    print(f"matches where goal events != final score: {len(bad)}")
    if len(bad):
        print(bad.head(10))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="every men's competition in the open data (biased coverage)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    if a.all:
        comps = get_json("competitions.json")
        cs = [(c["competition_id"], c["season_id"]) for c in comps
              if c.get("competition_gender") == "male"]
    else:
        cs = FULL_LEAGUES_2015
    main(cs, a.out, a.workers)
