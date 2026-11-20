"""Index Betfair historical (BASIC) football market files.

Betfair's historic data comes as one bz2 file per market, each line a stream
message; the first line holds the market definition (event name, market type,
kick-off time, country). This script reads only that first line of every file
and keeps markets we price (MATCH_ODDS and OVER_UNDER_05..45), writing one CSV
per day folder to data/processed/betfair_index/. It is resumable: day folders
already indexed are skipped, and `--budget` stops cleanly after N seconds so it
can be run in chunks.

Note: the folder dates in Betfair's archive are not always the match date; use
`market_time` from the file, never the folder name.

Usage:
    python scripts/betfair_index.py                         # all of data/BASIC, no time limit
    python scripts/betfair_index.py --budget 150            # stop after ~150 s, rerun to resume
    python scripts/betfair_index.py --root data/BASIC --out data/processed/betfair_index
"""
from __future__ import annotations

import argparse
import bz2
import csv
import json
import time
from pathlib import Path

KEEP = {"MATCH_ODDS", "OVER_UNDER_05", "OVER_UNDER_15", "OVER_UNDER_25",
        "OVER_UNDER_35", "OVER_UNDER_45"}
FIELDS = ["path", "event_id", "market_id", "market_type", "event_name", "market_time",
          "open_date", "country", "runners"]


def first_definition(path: Path) -> dict | None:
    with bz2.open(path, "rb") as f:
        line = f.readline()
    try:
        mc = json.loads(line)["mc"][0]
    except (ValueError, KeyError, IndexError):
        return None
    md = mc.get("marketDefinition")
    if md is None:
        return None
    md["_market_id"] = mc.get("id")
    return md


def index_day(day: Path, root: Path) -> list[dict]:
    rows = []
    for f in sorted(day.rglob("*.bz2")):
        md = first_definition(f)
        if not md or md.get("marketType") not in KEEP:
            continue
        rows.append({
            "path": str(f.relative_to(root)), "event_id": md.get("eventId"),
            "market_id": md.get("_market_id"), "market_type": md.get("marketType"),
            "event_name": md.get("eventName"), "market_time": md.get("marketTime"),
            "open_date": md.get("openDate"), "country": md.get("countryCode", ""),
            "runners": "|".join(f"{r['id']}:{r.get('name', '')}" for r in md.get("runners", [])),
        })
    return rows


def main(root="data/BASIC", out="data/processed/betfair_index", budget=None):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    days = sorted(p for p in root.glob("*/*/*") if p.is_dir())
    t0 = time.time()
    done = skipped = 0
    for day in days:
        target = out / ("_".join(day.relative_to(root).parts) + ".csv")
        if target.exists():
            skipped += 1
            continue
        if budget is not None and time.time() - t0 > budget:
            break
        rows = index_day(day, root)
        tmp = target.with_suffix(".tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS)
            w.writeheader()
            w.writerows(rows)
        tmp.rename(target)  # only complete days count as done
        done += 1
    remaining = sum(1 for d in days
                    if not (out / ("_".join(d.relative_to(root).parts) + ".csv")).exists())
    print(f"indexed {done} day folders this run, {skipped} already done, "
          f"{remaining} remaining, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/BASIC")
    ap.add_argument("--out", default="data/processed/betfair_index")
    ap.add_argument("--budget", type=float, default=None, help="seconds before stopping")
    a = ap.parse_args()
    main(a.root, a.out, a.budget)
