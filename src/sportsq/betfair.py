"""Betfair historical (BASIC) stream files: parse markets and rebuild match timelines.

BASIC files hold one market each: a stream of JSON messages with a timestamp
`pt` (ms, UTC), occasional `marketDefinition` updates (status OPEN/SUSPENDED/
CLOSED, `inPlay`, runner results) and last-traded prices (`ltp`) at roughly
one-minute snapshots. Folder dates in the archive are not reliable; use the
market definition.

Timeline reconstruction (no event data needed):
- **kick-off** = first snapshot with inPlay = true (up to ~1 minute late);
- **half-time / second-half start** = market-definition updates around 45-50 and
  60-70 minutes after kick-off, when present (otherwise a default gap is used);
- **goals**: every goal (and VAR checks, incidents) briefly suspends *all*
  markets, so the first suspension says little. Market OVER_UNDER_k5 is decided
  when total goals reaches k+1; if Over wins, its *last* SUSPENDED snapshot
  before it closes marks goal k+1;
- **scorer side**: the change in match-odds probabilities around each goal (the
  side whose win probability rises scored). With the final score known, sides
  are assigned to match it, ranked by that signal;
- **full time**: the last suspension before the match-odds market closes.

Times are snapshot-resolution (about a minute). Results are checked against
football-data's full-time and half-time scores in `scripts/betfair_epl.py`.
"""
from __future__ import annotations

import bz2
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Market:
    market_id: str
    market_type: str
    event_name: str
    market_time: pd.Timestamp
    runners: dict                       # runner id -> name
    prices: pd.DataFrame                # columns: t (UTC), runner, ltp
    defs: pd.DataFrame                  # columns: t, status, in_play
    result: dict = field(default_factory=dict)   # runner id -> WINNER/LOSER/...

    def first(self, status=None, in_play=None):
        d = self.defs
        if status is not None:
            d = d[d.status == status]
        if in_play is not None:
            d = d[d.in_play == in_play]
        return d.t.iloc[0] if len(d) else None

    def runner_id(self, startswith: str):
        for rid, name in self.runners.items():
            if name.lower().startswith(startswith.lower()):
                return rid
        return None


def read_market(path, market_id=None) -> Market:
    """Parse one BASIC .bz2 market file.

    Only messages for one market are kept: `market_id` if given, otherwise the
    file's name (market files are named like 1.229974294.bz2) or, failing that,
    the first market seen. Some archive files are event-level bundles holding
    every market of a match; without this filter their states get mixed up.
    """
    prices, defs = [], []
    runners, result = {}, {}
    md_last = None
    stem = Path(path).name.removesuffix(".bz2")
    want = market_id or (stem if stem.startswith("1.") else None)
    with bz2.open(path, "rb") as f:
        for line in f:
            msg = json.loads(line)
            t = msg["pt"]
            for mc in msg.get("mc", []):
                if want is None:
                    want = mc.get("id")
                if mc.get("id") != want:
                    continue
                md = mc.get("marketDefinition")
                if md is not None:
                    md_last = md
                    defs.append((t, md.get("status"), bool(md.get("inPlay"))))
                    settled = any(r.get("status") in ("WINNER", "LOSER") for r in md.get("runners", []))
                    for r in md.get("runners", []):
                        if r.get("name"):
                            runners[r["id"]] = r["name"]
                        # keep the settlement once a market is settled
                        if settled or result.get(r["id"]) not in ("WINNER", "LOSER"):
                            result[r["id"]] = r.get("status")
                for rc in mc.get("rc", []):
                    if "ltp" in rc:
                        prices.append((t, rc["id"], rc["ltp"]))
    to_ts = lambda x: pd.to_datetime(x, unit="ms", utc=True)
    p = pd.DataFrame(prices, columns=["t", "runner", "ltp"])
    p["t"] = to_ts(p.t)
    d = pd.DataFrame(defs, columns=["t", "status", "in_play"])
    d["t"] = to_ts(d.t)
    return Market(want, md_last.get("marketType"), md_last.get("eventName"),
                  pd.Timestamp(md_last.get("marketTime")), runners, p, d, result)


def price_at(m: Market, t, runner) -> float:
    """Last traded price of `runner` at or before time t (NaN if none)."""
    p = m.prices[(m.prices.runner == runner) & (m.prices.t <= t)]
    return float(p.ltp.iloc[-1]) if len(p) else np.nan


def match_probs(mo: Market, t, home: str, away: str) -> np.ndarray:
    """Normalised home/draw/away probabilities from last traded prices at t."""
    ids = [mo.runner_id(home), mo.runner_id("The Draw"), mo.runner_id(away)]
    q = np.array([1 / price_at(mo, t, i) if i is not None else np.nan for i in ids])
    return q / q.sum()


@dataclass
class Timeline:
    kickoff: pd.Timestamp
    second_half: pd.Timestamp
    full_time: pd.Timestamp | None
    goals: list            # [(utc time, match minute, period, side, price signal)]
    ht_source: str
    notes: list = field(default_factory=list)

    def minute(self, t) -> tuple[float, int]:
        """(StatsBomb-style minute, period) for a UTC time."""
        if t < self.second_half:
            return (t - self.kickoff).total_seconds() / 60, 1
        return 45 + (t - self.second_half).total_seconds() / 60, 2

    def score_at(self, minute, period):
        h = sum(1 for g in self.goals if (g[2], g[1]) <= (period, minute) and g[3] == "H")
        a = sum(1 for g in self.goals if (g[2], g[1]) <= (period, minute) and g[3] == "A")
        return h, a


def _decisive_suspension(m: Market):
    """Last in-play SUSPENDED snapshot before the market first shows CLOSED.

    Every goal (and VAR checks, incidents) suspends *all* markets briefly, so the
    first suspension is not the decisive one. A market closes shortly after it
    is decided; the suspension just before that is the deciding event.
    """
    d = m.defs[m.defs.in_play]
    closed = d[d.status == "CLOSED"]
    if not len(closed):
        return None
    before = d[(d.status == "SUSPENDED") & (d.t <= closed.t.iloc[0])]
    if len(before):
        return before.t.iloc[-1]
    return closed.t.iloc[0] - pd.Timedelta(minutes=1)


def build_timeline(markets: dict, home: str, away: str, final_score=None,
                   default_break=15.0) -> Timeline:
    """Rebuild kick-off, half-time, goals and full time from one event's markets.

    `markets` maps market type -> Market (MATCH_ODDS and OVER_UNDER_05..45).
    If `final_score` (home, away) is given, scorer sides are chosen to match it,
    using the match-odds price moves to decide which goals were whose; otherwise
    each goal's side comes from its own price move.
    """
    mo = markets["MATCH_ODDS"]
    notes = []
    kickoff = mo.first(in_play=True)
    # half-time: in-play OPEN updates that are not just a re-open after a suspension
    d = mo.defs[mo.defs.in_play].reset_index(drop=True)
    cand = []
    for i, r in d.iterrows():
        if r.status != "OPEN" or i == 0:
            continue
        prev = d.iloc[i - 1]
        if prev.status == "SUSPENDED" and (r.t - prev.t) <= pd.Timedelta(minutes=4):
            continue  # re-open after a goal / check
        cand.append(r.t)
    cand = pd.Series(cand, dtype="datetime64[ns, UTC]")
    mins = (cand - kickoff).dt.total_seconds() / 60
    ht = cand[(mins >= 44) & (mins <= 56)]
    sh = cand[(mins >= 57) & (mins <= 75)]
    if len(sh):
        second_half = sh.iloc[0]
        ht_source = "definitions" if len(ht) else "second-half only"
    else:
        second_half = kickoff + pd.Timedelta(minutes=45 + 2 + default_break)
        ht_source = "default"
        notes.append("half-time not visible; assumed 47 + 15 minutes")
    # goals: the decisive suspension of each over/under market won by Over
    goal_times, full_time = [], None
    for k in range(5):
        m = markets.get(f"OVER_UNDER_{k}5")
        if m is None:
            notes.append(f"missing OVER_UNDER_{k}5")
            continue
        over = m.runner_id("Over")
        t = _decisive_suspension(m)
        if over is not None and m.result.get(over) == "WINNER":
            if t is not None:
                goal_times.append(t)
            else:
                notes.append(f"goal {k + 1} not timed")
        elif t is not None and full_time is None:
            full_time = t
    mo_ft = _decisive_suspension(mo)
    if mo_ft is not None:
        full_time = mo_ft
    goal_times = sorted(goal_times)
    o45 = markets.get("OVER_UNDER_45")
    if o45 is not None and o45.result.get(o45.runner_id("Over")) == "WINNER":
        notes.append("more than 5 goals: goals beyond the 5th are not timed")
    tl = Timeline(kickoff, second_half, full_time, [], ht_source, notes)
    # scorer side from the match-odds move around each goal
    signal = []
    for gt in goal_times:
        before = match_probs(mo, gt - pd.Timedelta(seconds=90), home, away)
        after = match_probs(mo, gt + pd.Timedelta(minutes=3), home, away)
        signal.append((after[0] - before[0]) - (after[2] - before[2]))
    signal = np.nan_to_num(np.array(signal, float))
    if final_score is not None and len(goal_times) == sum(final_score):
        sides = np.array(["A"] * len(goal_times), dtype=object)
        sides[np.argsort(-signal)[:final_score[0]]] = "H"
        tl.notes.append("sides fitted to final score")
    else:
        sides = np.where(signal > 0, "H", "A")
    for gt, side, sig in zip(goal_times, sides, signal):
        minute, period = tl.minute(gt - pd.Timedelta(seconds=30))  # goal is before the snapshot
        tl.goals.append((gt, round(minute, 1), period, side, round(float(sig), 3)))
    return tl


def load_event(paths: dict, root="data/BASIC") -> dict:
    """market type -> Market for one event, from {market_type: relative path}."""
    return {mt: read_market(Path(root) / p) for mt, p in paths.items()}
