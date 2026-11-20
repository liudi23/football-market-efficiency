"""Tests for the Betfair BASIC parser using small synthetic stream files."""
import bz2
import json

import numpy as np
import pandas as pd
import pytest

from sportsq import betfair as bf

KO = pd.Timestamp("2024-08-16 19:00:00", tz="UTC")


def ms(minutes):
    return int((KO + pd.Timedelta(minutes=minutes)).value // 1_000_000)


def _write(path, market_id, mtype, runners, events, extra_market=None):
    """events: list of (minute, kind, payload); kind 'def' -> (status, in_play, results),
    kind 'ltp' -> {runner_id: price}."""
    lines = []
    for minute, kind, payload in events:
        if kind == "def":
            status, in_play, res = payload
            md = {"marketType": mtype, "eventName": "Home v Away", "status": status,
                  "inPlay": in_play, "marketTime": KO.isoformat(),
                  "runners": [{"id": rid, "name": name, "status": res.get(rid, "ACTIVE")}
                              for rid, name in runners.items()]}
            mc = [{"id": market_id, "marketDefinition": md}]
        else:
            mc = [{"id": market_id, "rc": [{"id": k, "ltp": v} for k, v in payload.items()]}]
        if extra_market is not None:  # bundle another market into the same message
            mc.append({"id": extra_market, "marketDefinition": {
                "marketType": "CORRECT_SCORE", "status": "CLOSED", "inPlay": True,
                "runners": [{"id": 999, "name": "1 - 0", "status": "WINNER"}]}})
        lines.append(json.dumps({"op": "mcm", "pt": ms(minute), "mc": mc}))
    with bz2.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def _match(tmp_path):
    """Home 1-0 Away: goal at 30', a VAR check at 20', half-time 47-62, full time at 110."""
    H, D, A = 1, 2, 3
    mo_runners = {H: "Home", D: "The Draw", A: "Away"}
    common = [(-60, "def", ("OPEN", False, {})), (0, "def", ("OPEN", True, {})),
              (20, "def", ("SUSPENDED", True, {})), (21, "def", ("OPEN", True, {})),
              (47, "def", ("OPEN", True, {})), (62, "def", ("OPEN", True, {}))]
    prices = [(m, "ltp", {H: 2.2, D: 3.4, A: 3.6}) for m in range(-5, 31)] + \
             [(m, "ltp", {H: 1.4, D: 4.5, A: 9.0}) for m in range(31, 110)]
    _write(tmp_path / "1.1.bz2", "1.1", "MATCH_ODDS", mo_runners, sorted(
        common + prices + [(30, "def", ("SUSPENDED", True, {})), (31, "def", ("OPEN", True, {})),
                           (110, "def", ("SUSPENDED", True, {})),
                           (112, "def", ("CLOSED", True, {H: "WINNER", D: "LOSER", A: "LOSER"}))],
        key=lambda e: e[0]))
    paths = {"MATCH_ODDS": "1.1.bz2"}
    for k in range(5):
        u, o = 10 * k + 1, 10 * k + 2
        r = {u: f"Under {k}.5 Goals", o: f"Over {k}.5 Goals"}
        ev = list(common) + [(30, "def", ("SUSPENDED", True, {}))]
        if k == 0:   # decided by the goal at 30'; a closed market has no later updates
            ev = [e for e in ev if e[0] <= 30]
            ev += [(32, "def", ("CLOSED", True, {u: "LOSER", o: "WINNER"}))]
        else:        # settled at full time
            ev += [(31, "def", ("OPEN", True, {})), (110, "def", ("SUSPENDED", True, {})),
                   (112, "def", ("CLOSED", True, {u: "WINNER", o: "LOSER"}))]
        name = f"1.{k + 2}.bz2"
        _write(tmp_path / name, f"1.{k + 2}", f"OVER_UNDER_{k}5", r,
               sorted(ev, key=lambda e: e[0]))
        paths[f"OVER_UNDER_{k}5"] = name
    return bf.load_event(paths, root=tmp_path)


def test_read_market_ignores_other_markets_in_bundles(tmp_path):
    _write(tmp_path / "1.7.bz2", "1.7", "OVER_UNDER_25", {1: "Under 2.5 Goals", 2: "Over 2.5 Goals"},
           [(0, "def", ("OPEN", True, {})), (5, "ltp", {1: 1.8, 2: 2.1})], extra_market="1.99")
    m = bf.read_market(tmp_path / "1.7.bz2")
    assert m.market_id == "1.7" and m.market_type == "OVER_UNDER_25"
    assert set(m.runners) == {1, 2} and 999 not in m.result


def test_timeline_uses_decisive_suspension_not_the_first(tmp_path):
    tl = bf.build_timeline(_match(tmp_path), "Home", "Away")
    assert tl.kickoff == KO and tl.ht_source == "definitions"
    assert tl.second_half == KO + pd.Timedelta(minutes=62)
    assert len(tl.goals) == 1                       # the 20' VAR suspension is not a goal
    t, minute, period, side, _ = tl.goals[0]
    assert t == KO + pd.Timedelta(minutes=30) and period == 1 and side == "H"
    assert minute == pytest.approx(29.5)
    assert tl.full_time == KO + pd.Timedelta(minutes=110)
    assert tl.minute(tl.full_time) == (pytest.approx(45 + 48), 2)
    assert tl.score_at(45, 1) == (1, 0) and tl.score_at(10, 1) == (0, 0)


def test_sides_can_be_fitted_to_a_known_final_score(tmp_path):
    tl = bf.build_timeline(_match(tmp_path), "Home", "Away", final_score=(1, 0))
    assert [g[3] for g in tl.goals] == ["H"] and "sides fitted to final score" in tl.notes


def test_settlement_survives_later_definitions(tmp_path):
    _write(tmp_path / "1.8.bz2", "1.8", "OVER_UNDER_05", {1: "Under 0.5 Goals", 2: "Over 0.5 Goals"},
           [(0, "def", ("OPEN", True, {})), (30, "def", ("SUSPENDED", True, {})),
            (32, "def", ("CLOSED", True, {1: "LOSER", 2: "WINNER"})),
            (40, "def", ("CLOSED", True, {}))])          # a later update without results
    assert bf.read_market(tmp_path / "1.8.bz2").result == {1: "LOSER", 2: "WINNER"}


def test_match_probs_normalised(tmp_path):
    mk = _match(tmp_path)
    p = bf.match_probs(mk["MATCH_ODDS"], KO + pd.Timedelta(minutes=10), "Home", "Away")
    assert np.isclose(p.sum(), 1) and p[0] > p[2]
