import numpy as np
import pandas as pd
from scipy.optimize import check_grad

from sportsq.xg import RateModel, match_understat


def _fixtures(n_teams=8, rounds=30, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    att = rng.normal(0, 0.3, n_teams); att -= att.mean()
    dfn = rng.normal(0, 0.3, n_teams); dfn -= dfn.mean()
    rows = []
    for r in range(rounds):
        perm = rng.permutation(n_teams)
        for i in range(0, n_teams, 2):
            h, a = perm[i], perm[i + 1]
            lh = np.exp(0.1 + 0.25 + att[h] - dfn[a]); la = np.exp(0.1 + att[a] - dfn[h])
            rows.append((teams[h], teams[a], lh, la))
    d = pd.DataFrame(rows, columns=["home", "away", "lh", "la"])
    d["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(len(d)) // 4, "D")
    return d, att, dfn, rng


def test_gradient_is_exact():
    d, _, _, rng = _fixtures()
    m = RateModel()
    m.teams_ = {t: i for i, t in enumerate(sorted(set(d.home) | set(d.away)))}
    m.n_ = len(m.teams_)
    h, a = d.home.map(m.teams_).to_numpy(), d.away.map(m.teams_).to_numpy()
    x, y, w = rng.gamma(2, .7, len(d)), rng.gamma(2, .6, len(d)), rng.uniform(.2, 1, len(d))
    f = m._objective(h, a, x, y, w)
    th = rng.normal(0, 0.2, 2 * m.n_)
    assert check_grad(lambda t: f(t)[0], lambda t: f(t)[1], th) < 1e-4


def test_recovers_strengths_from_true_rates():
    d, att, dfn, _ = _fixtures(rounds=60)
    m = RateModel(xi=0).fit(d.home, d.away, d.lh, d.la, d.date, d.date.max())
    order = [m.teams_[f"T{i}"] for i in range(len(att))]
    assert np.allclose(m.att_[order], att, atol=1e-4)
    assert np.allclose(m.def_[order], dfn, atol=1e-4)
    assert abs(m.home_ - 0.25) < 1e-4


def test_match_understat_learns_names():
    fd = pd.DataFrame({"league": "E0", "date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15", "2020-01-22"]),
                       "home": ["Man United", "Spurs", "Man United", "Spurs"],
                       "away": ["Spurs", "Man United", "Spurs", "Man United"],
                       "hg": [1, 2, 0, 3], "ag": [0, 2, 0, 1]})
    us = pd.DataFrame({"league": "E0", "date": pd.to_datetime(["2020-01-01 20:00", "2020-01-08 15:00", "2020-01-16 15:00", "2020-01-22 15:00"]),
                       "home_us": ["Manchester United", "Tottenham", "Manchester United", "Tottenham"],
                       "away_us": ["Tottenham", "Manchester United", "Tottenham", "Manchester United"],
                       "hg_us": [1, 2, 0, 3], "ag_us": [0, 2, 0, 1], "hxg": [1.1, 2.0, .3, 2.5], "axg": [.4, 1.9, .2, .9]})
    m, names, missing = match_understat(fd, us)
    assert names[("E0", "Manchester United")] == "Man United" and missing == 0
    assert m.hxg.tolist() == [1.1, 2.0, .3, 2.5]          # third row matched with a 1-day shift
