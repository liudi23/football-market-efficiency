import numpy as np
import pandas as pd
import pytest

from sportsq import implied, metrics
from sportsq.backtest import MarketBlend
from sportsq.dixon_coles import DixonColes
from sportsq.ratings import Elo, OrderedLogit, PiRatings
from sportsq.simulate import simulate
from sportsq.staking import kelly_fraction


# ---------- implied probabilities
@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_implied_sums_to_one(method):
    odds = np.array([[2.10, 3.40, 3.60], [1.25, 6.0, 12.0]])
    p = implied.implied(odds, method)
    assert np.allclose(p.sum(1), 1)
    assert (p > 0).all()


def test_overround():
    assert implied.overround([2.10, 3.40, 3.60]) == pytest.approx(0.0477, abs=1e-3)


def test_shin_shrinks_longshot_more_than_multiplicative():
    odds = [1.25, 6.0, 12.0]
    assert implied.shin(odds)[2] < implied.multiplicative(odds)[2]


def test_no_margin_is_identity():
    p = np.array([0.5, 0.3, 0.2])
    for m in ["multiplicative", "power", "shin"]:
        assert np.allclose(implied.implied(1 / p, m), p)


# ---------- metrics
def test_rps_known_values():
    assert metrics.rps([[1, 0, 0]], [0])[0] == 0
    assert metrics.rps([[0, 0, 1]], [0])[0] == 1
    # predicting draw when home wins is half as bad as predicting away
    assert metrics.rps([[0, 1, 0]], [0])[0] == pytest.approx(0.5)


def test_bootstrap_detects_better_model():
    rng = np.random.default_rng(1)
    a = rng.normal(0.20, 0.05, 2000)
    b = a + 0.01
    r = metrics.paired_bootstrap(a, b)
    assert r["ci_high"] < 0


# ---------- ratings: no lookahead
def test_ratings_are_pre_match():
    df = simulate(n_seasons=1, seed=3)
    for model in (Elo(), PiRatings()):
        f1 = model.features(df)
        df2 = df.copy()
        df2.loc[df2.index[-1], ["hg", "ag"]] = [9, 0]  # change last result
        f2 = model.features(df2)
        assert f1.iloc[-1] == f2.iloc[-1]


def test_ordered_logit_monotone():
    df = simulate(n_seasons=3, seed=4)
    x = PiRatings().features(df)
    y = metrics.encode(df.result)
    ol = OrderedLogit().fit(x, y)
    p = ol.predict_proba(np.array([-2.0, 0.0, 2.0]))
    assert np.allclose(p.sum(1), 1)
    assert p[0, 0] < p[1, 0] < p[2, 0]  # home-win prob rises with rating gap


# ---------- Dixon-Coles parameter recovery on a static league
def test_dixon_coles_recovers_home_advantage():
    df = simulate(n_seasons=4, drift_sd=0.0, mean_revert=0.0, seed=7)
    m = DixonColes(xi=0).fit(df)
    assert m.home_ == pytest.approx(0.25, abs=0.06)
    assert m.mu_ == pytest.approx(0.18, abs=0.08)
    p = m.predict_proba(df.home, df.away)
    truth = df[["true_h", "true_d", "true_a"]].to_numpy()
    assert np.abs(p - truth).mean() < 0.03


# ---------- blend learns to trust the better source
def test_blend_prefers_truthful_source():
    df = simulate(n_seasons=3, seed=5)
    y = metrics.encode(df.result)
    truth = df[["true_h", "true_d", "true_a"]].to_numpy()
    noise = np.random.default_rng(0).dirichlet([2, 2, 2], len(df))
    w = MarketBlend().fit(truth, noise, y).w_
    assert w[0] > 0.7 and abs(w[1]) < 0.2


def test_kelly():
    assert kelly_fraction(0.5, 2.0) == 0
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
