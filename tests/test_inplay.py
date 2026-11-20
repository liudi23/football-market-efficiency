"""Step 1 tests for the analytic in-play pricer.

Monte Carlo agreement tests are added in Step 2.
"""
import numpy as np
import pytest
from scipy.stats import binom, poisson

from sportsq import inplay
from sportsq.dixon_coles import DixonColes, grid_to_1x2, grid_to_over, score_grid
from sportsq.simulate import simulate

LH, LA, RHO = 1.5, 1.1, -0.08


def test_probabilities_sum_to_one():
    p = inplay.price([1.5, 2.2, 0.4], [1.1, 0.7, 0.3], [0, 2, 1], [1, 0, 3],
                     [10, 55, 80], rho=RHO)
    assert np.allclose(p["1x2"].sum(1), 1)
    assert ((p["over"] >= 0) & (p["over"] <= 1)).all()


def test_minute_zero_equals_prematch_dixon_coles():
    pre = score_grid(LH, LA, RHO)
    p = inplay.price(LH, LA, 0, 0, 0, rho=RHO, lines=(0.5, 1.5, 2.5, 3.5))
    assert np.allclose(p["1x2"], grid_to_1x2(pre), atol=1e-12)
    assert np.allclose(p["over"][0], [grid_to_over(pre, l)[0] for l in (0.5, 1.5, 2.5, 3.5)],
                       atol=1e-12)


@pytest.mark.parametrize("score,res", [((2, 1), [1, 0, 0]), ((1, 1), [0, 1, 0]),
                                       ((0, 3), [0, 0, 1])])
def test_minute_90_equals_final_result(score, res):
    for minute in (90, 95):  # stoppage time is clipped to 90 in Step 1
        p = inplay.price(LH, LA, *score, minute, rho=RHO)
        assert np.allclose(p["1x2"][0], res)
        total = sum(score)
        assert np.allclose(p["over"][0], [total > l for l in inplay.DEFAULT_LINES])


def test_known_values_independent_poisson():
    # Hand-checked figures (also in interview-prep/math/strategy_design.md, chain 8)
    ht = inplay.price(LH, LA, 1, 0, 45)["1x2"][0]
    assert np.allclose(ht, [0.7602, 0.1830, 0.0568], atol=1e-4)
    m75 = inplay.price(LH, LA, 1, 0, 75)["1x2"][0]
    assert np.allclose(m75, [0.8666, 0.1216, 0.0118], atol=1e-4)


def test_leader_win_probability_rises_with_time():
    mins = np.arange(0, 91, 5)
    p = inplay.price(np.full(len(mins), LH), LA, 1, 0, mins, rho=RHO)["1x2"][:, 0]
    assert (np.diff(p) > 0).all()


def test_over_line_already_settled():
    p = inplay.price(LH, LA, 2, 1, 30, rho=RHO)
    lines = np.array(inplay.DEFAULT_LINES)
    assert np.allclose(p["over"][0][lines < 3], 1.0)


def test_dc_correction_is_exact_conditioning():
    """Brute force: condition the full-match DC distribution on the score at minute m,
    assuming goal times are i.i.d. uniform given the final counts."""
    G = 25
    full = score_grid(LH, LA, RHO, max_goals=G - 1)[0]
    for (h, a, m) in [(0, 0, 30), (1, 0, 30), (0, 1, 60), (1, 1, 75), (2, 0, 20)]:
        s = m / 90
        x = np.arange(G)
        like = binom.pmf(h, x, s)[:, None] * binom.pmf(a, x, s)[None, :]
        post = full * like
        post /= post.sum()
        ours = inplay.inplay_grid(LH, LA, h, a, m, rho=RHO, max_remaining=G - 1 - max(h, a))[0]
        n = min(post.shape[0], ours.shape[0])
        assert np.allclose(ours[:n, :n], post[:n, :n], atol=1e-10), (h, a, m)


def test_rho_zero_is_independent_poisson_on_remaining_goals():
    grid = inplay.inplay_grid(LH, LA, 1, 0, 60)[0]
    r = 30 / 90
    x = np.arange(grid.shape[0])
    expect = poisson.pmf(x - 1, LH * r)[:, None] * poisson.pmf(x, LA * r)[None, :]
    assert np.allclose(grid, expect / expect.sum(), atol=1e-12)


def test_price_path_and_model_wrapper():
    df = simulate(n_seasons=2, seed=2)
    model = DixonColes(xi=0).fit(df)
    home, away = df.home.iloc[0], df.away.iloc[0]
    p0 = inplay.price_from_model(model, home, away)["1x2"]
    assert np.allclose(p0, model.predict_proba([home], [away]), atol=1e-10)
    lh, la = model.rates([home], [away])
    path = inplay.price_path(lh[0], la[0], [(23, "H"), (67, "A")], rho=model.rho_)
    assert path["home_goals"][-1] == 1 and path["away_goals"][-1] == 1
    assert np.allclose(path["1x2"][-1], [0, 1, 0])
