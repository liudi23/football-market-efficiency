"""Tests for in-play pricing.

Step 1: analytic pricer. Step 2: Monte Carlo simulator agrees with it.
"""
import numpy as np
import pytest
import pandas as pd
from scipy.stats import binom, poisson

from sportsq import inplay
from sportsq.dixon_coles import DixonColes, grid_to_1x2, grid_to_over, score_grid
from sportsq.simulate import simulate

LH, LA, RHO = 1.5, 1.1, -0.08


def _compare_mc(est, se, exact, n_sims, min_count=30):
    """z-scores for cells with enough expected hits; exact Poisson check for rare ones.

    The normal approximation behind z = (MC - exact) / se fails when an event is
    expected only a handful of times in n_sims draws, so those cells are checked
    instead by asking whether the observed count is plausible under
    Poisson(expected count) (two-sided p > 1e-4).
    """
    est, se, exact = map(np.asarray, (est, se, exact))
    p_small = np.minimum(exact, 1 - exact)
    obs_small = np.where(exact <= 0.5, est, 1 - est) * n_sims
    rare = p_small * n_sims < min_count
    for mu, k in zip(p_small[rare] * n_sims, np.round(obs_small[rare])):
        if mu < 1e-12:
            assert k == 0
        else:
            pval = 2 * min(poisson.cdf(k, mu), poisson.sf(k - 1, mu))
            assert pval > 1e-4, (mu, k)
    return (est[~rare] - exact[~rare]) / se[~rare]


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


# ---------------------------------------------------------------------------
# Step 2: Monte Carlo simulator reproduces the analytic prices
# ---------------------------------------------------------------------------
N_SIMS = 200_000
STATES = [(0, 0, 0), (1, 0, 30), (0, 1, 60), (1, 1, 75), (2, 0, 45), (0, 0, 89),
          (0, 0, 60), (3, 2, 20)]
RATES = [(1.5, 1.1), (2.6, 0.6)]


def _z_scores(rho, seed0=100, n_sims=N_SIMS, weighted=True):
    """z = (MC - analytic) / MC standard error, over all states, rates and markets."""
    zs = []
    for i, (h, a, m) in enumerate(STATES):
        for lh, la in RATES:
            mc = inplay.mc_price(lh, la, h, a, m, n_sims, rho=rho if weighted else 0.0,
                                 seed=seed0 + i)
            ex = inplay.price(lh, la, h, a, m, rho=rho)
            est = np.r_[mc["1x2"], mc["over"]]
            se = np.r_[mc["1x2_se"], mc["over_se"]]
            exact = np.r_[ex["1x2"][0], ex["over"][0]]
            zs.extend(_compare_mc(est, se, exact, n_sims))
    return np.array(zs)


@pytest.mark.parametrize("rho", [0.0, -0.08, 0.1])
def test_monte_carlo_matches_analytic(rho):
    z = _z_scores(rho)
    # Tolerances were set from 90 runs with other seeds (max |z| 3.4, mean z in
    # [-0.39, 0.23], mean z^2 in [0.57, 1.41]). z-scores are correlated: the 1X2
    # outcomes and nested over lines share the same simulated matches, so the
    # pooled statistics vary more than for independent z's. These bounds still
    # catch a factor-2 error in the standard errors (mean z^2 of 4 or 0.25) and
    # any bias of the size in the negative control below.
    assert len(z) > 80
    assert np.abs(z).max() < 4.5              # no single price is off
    assert abs(z.mean()) < 0.6                # no systematic bias
    assert 0.4 < (z ** 2).mean() < 1.8        # standard errors are honest


def test_negative_control_missing_dc_weights_is_detected():
    """Simulating plain Poisson but comparing with rho != 0 prices must fail loudly."""
    z = _z_scores(-0.15, weighted=False)
    assert np.abs(z).max() > 6


def test_mc_minute_zero_equals_prematch_and_full_time_is_settled():
    mc = inplay.mc_price(LH, LA, 0, 0, 0, N_SIMS, rho=RHO, seed=5)
    pre = grid_to_1x2(score_grid(LH, LA, RHO))[0]
    assert (np.abs(mc["1x2"] - pre) < 4 * mc["1x2_se"]).all()
    done = inplay.mc_price(LH, LA, 2, 1, 90, 1000, rho=RHO, seed=5)
    assert np.allclose(done["1x2"], [1, 0, 0]) and np.allclose(done["1x2_se"], 0)


def test_thinning_with_loose_bound_gives_same_answer():
    """Custom rate_fn with a bound 3x too high: rejections happen, answer unchanged."""
    rate_fn, bound = inplay.constant_rates(LH, LA)
    mc = inplay.mc_price(LH, LA, 1, 0, 30, N_SIMS, seed=9, rate_fn=rate_fn,
                         rate_bound=3 * bound)
    ex = inplay.price(LH, LA, 1, 0, 30)["1x2"][0]
    assert (np.abs(mc["1x2"] - ex) < 4 * mc["1x2_se"]).all()


def test_rate_bound_violation_raises():
    rate_fn, bound = inplay.constant_rates(LH, LA)
    with pytest.raises(ValueError):
        inplay.simulate_remaining(LH, LA, n_sims=100, seed=0, rate_fn=rate_fn,
                                  rate_bound=bound / 2)


def test_simulation_is_reproducible_with_seed():
    a = inplay.simulate_remaining(LH, LA, 0, 0, 10, 1000, seed=42)
    b = inplay.simulate_remaining(LH, LA, 0, 0, 10, 1000, seed=42)
    assert (a["home"] == b["home"]).all() and (a["away"] == b["away"]).all()


# ---------------------------------------------------------------------------
# Step 3: time-varying intensity and stoppage time
# ---------------------------------------------------------------------------
def _toy_profile():
    """Rising intensity with random stoppage, built by hand (no data files needed)."""
    w = np.array([0.8, 0.9, 1.0, 1.3, 1.0, 1.05, 1.15, 1.25])
    prof = inplay.GoalIntensity([0, 15, 30, 45], [45, 60, 75, 90], w,
                                s1=[0.0, 1.0, 2.5, 4.0], s2=[2.0, 3.5, 4.0, 6.5],
                                probs=[0.2, 0.4, 0.3, 0.1])
    prof.weights = prof.weights / prof.total()
    return prof


TV_STATES = [(0, 0, 0, 1), (1, 0, 30, 1), (0, 0, 46, 1), (1, 1, 45, 2), (0, 1, 70, 2),
             (2, 1, 85, 2), (0, 0, 91, 2), (1, 0, 93.5, 2)]


def test_constant_profile_reproduces_step1():
    prof = inplay.GoalIntensity.constant()
    for h, a, m in [(0, 0, 0), (1, 0, 30), (0, 2, 60), (1, 1, 89)]:
        for rho in (0.0, RHO):
            tv = inplay.price_tv(prof, LH, LA, h, a, m, rho=rho)
            c = inplay.price(LH, LA, h, a, m, rho=rho)
            assert np.allclose(tv["1x2"], c["1x2"], atol=1e-12)
            assert np.allclose(tv["over"], c["over"], atol=1e-12)


def test_normalisation_keeps_prematch_expected_goals():
    prof = _toy_profile()
    assert prof.total() == pytest.approx(1.0)
    r, w = prof.remaining_share(1, 0.0)
    assert w @ r == pytest.approx(1.0)


def test_mean_stoppage_at_kickoff_equals_prematch_exactly():
    prof = _toy_profile().with_mean_stoppage()
    p = inplay.price_tv(prof, LH, LA, 0, 0, 0, period=1, rho=RHO)
    assert np.allclose(p["1x2"], grid_to_1x2(score_grid(LH, LA, RHO)), atol=1e-12)


def test_random_stoppage_at_kickoff_is_close_to_prematch():
    # mixing over stoppage lengths adds slight overdispersion, nothing more
    p = inplay.price_tv(_toy_profile(), LH, LA, 0, 0, 0, period=1, rho=RHO)
    assert np.allclose(p["1x2"], grid_to_1x2(score_grid(LH, LA, RHO)), atol=2e-3)


def test_stoppage_time_is_priced_and_match_ends_after_longest_stoppage():
    prof = _toy_profile()
    at90 = inplay.price_tv(prof, LH, LA, 0, 0, 90, period=2)["1x2"][0]
    assert 0.0 < at90[0] < 0.1 and at90[1] < 1.0  # a late goal is still possible
    later = inplay.price_tv(prof, LH, LA, 0, 0, 95, period=2)["1x2"][0]
    assert later[1] > at90[1]                       # draw firms as stoppage runs on
    over = inplay.price_tv(prof, LH, LA, 2, 1, 97, period=2)["1x2"][0]
    assert np.allclose(over, [1, 0, 0])             # beyond every scenario: settled


def test_first_half_stoppage_conditions_on_time_played():
    prof = _toy_profile()
    r46, w46 = prof.remaining_share(1, 46.0)
    assert (prof.s1[prof.scenario_weights(1, 46.0) > 0] >= 1.0).all()
    assert w46.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("rho", [0.0, -0.08])
def test_tv_monte_carlo_matches_tv_analytic(rho):
    prof = _toy_profile()
    zs = []
    for i, (h, a, m, p) in enumerate(TV_STATES):
        mc = inplay.mc_price_tv(prof, LH, LA, h, a, m, p, N_SIMS, rho=rho, seed=300 + i)
        ex = inplay.price_tv(prof, LH, LA, h, a, m, p, rho=rho)
        est = np.r_[mc["1x2"], mc["over"]]
        se = np.r_[mc["1x2_se"], mc["over_se"]]
        exact = np.r_[ex["1x2"][0], ex["over"][0]]
        zs.extend(_compare_mc(est, se, exact, N_SIMS))
    z = np.array(zs)
    assert np.abs(z).max() < 4.5 and abs(z.mean()) < 0.6 and 0.4 < (z ** 2).mean() < 1.8


def test_fit_recovers_known_intensity_shape():
    """Simulate goal times from a known shape, fit, and recover the weights."""
    rng = np.random.default_rng(0)
    true = _toy_profile()
    n = 6000
    sc = rng.choice(len(true.probs), n, p=true.probs)
    b = true.bounds(true.s1[sc], true.s2[sc])
    rows, mrows = [], []
    for i in range(n):
        lam = 2.6 * true.weights * np.diff(b[i])          # expected goals per bin
        for k, c in enumerate(rng.poisson(lam)):
            u = rng.uniform(b[i, k], b[i, k + 1], c)
            for x in u:
                end1 = b[i, len(true.edges1)]
                per, t = (1, x) if x < end1 else (2, 45 + (x - end1))
                rows.append({"match_id": i, "period": per, "t": t, "event": "goal"})
        mrows.append({"match_id": i, "end_p1": 45 + true.s1[sc[i]],
                      "end_p2": 90 + true.s2[sc[i]]})
    fit = inplay.fit_goal_intensity(pd.DataFrame(mrows), pd.DataFrame(rows))
    assert fit.total() == pytest.approx(1.0)
    rel_err = np.abs(fit.weights - true.weights) / true.weights
    assert rel_err[[0, 1, 2, 4, 5, 6]].max() < 0.08    # regulation bins: lots of exposure
    assert fit.table.attrs["goals_per_match"] == pytest.approx(2.6, rel=0.03)


# ---------------------------------------------------------------------------
# Step 4: score effects
# ---------------------------------------------------------------------------
TOY_MULT = np.array([1.4, 1.2, 1.0, 0.85, 0.7])  # strong effects so tests have power


def _toy_effects(mult=TOY_MULT):
    return inplay.ScoreEffects(np.asarray(mult, float), _toy_profile())


def test_no_score_effects_reproduces_step3():
    eff = inplay.ScoreEffects.none(_toy_profile())
    for h, a, m, p in TV_STATES:
        for rho in (0.0, RHO):
            se = inplay.price_se(eff, LH, LA, h, a, m, p, rho=rho, scale=(1.0, 1.0))
            tv = inplay.price_tv(eff.profile, LH, LA, h, a, m, p, rho=rho)
            assert np.allclose(se["1x2"], tv["1x2"][0], atol=1e-6)
            assert np.allclose(se["over"], tv["over"][0], atol=1e-6)


def test_calibration_preserves_prematch_expected_goals():
    eff = _toy_effects()
    c = inplay.calibrate(eff, LH, LA)
    g = np.arange(11)
    dist = inplay._remaining_dist_se(eff, c[0] * LH, c[1] * LA, 0, 0, 1, 0.0, 10)
    assert g @ dist.sum(1) == pytest.approx(LH, abs=1e-3)
    assert g @ dist.sum(0) == pytest.approx(LA, abs=1e-3)
    exact = inplay.calibrate(eff, LH, LA, exact=True)
    assert np.allclose(c, exact, atol=1e-3)


@pytest.mark.parametrize("rho", [0.0, -0.08])
def test_score_effects_monte_carlo_matches_markov_chain(rho):
    eff = _toy_effects()
    c = inplay.calibrate(eff, LH, LA)
    zs = []
    for i, (h, a, m, p) in enumerate(TV_STATES):
        mc = inplay.mc_price_se(eff, LH, LA, h, a, m, p, N_SIMS, rho=rho, seed=500 + i, scale=c)
        ex = inplay.price_se(eff, LH, LA, h, a, m, p, rho=rho, scale=c)
        zs.extend(_compare_mc(np.r_[mc["1x2"], mc["over"]], np.r_[mc["1x2_se"], mc["over_se"]],
                              np.r_[ex["1x2"], ex["over"]], N_SIMS))
    z = np.array(zs)
    assert np.abs(z).max() < 4.5 and abs(z.mean()) < 0.6 and 0.4 < (z ** 2).mean() < 1.8


def _simulate_league(mult, n_seasons=3, n_teams=20, strength_sd=0.35, seed=1):
    """Matches with unequal teams, a known time shape and known score effects."""
    rng = np.random.default_rng(seed)
    prof = _toy_profile()
    pairs, lam_h, lam_a = [], [], []
    for s in range(n_seasons):
        att, dfn = rng.normal(0, strength_sd, n_teams), rng.normal(0, strength_sd, n_teams)
        for i in range(n_teams):
            for j in range(n_teams):
                if i != j:
                    pairs.append((f"S{s}T{i}", f"S{s}T{j}"))
                    lam_h.append(np.exp(0.3 + 0.25 + att[i] - dfn[j]))
                    lam_a.append(np.exp(0.3 + att[j] - dfn[i]))
    n = len(pairs)
    lam_h, lam_a = np.array(lam_h), np.array(lam_a)
    sc = rng.choice(len(prof.probs), n, p=prof.probs)
    bounds = prof.bounds(prof.s1[sc], prof.s2[sc])
    wb, mult = prof.weights, np.asarray(mult)
    m_of = lambda d: mult[np.clip(d, -2, 2) + 2]

    def rate_fn(u, h, a, idx):
        b = (u[:, None] >= bounds[idx, 1:-1]).sum(1)
        return lam_h[idx] * wb[b] * m_of(h - a), lam_a[idx] * wb[b] * m_of(a - h)

    hg, ag, log = np.zeros(n, int), np.zeros(n, int), []
    bound = (lam_h + lam_a).max() * wb.max() * mult.max()
    inplay._thinning(np.zeros(n), bounds[:, -1], hg, ag, rate_fn, bound, rng, log)
    end1 = bounds[:, len(prof.edges1)]
    ev = pd.DataFrame([{"match_id": i, "period": 1 if u < end1[i] else 2,
                        "t": u if u < end1[i] else 45 + (u - end1[i]),
                        "event": "goal", "side": side} for i, u, side in log])
    mt = pd.DataFrame({"match_id": np.arange(n), "home": [p[0] for p in pairs],
                       "away": [p[1] for p in pairs], "end_p1": 45 + prof.s1[sc],
                       "end_p2": 90 + prof.s2[sc]})
    return mt, ev


def test_fit_recovers_score_effects_and_shows_the_biased_alternatives():
    true = np.array([1.25, 1.12, 1.0, 0.9, 0.8])
    mt, ev = _simulate_league(true)
    fit = inplay.fit_score_effects(mt, ev)
    z = (np.log(fit.multipliers) - np.log(true))[[0, 1, 3, 4]] / fit.se_log[[0, 1, 3, 4]]
    assert np.abs(z).max() < 3.5
    # time shape net of score effects is recovered too
    assert np.allclose(fit.profile.weights[[0, 1, 2, 4, 5, 6]] / fit.profile.weights[0],
                       _toy_profile().weights[[0, 1, 2, 4, 5, 6]] / _toy_profile().weights[0],
                       rtol=0.12)
    # ignoring team strength: leaders look like they score MORE (confounding)
    naive = inplay.fit_score_effects(mt, ev, method="naive")
    assert naive.multipliers[4] > true[4] * 1.15
    # conditioning on each team's goals: wildly overstated effects (invalid likelihood)
    cond = inplay.fit_score_effects(mt, ev, method="conditional")
    assert cond.multipliers[3] < true[3] * 0.6 and cond.multipliers[1] > true[1] * 1.3


def test_predictive_effects_include_learning_about_strength():
    """With strengths estimated out-of-fold, leading is also evidence of being stronger
    than estimated, so predictive multipliers for leaders sit above the tactical ones."""
    mt, ev = _simulate_league(np.array([1.25, 1.12, 1.0, 0.9, 0.8]))
    tac = inplay.fit_score_effects(mt, ev).multipliers
    pred = inplay.fit_score_effects(mt, ev, method="predictive")
    assert pred.multipliers[4] > tac[4] * 1.04 and pred.multipliers[3] > tac[3]
    assert np.all(np.isfinite(pred.se_log)) and pred.profile.total() == pytest.approx(1.0)
