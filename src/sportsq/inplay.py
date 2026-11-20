"""In-play pricing of 1X2 and over/under markets from the current match state.

Step 1: analytic baseline (see docs/inplay-plan.md)
-----------------------------------------------------
Inputs are the pre-match goal rates (lambda_h, lambda_a) from Dixon-Coles,
the current score (h, a) and the minutes played m. With r = (T - m) / T the
fraction of the match left (T = 90), the goals still to come are

    X_rem ~ Poisson(lambda_h * r),   Y_rem ~ Poisson(lambda_a * r),  independent,

and the final score is (h + X_rem, a + Y_rem). Prices are read off that
final-score grid.

How the Dixon-Coles low-score correction is handled in-play
-----------------------------------------------------------
Dixon-Coles defines a distribution for the *full-match* score:

    P(x, y) ∝ Pois(x; lambda_h) Pois(y; lambda_a) tau(x, y; lambda_h, lambda_a, rho).

To condition it on the score at minute m we need a model for *when* goals
happen. The Step 1 assumption is the one that also justifies constant rates:
given the final counts, goal times are i.i.d. uniform on [0, T]. Then each
goal has already been scored with probability s = m / T, so

    P(score at m = (h, a) | final = (x, y)) = Bin(h; x, s) Bin(a; y, s)

and Bayes' rule with Poisson thinning gives

    P(final = (x, y) | h, a at m) ∝ Pois(x - h; lambda_h r) Pois(y - a; lambda_a r) tau(x, y).

So the correction is applied to the **final** score cells (0-0, 1-0, 0-1, 1-1)
with the **full-match** rates, and the grid is renormalised. Consequences:
- at m = 0 and 0-0 the prices equal the pre-match Dixon-Coles prices exactly;
- once either final count is above 1 for sure (e.g. any score with 2+ goals
  for one side), tau = 1 everywhere it matters and the model is plain Poisson;
- at m = T the grid collapses on the current score;
- rho = 0 gives independent Poisson for the remaining goals.

Stoppage time is not modelled in Step 1: minutes are clipped to [0, T], so
"90+3" is treated as 90 (match over). Step 3 revisits this.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from .dixon_coles import grid_to_1x2

MATCH_MINUTES = 90.0
DEFAULT_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)


def remaining_fraction(minute, match_minutes=MATCH_MINUTES) -> np.ndarray:
    """Share of the match still to play, r = (T - m) / T, clipped to [0, 1]."""
    m = np.clip(np.asarray(minute, float), 0.0, match_minutes)
    return (match_minutes - m) / match_minutes


def _remaining_grid(lam_h, lam_a, hg, ag, r, rho, size) -> np.ndarray:
    """Unnormalised final-score grid: Pois(x-h; lam_h r) Pois(y-a; lam_a r) tau(x, y).

    All inputs are 1-d arrays over matches (r = remaining share of goal intensity).
    tau uses the full-match rates, as derived in the module docstring.
    """
    g = np.arange(size)
    # final x has probability Pois(x - h; lambda * r); pmf is 0 for x < h
    ph = poisson.pmf(g[None, :] - hg[:, None], (lam_h * r)[:, None])
    pa = poisson.pmf(g[None, :] - ag[:, None], (lam_a * r)[:, None])
    # mu = 0 (match over): scipy gives pmf(0, 0) = 1, but guard explicitly
    done = r <= 0
    if done.any():
        ph[done] = (g[None, :] == hg[done, None]).astype(float)
        pa[done] = (g[None, :] == ag[done, None]).astype(float)
    grid = ph[:, :, None] * pa[:, None, :]
    if rho != 0.0:
        grid[:, 0, 0] *= 1 - lam_h * lam_a * rho
        grid[:, 0, 1] *= 1 + lam_h * rho
        grid[:, 1, 0] *= 1 + lam_a * rho
        grid[:, 1, 1] *= 1 - rho
    return grid


def inplay_grid(lam_h, lam_a, home_goals=0, away_goals=0, minute=0.0, rho=0.0,
                max_remaining=10, match_minutes=MATCH_MINUTES) -> np.ndarray:
    """Final-score probabilities given the state at `minute` (constant goal rates).

    All state arguments broadcast over matches. Returns an array (n, G, G) with
    grid[k, x, y] = P(final score = x-y), where
    G = max_remaining + 1 + max current goals for either side.
    `max_remaining` truncates goals still to come per team. With 10, the mass
    left out before renormalising is < 1e-5 for a remaining rate of 2.0 and
    about 1e-3 at 3.5, which is far above any realistic in-play rate.
    """
    lam_h, lam_a, hg, ag, r = np.broadcast_arrays(
        np.atleast_1d(np.asarray(lam_h, float)), np.asarray(lam_a, float),
        np.asarray(home_goals, int), np.asarray(away_goals, int),
        remaining_fraction(minute, match_minutes))
    size = max_remaining + 1 + int(max(hg.max(), ag.max()))
    grid = _remaining_grid(lam_h, lam_a, hg, ag, r, rho, size)
    return grid / grid.sum((1, 2), keepdims=True)


def grid_to_overs(grid, lines=DEFAULT_LINES) -> np.ndarray:
    """P(total goals > line) for each line; shape (n, len(lines))."""
    g = np.arange(grid.shape[1])
    total = g[:, None] + g[None, :]
    return np.column_stack([grid[:, total > line].sum(1) for line in lines])


def price(lam_h, lam_a, home_goals=0, away_goals=0, minute=0.0, rho=0.0,
          lines=DEFAULT_LINES, max_remaining=10, match_minutes=MATCH_MINUTES) -> dict:
    """Fair in-play probabilities.

    Returns {"1x2": (n, 3) home/draw/away, "over": (n, len(lines)), "lines": lines}.
    Fair decimal odds are 1 / probability.
    """
    grid = inplay_grid(lam_h, lam_a, home_goals, away_goals, minute, rho,
                       max_remaining, match_minutes)
    return {"1x2": grid_to_1x2(grid), "over": grid_to_overs(grid, lines),
            "lines": tuple(lines)}


def price_from_model(model, home, away, home_goals=0, away_goals=0, minute=0.0,
                     lines=DEFAULT_LINES, use_rho=True) -> dict:
    """In-play prices for fixtures using a fitted `DixonColes` model's pre-match rates."""
    lam_h, lam_a = model.rates(np.atleast_1d(home), np.atleast_1d(away))
    rho = model.rho_ if use_rho else 0.0
    return price(lam_h, lam_a, home_goals, away_goals, minute, rho, lines,
                 max_remaining=model.max_goals)


def price_path(lam_h, lam_a, goals, minutes=range(0, 91, 5), rho=0.0,
               lines=DEFAULT_LINES) -> dict:
    """Prices through one match. `goals` is a list of (minute, "H" or "A") events.

    A goal at minute t counts for every evaluation minute >= t.
    """
    minutes = np.asarray(list(minutes), float)
    hg = np.array([sum(1 for t, s in goals if s == "H" and t <= m) for m in minutes])
    ag = np.array([sum(1 for t, s in goals if s == "A" and t <= m) for m in minutes])
    out = price(np.full(len(minutes), lam_h), np.full(len(minutes), lam_a),
                hg, ag, minutes, rho, lines)
    out.update({"minute": minutes, "home_goals": hg, "away_goals": ag})
    return out


# ---------------------------------------------------------------------------
# Step 2: Monte Carlo match-state simulator
# ---------------------------------------------------------------------------
"""
Goals for the rest of the match are simulated as event times of a point
process, starting from the current state (minute, score). The engine uses
thinning (Lewis & Shedler 1979): candidate events arrive at a constant bound
rate, and each is accepted as a home goal, an away goal, or rejected,
according to the current rates. Rates may depend on time and match state
through `rate_fn`, which is what Steps 3-5 plug in. With constant rates
(Step 2) the bound equals the total rate and nothing is rejected.

The Dixon-Coles correction is reproduced by self-normalised importance
weighting: each simulated match gets weight tau(final_h, final_a) with the
full-match rates, which is exactly the posterior derived for Step 1.
"""


def constant_rates(lam_h, lam_a, match_minutes=MATCH_MINUTES):
    """rate_fn for Step 2: per-minute rates that ignore time and state."""
    rh, ra = lam_h / match_minutes, lam_a / match_minutes

    def rate_fn(t, hg, ag, idx=None):
        return np.full(t.shape, rh), np.full(t.shape, ra)

    return rate_fn, rh + ra


def dc_tau(x, y, lam_h, lam_a, rho) -> np.ndarray:
    """Dixon-Coles low-score factor for final scores (x, y)."""
    x, y = np.asarray(x), np.asarray(y)
    t = np.ones(np.broadcast(x, y).shape)
    t = np.where((x == 0) & (y == 0), 1 - lam_h * lam_a * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam_h * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + lam_a * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


def _thinning(start, end, hg, ag, rate_fn, rate_bound, rng):
    """Simulate goals on [start, end) per path by thinning; updates hg, ag in place.

    `rate_fn(t, hg, ag, idx) -> (rate_home, rate_away)` per minute, evaluated for
    the live paths `idx`; `rate_bound` must be >= their sum everywhere.
    """
    t = np.asarray(start, float).copy()
    end = np.broadcast_to(np.asarray(end, float), t.shape)
    live = np.flatnonzero(t < end) if rate_bound > 0 else np.array([], int)
    while live.size:
        t[live] += rng.exponential(1.0 / rate_bound, live.size)
        live = live[t[live] < end[live]]
        if not live.size:
            break
        rh, ra = rate_fn(t[live], hg[live], ag[live], live)
        if np.any(rh + ra > rate_bound * (1 + 1e-12)):
            raise ValueError("rate_fn exceeded rate_bound; thinning would be biased")
        u = rng.random(live.size) * rate_bound
        hg[live[u < rh]] += 1
        ag[live[(u >= rh) & (u < rh + ra)]] += 1
    return hg, ag


def simulate_remaining(lam_h, lam_a, home_goals=0, away_goals=0, minute=0.0,
                       n_sims=100_000, rho=0.0, seed=None, rate_fn=None,
                       rate_bound=None, match_minutes=MATCH_MINUTES) -> dict:
    """Simulate the rest of one match `n_sims` times from the given state.

    Returns {"home": final home goals, "away": final away goals,
             "weights": importance weights (all 1 when rho = 0)}.
    `rate_fn(t, hg, ag, idx) -> (rate_home, rate_away)` gives per-minute rates for
    arrays of current times, scores and path indices; `rate_bound` must be >= their sum.
    """
    rng = np.random.default_rng(seed)
    if rate_fn is None:
        rate_fn, rate_bound = constant_rates(lam_h, lam_a, match_minutes)
    elif rate_bound is None:
        raise ValueError("rate_bound is required with a custom rate_fn")
    start = np.full(n_sims, float(np.clip(minute, 0, match_minutes)))
    hg = np.full(n_sims, int(home_goals))
    ag = np.full(n_sims, int(away_goals))
    _thinning(start, match_minutes, hg, ag, rate_fn, rate_bound, rng)
    w = dc_tau(hg, ag, lam_h, lam_a, rho) if rho != 0.0 else np.ones(n_sims)
    return {"home": hg, "away": ag, "weights": w}


def _weighted_prob(indicator, w):
    """Self-normalised estimate of P(event) and its delta-method standard error."""
    p = (w * indicator).sum() / w.sum()
    se = np.sqrt((w ** 2 * (indicator - p) ** 2).sum()) / w.sum()
    return p, se


def mc_price(lam_h, lam_a, home_goals=0, away_goals=0, minute=0.0, n_sims=100_000,
             rho=0.0, lines=DEFAULT_LINES, seed=None, **sim_kwargs) -> dict:
    """Monte Carlo prices for one match state, with standard errors.

    Returns {"1x2", "1x2_se", "over", "over_se", "lines", "ess"} where ess is
    the effective sample size (n_sims when unweighted).
    """
    sim = simulate_remaining(lam_h, lam_a, home_goals, away_goals, minute, n_sims,
                             rho, seed, **sim_kwargs)
    return _mc_estimates(sim["home"], sim["away"], sim["weights"], lines)


def _mc_estimates(h, a, w, lines) -> dict:
    events = [h > a, h == a, h < a] + [(h + a) > line for line in lines]
    est = np.array([_weighted_prob(e.astype(float), w) for e in events])
    k = 3
    return {"1x2": est[:k, 0], "1x2_se": est[:k, 1],
            "over": est[k:, 0], "over_se": est[k:, 1], "lines": tuple(lines),
            "ess": w.sum() ** 2 / (w ** 2).sum()}


# ---------------------------------------------------------------------------
# Step 3: time-varying goal intensity and stoppage time
# ---------------------------------------------------------------------------
"""
Goal rates rise through a match, and each half runs into stoppage time of
random length. The model:

- **Clock.** A state is (period, t) on the StatsBomb clock: period 1 runs
  0-45 then first-half stoppage (t > 45); period 2 restarts at 45, runs to 90,
  then second-half stoppage. Internally both halves are laid end to end on a
  "played time" axis u in [0, 90 + s1 + s2].
- **Shape.** A piecewise-constant intensity with weights w_b per bin (by
  default 15-minute bins plus one stoppage bin per half). Each team's
  per-minute goal rate in bin b is lambda_team * w_b, where lambda_team is the
  pre-match expected goals for the full match (Dixon-Coles is fitted on
  full-time scores, so stoppage goals are included).
- **Normalisation.** The weights are scaled so that the expected integrated
  intensity over a whole match, averaged over stoppage lengths, is 1. So the
  expected full-match goals are still lambda_team.
- **Stoppage.** Stoppage lengths (s1, s2) are drawn from their empirical joint
  distribution, discretised into scenarios. During stoppage the scenarios are
  conditioned on the stoppage already played (s >= time elapsed past 45/90).
- **Pricing.** Within a scenario the remaining goals are Poisson with mean
  lambda * (remaining integrated intensity). Prices are the scenario mixture.
- **Dixon-Coles correction.** With an inhomogeneous intensity that has the
  same shape for both teams, goal times given the final counts are i.i.d. with
  density proportional to the intensity. So the Step 1 derivation holds with
  "share of the match played" replaced by "share of the integrated intensity
  used", and tau is applied to final scores with the full-match rates. Across
  stoppage scenarios the grid is mixed *before* normalising, so the scenario
  weights are updated by the tau likelihood exactly as in the Monte Carlo
  estimator.
- **What stays out:** score effects (Step 4) and red cards (Step 5). Part of
  the late rise in goals is caused by score effects (trailing teams attack),
  so the Step 3 shape will partly absorb them until Step 4 separates them.
"""

from dataclasses import dataclass, field  # noqa: E402

import pandas as pd  # noqa: E402
from scipy.stats import chi2  # noqa: E402


@dataclass
class GoalIntensity:
    """Piecewise-constant goal intensity shape plus a stoppage-time distribution.

    edges1 / edges2: regulation bin edges for each half on the StatsBomb clock,
        e.g. [0, 15, 30, 45] and [45, 60, 75, 90].
    weights: one weight per bin in play order: period-1 regulation bins,
        period-1 stoppage, period-2 regulation bins, period-2 stoppage.
    s1, s2, probs: stoppage scenarios (minutes added in each half) and their
        probabilities.
    """
    edges1: np.ndarray
    edges2: np.ndarray
    weights: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    probs: np.ndarray
    table: pd.DataFrame | None = field(default=None, repr=False)

    def __post_init__(self):
        self.edges1 = np.asarray(self.edges1, float)
        self.edges2 = np.asarray(self.edges2, float)
        self.weights = np.asarray(self.weights, float)
        self.s1 = np.asarray(self.s1, float)
        self.s2 = np.asarray(self.s2, float)
        self.probs = np.asarray(self.probs, float) / np.sum(self.probs)
        n_bins = (len(self.edges1) - 1) + 1 + (len(self.edges2) - 1) + 1
        if len(self.weights) != n_bins:
            raise ValueError(f"expected {n_bins} weights, got {len(self.weights)}")

    # -- constructors ------------------------------------------------------
    @classmethod
    def constant(cls, match_minutes=MATCH_MINUTES):
        """Uniform rate over 90 minutes and no stoppage: reproduces Step 1 exactly."""
        half = match_minutes / 2
        w = np.array([1.0, 0.0, 1.0, 0.0]) / match_minutes
        return cls([0, half], [half, match_minutes], w, [0.0], [0.0], [1.0])

    def with_mean_stoppage(self):
        """Same shape with stoppage fixed at its expected length (one scenario)."""
        return GoalIntensity(self.edges1, self.edges2, self.weights,
                             [self.probs @ self.s1], [self.probs @ self.s2], [1.0],
                             self.table)

    # -- geometry ------------------------------------------------------------
    def bounds(self, s1, s2) -> np.ndarray:
        """Bin boundaries on the played-time axis u, shape (K, n_bins + 1)."""
        s1, s2 = np.atleast_1d(s1), np.atleast_1d(s2)
        h1, h2 = self.edges1[-1], self.edges2[0]
        p1 = np.broadcast_to(self.edges1, (len(s1), len(self.edges1)))
        end1 = (h1 + s1)[:, None]
        p2 = end1 + (self.edges2[1:] - h2)[None, :]
        end2 = p2[:, -1:] + s2[:, None]
        return np.hstack([p1, end1, p2, end2])

    def played_time(self, period, t, s1) -> np.ndarray:
        """Map (period, t) on the StatsBomb clock to played time u for each scenario s1."""
        h1, h2 = self.edges1[-1], self.edges2[0]
        s1 = np.atleast_1d(s1)
        if period == 1:
            return np.minimum(np.full(s1.shape, float(t)), h1 + s1)
        return h1 + s1 + max(float(t) - h2, 0.0)

    def remaining(self, u, bounds) -> np.ndarray:
        """Integrated intensity still to come after played time u (per scenario)."""
        lo = np.maximum(bounds[:, :-1], np.asarray(u)[:, None])
        return (self.weights * np.clip(bounds[:, 1:] - lo, 0, None)).sum(1)

    def total(self) -> float:
        """Expected integrated intensity over a whole match (1 after fitting)."""
        b = self.bounds(self.s1, self.s2)
        return float(self.probs @ self.remaining(np.zeros(len(self.s1)), b))

    def scenario_weights(self, period, t) -> np.ndarray:
        """Stoppage scenarios conditioned on the stoppage already played."""
        p = self.probs.copy()
        h1, end2 = self.edges1[-1], self.edges2[-1]
        if period == 1 and t > h1:
            p = p * (self.s1 >= t - h1)
        if period == 2 and t > end2:
            p = p * (self.s2 >= t - end2)
        tot = p.sum()
        return p / tot if tot > 0 else p  # all zero: beyond every scenario, match over

    def remaining_share(self, period, t):
        """(remaining integrated intensity per scenario, scenario weights)."""
        w = self.scenario_weights(period, t)
        b = self.bounds(self.s1, self.s2)
        r = self.remaining(self.played_time(period, t, self.s1), b)
        if w.sum() == 0:
            return np.zeros(1), np.ones(1)
        keep = w > 0
        return r[keep], w[keep]

    # -- persistence ------------------------------------------------------------
    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in
                ("edges1", "edges2", "weights", "s1", "s2", "probs")}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in ("edges1", "edges2", "weights", "s1", "s2", "probs")})


def default_period(minute) -> int:
    """Period implied by a minute when not given: < 45 is the first half."""
    return 1 if minute < 45 else 2


def fit_goal_intensity(matches: pd.DataFrame, events: pd.DataFrame, bin_minutes=15,
                       stoppage_step=0.5) -> GoalIntensity:
    """Fit the intensity shape from goal times (StatsBomb files in data/processed).

    Per bin, the maximum-likelihood rate of a piecewise-constant Poisson process
    is goals / exposure minutes. Regulation exposure is (matches x bin width);
    stoppage exposure is the total stoppage time played. 95% intervals are exact
    Poisson (Garwood) intervals on the counts.
    """
    m = matches.copy()
    s1 = (m.end_p1 - 45).clip(lower=0).to_numpy()
    s2 = (m.end_p2 - 90).clip(lower=0).to_numpy()
    n = len(m)
    edges1 = np.arange(0, 45 + 1e-9, bin_minutes)
    edges2 = np.arange(45, 90 + 1e-9, bin_minutes)
    if edges1[-1] != 45 or edges2[-1] != 90:
        raise ValueError("bin_minutes must divide 45")
    g = events[(events.event == "goal") & events.match_id.isin(m.match_id)]
    p1, p2 = g[g.period == 1].t.to_numpy(), g[g.period == 2].t.to_numpy()
    counts = np.concatenate([
        np.histogram(p1[p1 < 45], edges1)[0], [(p1 >= 45).sum()],
        np.histogram(p2[p2 < 90], edges2)[0], [(p2 >= 90).sum()]])
    exposure = np.concatenate([
        np.full(len(edges1) - 1, n * bin_minutes), [s1.sum()],
        np.full(len(edges2) - 1, n * bin_minutes), [s2.sum()]])
    rate = counts / exposure  # goals per minute, both teams

    # discretised joint stoppage distribution
    d1 = np.round(s1 / stoppage_step) * stoppage_step
    d2 = np.round(s2 / stoppage_step) * stoppage_step
    sc = pd.Series(1, index=pd.MultiIndex.from_arrays([d1, d2])).groupby(level=[0, 1]).sum()
    S1 = sc.index.get_level_values(0).to_numpy(float)
    S2 = sc.index.get_level_values(1).to_numpy(float)
    P = sc.to_numpy(float) / sc.sum()

    prof = GoalIntensity(edges1, edges2, rate, S1, S2, P)
    total = prof.total()  # expected goals per match (both teams) under this rate
    prof.weights = rate / total
    labels = ([f"{a:.0f}-{b:.0f}" for a, b in zip(edges1[:-1], edges1[1:])] + ["45+"] +
              [f"{a:.0f}-{b:.0f}" for a, b in zip(edges2[:-1], edges2[1:])] + ["90+"])
    lo = chi2.ppf(0.025, 2 * counts) / 2
    hi = chi2.ppf(0.975, 2 * (counts + 1)) / 2
    overall = counts.sum() / exposure.sum()
    prof.table = pd.DataFrame({
        "bin": labels, "goals": counts, "exposure_min": exposure.round(1),
        "goals_per_90": (rate * 90).round(3),
        "relative_rate": (rate / overall).round(3),
        "rel_ci_low": (lo / exposure / overall).round(3),
        "rel_ci_high": (hi / exposure / overall).round(3),
    })
    prof.table.attrs.update(goals_per_match=total, n_matches=n,
                            mean_s1=float(s1.mean()), mean_s2=float(s2.mean()))
    return prof


def price_tv(profile: GoalIntensity, lam_h, lam_a, home_goals=0, away_goals=0,
             minute=0.0, period=None, rho=0.0, lines=DEFAULT_LINES,
             max_remaining=10) -> dict:
    """Analytic in-play prices with a time-varying intensity and stoppage time.

    (minute, period) is on the StatsBomb clock; e.g. first-half stoppage 45+2 is
    minute=47, period=1, and second-half 90+3 is minute=93, period=2. Arguments
    broadcast over matches; `minute` and `period` may be arrays.
    """
    lam_h, lam_a, hg, ag, mins = np.broadcast_arrays(
        np.atleast_1d(np.asarray(lam_h, float)), np.asarray(lam_a, float),
        np.asarray(home_goals, int), np.asarray(away_goals, int),
        np.asarray(minute, float))
    pers = (np.array([default_period(x) for x in mins]) if period is None
            else np.broadcast_to(np.asarray(period, int), mins.shape))
    size = max_remaining + 1 + int(max(hg.max(), ag.max()))
    grid = np.zeros((len(mins), size, size))
    cache = {}
    for i, (p, t) in enumerate(zip(pers, mins)):
        key = (int(p), float(t))
        if key not in cache:
            cache[key] = profile.remaining_share(*key)
        r, w = cache[key]
        k = len(r)
        sub = _remaining_grid(np.repeat(lam_h[i], k), np.repeat(lam_a[i], k),
                              np.repeat(hg[i], k), np.repeat(ag[i], k), r, rho, size)
        grid[i] = np.tensordot(w, sub, axes=1)
    grid /= grid.sum((1, 2), keepdims=True)
    return {"1x2": grid_to_1x2(grid), "over": grid_to_overs(grid, lines),
            "lines": tuple(lines)}


def mc_price_tv(profile: GoalIntensity, lam_h, lam_a, home_goals=0, away_goals=0,
                minute=0.0, period=None, n_sims=100_000, rho=0.0,
                lines=DEFAULT_LINES, seed=None) -> dict:
    """Monte Carlo prices for one state under `profile`, by thinning on played time."""
    rng = np.random.default_rng(seed)
    period = default_period(minute) if period is None else period
    probs = profile.scenario_weights(period, minute)
    hg = np.full(n_sims, int(home_goals))
    ag = np.full(n_sims, int(away_goals))
    if probs.sum() > 0:
        sc = rng.choice(len(probs), n_sims, p=probs)
        bounds = profile.bounds(profile.s1[sc], profile.s2[sc])
        start = np.array([profile.played_time(period, minute, s)[0]
                          for s in profile.s1[sc]]) if period == 1 else \
            profile.played_time(period, minute, profile.s1[sc])
        wb = profile.weights

        def rate_fn(u, h, a, idx):
            b = (u[:, None] >= bounds[idx, 1:-1]).sum(1)
            return lam_h * wb[b], lam_a * wb[b]

        _thinning(start, bounds[:, -1], hg, ag, rate_fn, (lam_h + lam_a) * wb.max(), rng)
    w = dc_tau(hg, ag, lam_h, lam_a, rho) if rho != 0.0 else np.ones(n_sims)
    return _mc_estimates(hg, ag, w, lines)


def load_statsbomb(processed_dir="data/processed"):
    """(matches, events) frames written by scripts/download_statsbomb.py."""
    from pathlib import Path
    d = Path(processed_dir)
    return (pd.read_csv(d / "statsbomb_matches.csv"),
            pd.read_csv(d / "statsbomb_events.csv"))
