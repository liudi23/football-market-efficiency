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


def _thinning(start, end, hg, ag, rate_fn, rate_bound, rng, log=None):
    """Simulate goals on [start, end) per path by thinning; updates hg, ag in place.

    `rate_fn(t, hg, ag, idx) -> (rate_home, rate_away)` per minute, evaluated for
    the live paths `idx`; `rate_bound` must be >= their sum everywhere. If `log`
    is a list, (path index, time, "H"/"A") is appended for every goal.
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
        home, away = live[u < rh], live[(u >= rh) & (u < rh + ra)]
        hg[home] += 1
        ag[away] += 1
        if log is not None:
            log.extend((int(i), float(t[i]), "H") for i in home)
            log.extend((int(i), float(t[i]), "A") for i in away)
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


# ---------------------------------------------------------------------------
# Step 4: score effects
# ---------------------------------------------------------------------------
"""
A team's scoring rate depends on the current goal difference from its own
point of view (Dixon & Robinson 1998). Per-minute rate for the home team:

    c_h * lambda_h * w_b(u) * m(hg - ag),     and for away: c_a * lambda_a * w_b(u) * m(ag - hg)

- m(d) are multipliers for goal-difference states d in {<=-2, -1, 0, +1, >=+2},
  with m(0) = 1 (level score is the reference).
- w_b is the time shape *net of* score effects, estimated jointly with m.
- c_h, c_a are per-match calibration constants chosen so that the expected
  full-match goals still equal the pre-match lambda_h, lambda_a.

Estimation: full Poisson-process likelihood on segments of constant
(time bin, score), with a log-exposure offset and log-rate

    mu + home + attack[team] - defence[opponent] + gamma[bin] + beta[state].

Team attack/defence (per team, i.e. per league-season here) control for the
fact that stronger teams both lead more often and score more. The full
likelihood is valid even though the state depends on earlier goals (counting
process with predictable intensity). Do NOT condition on each team's total
goals in the match: the states are created by the team's own goals, so the
segment exposures depend on the outcome. For example, a team that scored
exactly once has its goal in the pre-goal state and, by construction, no
goals after it while leading. That conditional likelihood is invalid and
grossly overstates score effects (kept as method="conditional" only to
demonstrate the bias; see reports/inplay_score_effects.md).

Tactical vs predictive effects: with team strengths estimated in-sample (from
the same matches), the multipliers measure the *tactical* effect of the score
(leaders sit back, trailers push). A live pricer only has a pre-match
*estimate* of strength, and the score is also evidence about that estimate's
error (a team that leads is probably stronger than estimated). For pricing,
the multipliers must be measured relative to the strength estimate actually
used: method="predictive" cross-fits team strengths on other matches
(out-of-fold) and fits time and score effects on top of those offsets. In the
2015/16 data the two differ for leading teams (tactical 0.93-0.99,
predictive ~1.03); trailing teams score more under both.

Pricing: rates now depend on the score, so remaining goals are no longer
Poisson. The exact distribution comes from the continuous-time Markov chain
over scores (hg, ag): within each time bin the generator is constant, so the
forward equation is solved with a matrix exponential per bin, per stoppage
scenario. Goals per team are truncated at current + max_remaining (the last
level is absorbing; the mass that reaches it is negligible).

Dixon-Coles correction: with score effects goal times are no longer i.i.d.
given the counts, so the Step 1 derivation does not apply. tau is kept as an
optional reweighting of final scores (same in the analytic and Monte Carlo
pricers), but because score effects already change the low-score structure,
rho = 0 is the recommended setting here; Step 6 decides empirically.
"""

from scipy.optimize import minimize as _minimize  # noqa: E402
from scipy.sparse import csr_matrix as _csr  # noqa: E402
from scipy.sparse.linalg import expm_multiply as _expm_multiply  # noqa: E402

DIFF_STATES = (-2, -1, 0, 1, 2)  # goal difference from the scoring team's view, clipped


@dataclass
class ScoreEffects:
    """Score-state multipliers and the time shape estimated jointly with them."""
    multipliers: np.ndarray            # m(d) for d in DIFF_STATES, m(0) = 1
    profile: GoalIntensity             # time shape net of score effects
    se_log: np.ndarray | None = None   # standard errors of log m (same order)
    table: pd.DataFrame | None = field(default=None, repr=False)

    @classmethod
    def none(cls, profile: GoalIntensity):
        return cls(np.ones(len(DIFF_STATES)), profile)

    def m(self, d):
        return self.multipliers[np.clip(np.asarray(d), -2, 2) + 2]

    def to_dict(self):
        return {"multipliers": self.multipliers.tolist(), "profile": self.profile.to_dict()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["multipliers"]), GoalIntensity.from_dict(d["profile"]))


def build_segments(matches: pd.DataFrame, events: pd.DataFrame,
                   edges1=(0, 15, 30, 45), edges2=(45, 60, 75, 90)) -> pd.DataFrame:
    """Split each match into intervals of constant (time bin, score).

    One row per (match, interval, team) with: `match_id`, `team`, `opponent`,
    `is_home`, `bin`, `diff` (goal difference from that team's view, clipped to
    [-2, 2]), `length` in minutes, and `goals` scored by that team at the end of
    the interval. Goals use the score *before* the goal. Red cards are ignored
    here (Step 5).
    """
    geo = GoalIntensity(edges1, edges2, np.ones(len(edges1) + len(edges2)), [0], [0], [1])
    gl = events[events.event == "goal"]
    by_match = {k: v for k, v in gl.groupby("match_id")}
    rows = []
    for mt in matches.itertuples(index=False):
        s1, s2 = max(mt.end_p1 - 45, 0.0), max(mt.end_p2 - 90, 0.0)
        b = geo.bounds([s1], [s2])[0]
        end1 = b[len(geo.edges1)]
        g = by_match.get(mt.match_id)
        times, sides = [], []
        if g is not None:
            for r in g.itertuples(index=False):
                u = r.t if r.period == 1 else end1 + (r.t - 45)
                times.append(min(max(u, 0.0), b[-1] - 1e-9))
                sides.append(r.side)
        order = np.argsort(times, kind="stable")
        times, sides = np.asarray(times, float)[order], np.asarray(sides)[order]
        cuts = np.unique(np.concatenate([b, times]))
        hg = ag = 0
        gi = 0
        for a, z in zip(cuts[:-1], cuts[1:]):
            k = min(int((a >= b[1:-1]).sum()), len(b) - 2)
            gh = ga = 0
            while gi < len(times) and abs(times[gi] - z) < 1e-12:
                gh += sides[gi] == "H"
                ga += sides[gi] == "A"
                gi += 1
            rows.append((mt.match_id, mt.home, mt.away, 1, k, int(np.clip(hg - ag, -2, 2)), z - a, gh))
            rows.append((mt.match_id, mt.away, mt.home, 0, k, int(np.clip(ag - hg, -2, 2)), z - a, ga))
            hg, ag = hg + gh, ag + ga
    seg = pd.DataFrame(rows, columns=["match_id", "team", "opponent", "is_home", "bin",
                                      "diff", "length", "goals"])
    return seg[seg.length > 0].reset_index(drop=True)


def _poisson_newton(X, offset, y, ridge, max_iter=100, tol=1e-10):
    """Penalised Poisson regression by Newton's method (X sparse or dense)."""
    from scipy.sparse import issparse
    th = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = offset + X @ th
        mu = np.exp(eta)
        grad = X.T @ (y - mu) - ridge * th
        W = X.multiply(mu[:, None]) if issparse(X) else X * mu[:, None]
        H = np.asarray((X.T @ W).todense() if issparse(W) else X.T @ W) + np.diag(ridge)
        step = np.linalg.solve(H, grad)
        th += step
        if np.max(np.abs(step)) < tol:
            break
    eta = offset + X @ th
    ll = float(y @ eta - np.exp(eta).sum())
    return th, np.linalg.inv(H), ll


def _cond_loglik(theta, X, length, goals, tm_codes, n_tm):
    eta = X @ theta
    mu = length * np.exp(eta)
    denom = np.bincount(tm_codes, mu)
    keep = n_tm > 0
    ll = goals @ eta - (n_tm[keep] * np.log(denom[keep])).sum()
    share = mu / denom[tm_codes]
    grad = X.T @ goals - X.T @ (share * n_tm[tm_codes])
    return ll, grad


def score_design(seg, n_bins, states, teams=None, ridge=1e-4):
    """Sparse design for the segment Poisson model.

    Columns: intercept, home, time bins 1..n_bins-1, score states (level is the
    reference), then attack and defence per team if `teams` is given. Returns
    (X, ridge penalties, names of the shape columns, number of fixed columns).
    """
    from scipy.sparse import csr_matrix, hstack
    n = len(seg)
    dense = [np.ones(n), seg.is_home.to_numpy(float)]
    dense += [(seg.bin.to_numpy() == k).astype(float) for k in range(1, n_bins)]
    dense += [(seg["diff"].to_numpy() == d).astype(float) for d in states]
    names = [f"bin {k}" for k in range(1, n_bins)] + [f"diff {d:+d}" for d in states]
    n_fixed = len(dense)
    blocks = [csr_matrix(np.column_stack(dense))]
    pen = [0.0] * n_fixed
    if teams is not None:
        ti, oi = teams.get_indexer(seg.team), teams.get_indexer(seg.opponent)
        if (ti < 0).any() or (oi < 0).any():
            raise ValueError("segments contain teams not in `teams`")
        blocks += [csr_matrix((np.ones(n), (np.arange(n), ti)), shape=(n, len(teams))),
                   csr_matrix((-np.ones(n), (np.arange(n), oi)), shape=(n, len(teams)))]
        pen += [ridge] * (2 * len(teams))  # pins the attack/defence level per league
    return hstack(blocks).tocsr(), np.array(pen), names, n_fixed


def fit_score_effects(matches: pd.DataFrame, events: pd.DataFrame, bin_minutes=15,
                      stoppage_step=0.5, score_effects=True, method="poisson",
                      ridge=1e-4, folds=5, seed=0) -> ScoreEffects:
    """Fit time-bin and score-state effects jointly.

    method:
      "poisson"     full likelihood with in-sample team attack/defence and home
                    advantage: the *tactical* effect of the score
      "predictive"  team strengths cross-fitted on other matches (out-of-fold
                    offsets); effects relative to an estimated strength. Use
                    this for pricing.
      "naive"       full likelihood without team strengths (confounded by strength)
      "conditional" conditions on each team's goals per match (invalid; see module notes)

    Returns multipliers m(d) with standard errors, and a GoalIntensity whose
    weights are the time effects net of score effects (normalised to expected
    integrated intensity 1 *before* score effects; the pricer recalibrates per
    match so expected goals equal the pre-match lambdas).
    """
    base = fit_goal_intensity(matches, events, bin_minutes, stoppage_step)
    seg = build_segments(matches, events, tuple(base.edges1), tuple(base.edges2))
    n_bins = len(base.weights)
    states = [d for d in DIFF_STATES if d != 0] if score_effects else []
    L, Y = seg.length.to_numpy(float), seg.goals.to_numpy(float)

    if method == "conditional":
        X = np.column_stack([(seg.bin.to_numpy() == k).astype(float) for k in range(1, n_bins)] +
                            [(seg["diff"].to_numpy() == d).astype(float) for d in states])
        names = [f"bin {k}" for k in range(1, n_bins)] + [f"diff {d:+d}" for d in states]
        tm_codes, _ = pd.factorize(seg.match_id.astype(str) + "|" + seg.team)
        n_tm = np.bincount(tm_codes, Y)

        def f(th):
            ll, g = _cond_loglik(th, X, L, Y, tm_codes, n_tm)
            return -ll, -g

        res = _minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                        options={"maxiter": 5000, "gtol": 1e-9})
        th, ll = res.x, -res.fun
        eps = 1e-5
        H = np.empty((len(th), len(th)))
        for j in range(len(th)):
            e = np.zeros(len(th)); e[j] = eps
            H[:, j] = (f(th + e)[1] - f(th - e)[1]) / (2 * eps)
        cov = np.linalg.inv((H + H.T) / 2)
    elif method == "predictive":
        teams = pd.Index(sorted(set(seg.team) | set(seg.opponent)))
        X0, pen0, _, nf0 = score_design(seg, n_bins, [], teams, ridge)
        rng = np.random.default_rng(seed)
        mids = seg.match_id.unique()
        fold = seg.match_id.map(dict(zip(mids, rng.integers(0, folds, len(mids))))).to_numpy()
        strength_cols = np.r_[[0, 1], np.arange(nf0, X0.shape[1])]  # intercept, home, att, def
        offset = np.zeros(len(seg))
        for k in range(folds):
            tr, te = fold != k, fold == k
            th0, _, _ = _poisson_newton(X0[tr], np.log(L[tr]), Y[tr], pen0)
            offset[te] = X0[te][:, strength_cols] @ th0[strength_cols]
        X, _, names, n_fixed = score_design(seg, n_bins, states, None, ridge)
        keep = np.r_[0, np.arange(2, X.shape[1])]  # home is inside the offset
        X = X[:, keep]
        th_all, cov_all, ll = _poisson_newton(X, np.log(L) + offset, Y, np.zeros(X.shape[1]))
        th = th_all[1:]
        cov = cov_all[1:, 1:]
    elif method in ("poisson", "naive"):
        teams = (pd.Index(sorted(set(seg.team) | set(seg.opponent)))
                 if method == "poisson" else None)
        X, pen, names, n_fixed = score_design(seg, n_bins, states, teams, ridge)
        th_all, cov_all, ll = _poisson_newton(X, np.log(L), Y, pen)
        th = th_all[2:n_fixed]
        cov = cov_all[2:n_fixed, 2:n_fixed]
    else:
        raise ValueError(method)
    se = np.sqrt(np.diag(cov))

    gamma = np.r_[0.0, th[:n_bins - 1]]
    prof = GoalIntensity(base.edges1, base.edges2, np.exp(gamma), base.s1, base.s2, base.probs)
    prof.weights = prof.weights / prof.total()
    beta = np.zeros(len(DIFF_STATES))
    beta_se = np.zeros(len(DIFF_STATES))
    for j, d in enumerate(states):
        beta[DIFF_STATES.index(d)] = th[n_bins - 1 + j]
        beta_se[DIFF_STATES.index(d)] = se[n_bins - 1 + j]
    table = pd.DataFrame({
        "goal difference (scoring team's view)": ["≤ −2", "−1", "0 (level)", "+1", "≥ +2"],
        "exposure_min": [round(seg.length[seg["diff"] == d].sum()) for d in DIFF_STATES],
        "goals": [int(seg.goals[seg["diff"] == d].sum()) for d in DIFF_STATES],
        "multiplier": np.exp(beta).round(3),
        "ci_low": np.exp(beta - 1.96 * beta_se).round(3),
        "ci_high": np.exp(beta + 1.96 * beta_se).round(3),
    })
    table.attrs.update(loglik=ll, method=method, param_names=names,
                       gamma_se=np.r_[0.0, se[:n_bins - 1]])
    return ScoreEffects(np.exp(beta), prof, beta_se, table)


# -- pricing with score effects ------------------------------------------------
def _score_index(H, A):
    return lambda x, y: x * (A + 1) + y


def _generators(eff: ScoreEffects, rh, ra, hg0, ag0, max_remaining):
    """Per-bin sparse generators (transposed, for column-vector propagation) on
    the truncated score grid hg0..hg0+R, ag0..ag0+R."""
    R = max_remaining
    xs, ys = np.meshgrid(np.arange(R + 1), np.arange(R + 1), indexing="ij")
    x, y = xs.ravel(), ys.ravel()
    d = (hg0 + x) - (ag0 + y)
    mh, ma = eff.m(d), eff.m(-d)
    n = len(x)
    idx = x * (R + 1) + y
    out = []
    for w in eff.profile.weights:
        qh = np.where(x < R, rh * w * mh, 0.0)
        qa = np.where(y < R, ra * w * ma, 0.0)
        rows = np.r_[idx, idx[x < R] + (R + 1), idx[y < R] + 1]
        cols = np.r_[idx, idx[x < R], idx[y < R]]
        vals = np.r_[-(qh + qa), qh[x < R], qa[y < R]]
        out.append(_csr((vals, (rows, cols)), shape=(n, n)))  # dp/dt = Q^T p
    return out


def _propagate(eff, gens, u0, scen_bounds, R):
    """Distribution over remaining goals (R+1, R+1) after play from u0 to the end."""
    p = np.zeros((R + 1) ** 2)
    p[0] = 1.0
    for k in range(len(eff.profile.weights)):
        dt = scen_bounds[k + 1] - max(scen_bounds[k], u0)
        if dt > 0:
            p = _expm_multiply(gens[k] * dt, p)
    return np.clip(p, 0, None).reshape(R + 1, R + 1)


def _remaining_dist_se(eff, rh, ra, hg, ag, period, minute, R):
    """Scenario-mixed distribution of remaining goals under score effects."""
    prof = eff.profile
    w = prof.scenario_weights(period, minute)
    if w.sum() == 0:
        out = np.zeros((R + 1, R + 1)); out[0, 0] = 1.0
        return out
    gens = _generators(eff, rh, ra, hg, ag, R)
    b = prof.bounds(prof.s1, prof.s2)
    u0 = prof.played_time(period, minute, prof.s1)
    dist = np.zeros((R + 1, R + 1))
    for k in np.flatnonzero(w > 0):
        dist += w[k] * _propagate(eff, gens, u0[k], b[k], R)
    return dist


def calibrate(eff: ScoreEffects, lam_h, lam_a, max_remaining=10, tol=1e-7, max_iter=50,
              exact=False):
    """Scale factors (c_h, c_a) so pre-match expected goals equal (lam_h, lam_a).

    By default the fixed-point iteration runs on the profile with stoppage fixed
    at its mean, which is ~100x faster and matches the exact (scenario-mixture)
    answer to about 1e-5 in expected goals; `exact=True` uses every scenario.
    """
    use = eff if exact else ScoreEffects(eff.multipliers, eff.profile.with_mean_stoppage())
    c = np.array([1.0, 1.0])
    g = np.arange(max_remaining + 1)
    for _ in range(max_iter):
        dist = _remaining_dist_se(use, c[0] * lam_h, c[1] * lam_a, 0, 0, 1, 0.0, max_remaining)
        e = np.array([g @ dist.sum(1), g @ dist.sum(0)])
        new = c * np.array([lam_h, lam_a]) / e
        if np.max(np.abs(new - c)) < tol:
            return new
        c = new
    return c


def price_se(eff: ScoreEffects, lam_h, lam_a, home_goals=0, away_goals=0, minute=0.0,
             period=None, rho=0.0, lines=DEFAULT_LINES, max_remaining=10,
             scale=None) -> dict:
    """Exact in-play prices for one match state with score effects (Markov chain).

    Returns 1-d arrays (one state), like the Monte Carlo pricers.

    `scale` = (c_h, c_a) from `calibrate`; computed if not given (cache it when
    pricing many states of the same match).
    """
    period = default_period(minute) if period is None else period
    R = max_remaining
    c = calibrate(eff, lam_h, lam_a, R) if scale is None else np.asarray(scale)
    dist = _remaining_dist_se(eff, c[0] * lam_h, c[1] * lam_a, int(home_goals),
                              int(away_goals), period, minute, R)
    size = R + 1 + max(int(home_goals), int(away_goals))
    grid = np.zeros((1, size, size))
    grid[0, home_goals:home_goals + R + 1, away_goals:away_goals + R + 1] = dist
    if rho != 0.0:
        grid[0] *= dc_tau(*np.meshgrid(np.arange(size), np.arange(size), indexing="ij"),
                          lam_h, lam_a, rho)
    grid /= grid.sum()
    return {"1x2": grid_to_1x2(grid)[0], "over": grid_to_overs(grid, lines)[0],
            "lines": tuple(lines), "scale": c}


def mc_price_se(eff: ScoreEffects, lam_h, lam_a, home_goals=0, away_goals=0, minute=0.0,
                period=None, n_sims=100_000, rho=0.0, lines=DEFAULT_LINES, seed=None,
                scale=None, max_remaining=10, log=None) -> dict:
    """Monte Carlo prices with score effects (thinning with state-dependent rates)."""
    rng = np.random.default_rng(seed)
    period = default_period(minute) if period is None else period
    c = calibrate(eff, lam_h, lam_a, max_remaining) if scale is None else np.asarray(scale)
    rh, ra = c[0] * lam_h, c[1] * lam_a
    prof = eff.profile
    probs = prof.scenario_weights(period, minute)
    hg = np.full(n_sims, int(home_goals))
    ag = np.full(n_sims, int(away_goals))
    if probs.sum() > 0:
        sc = rng.choice(len(probs), n_sims, p=probs)
        bounds = prof.bounds(prof.s1[sc], prof.s2[sc])
        start = prof.played_time(period, minute, prof.s1[sc])
        wb = prof.weights
        cap_h, cap_a = home_goals + max_remaining, away_goals + max_remaining

        def rate_fn(u, h, a, idx):
            b = (u[:, None] >= bounds[idx, 1:-1]).sum(1)
            return (np.where(h < cap_h, rh * wb[b] * eff.m(h - a), 0.0),
                    np.where(a < cap_a, ra * wb[b] * eff.m(a - h), 0.0))

        bound = (rh + ra) * wb.max() * eff.multipliers.max()
        _thinning(start, bounds[:, -1], hg, ag, rate_fn, bound, rng, log)
    w = dc_tau(hg, ag, lam_h, lam_a, rho) if rho != 0.0 else np.ones(n_sims)
    out = _mc_estimates(hg, ag, w, lines)
    out["scale"] = c
    return out
