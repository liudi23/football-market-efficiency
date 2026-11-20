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


def inplay_grid(lam_h, lam_a, home_goals=0, away_goals=0, minute=0.0, rho=0.0,
                max_remaining=10, match_minutes=MATCH_MINUTES) -> np.ndarray:
    """Final-score probabilities given the state at `minute`.

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
    g = np.arange(size)
    # final x has probability Pois(x - h; lambda * r); pmf is 0 for x < h
    ph = poisson.pmf(g[None, :] - hg[:, None], (lam_h * r)[:, None])
    pa = poisson.pmf(g[None, :] - ag[:, None], (lam_a * r)[:, None])
    # mu = 0 (match over): scipy gives pmf(0, 0) = 1, but guard explicitly
    done = r == 0
    if done.any():
        ph[done] = (g[None, :] == hg[done, None]).astype(float)
        pa[done] = (g[None, :] == ag[done, None]).astype(float)
    grid = ph[:, :, None] * pa[:, None, :]
    if rho != 0.0:
        grid[:, 0, 0] *= 1 - lam_h * lam_a * rho
        grid[:, 0, 1] *= 1 + lam_h * rho
        grid[:, 1, 0] *= 1 + lam_a * rho
        grid[:, 1, 1] *= 1 - rho
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
