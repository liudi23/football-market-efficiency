# In-play extension: plan

Status: Steps 1–2 done (20 Nov 2026); Steps 3–6 to do. Written 20 Nov 2026.

## Context (for picking this up in a new conversation)

- **Repository:** github.com/liudi23/football-market-efficiency. The local folder is
  `~/Library/CloudStorage/Dropbox/DS_projects/sports-trading`, package `sportsq`.
- **Done so far (pre-match):**
  - margin removal (multiplicative, power, Shin)
  - Elo and pi-ratings mapped to 1X2 with an ordered logit
  - walk-forward Dixon–Coles with time decay, plus a log-linear market blend
  - shrunk-Kelly staking with closing line value (CLV)
  - a simulation study with injected edges
- **Real-data baseline** (`reports/baseline_real.md`): 42,703 matches across E0, E1, D1,
  SP1, I1 and F1, scored on the 2019/20–2022/23 validation seasons.
  - Elo RPS 0.2107, pi-ratings 0.2092, Dixon–Coles 0.2097 (second half of validation only).
  - Pinnacle opening 0.2032, closing 0.2025.
  - The blend weights are market 0.974 and model −0.006, so there is no edge and only 3 bets.
  - The 2023/24+ holdout is locked and untouched.
- **Why in-play:** target roles (Sportradar Quant, Smartodds Quant Analyst – Football,
  Longshot Systems) price pre-match **and** in-play markets. This closes that gap.
- **Separately planned:** a Java port of the score-grid pricer with JUnit tests, checked
  against the Python version. Sportradar implements its models in Java, and my Java is
  basic and self-taught.

## Goal

Price 1X2 and over/under markets at any minute, given the current match state:
the score, the minute and red cards. The prices start from the pre-match Dixon–Coles
rates. Verify every model against a simpler one before trusting it.

Use the terms "match-state simulator" or "in-play pricing model", not "digital twin".

## Steps

### Step 1 — Analytic in-play pricer (baseline, about half a day) ✅ done

- Inputs:
  - pre-match goal rates λ_h and λ_a (from Dixon–Coles)
  - current score (h, a)
  - current minute m
- Assume the remaining goals follow Poisson(λ × (90 − m)/90) for each side,
  independently. The Dixon–Coles low-score correction applies to the full match only,
  so document how it is handled in-play.
- Add the current score to the grid of remaining goals, then compute the 1X2 and
  over/under N.5 prices from that grid.
- Reuse the existing score-grid code in `dixon_coles.py`.

### Step 2 — Monte Carlo simulator that reproduces Step 1 (about half a day) ✅ done

- Simulate goal arrival times for the rest of the match as a Poisson process.
- Unit test: with constant rates and no state effects, Monte Carlo prices match the
  analytic prices within the expected Monte Carlo error. Use enough simulations that
  the tolerance is meaningful, and fix the seed.

### Step 3 — Time-varying goal rates

- Goal intensity rises through the match. Fit a piecewise-constant or smooth
  intensity λ(t) from goal-minute data.
- Handle stoppage time: decide how minutes 45+ and 90+ are represented, and say so.

### Step 4 — Score effects

- Scale each team's rate by the score difference, e.g. the trailing team's rate
  rises and the leading team's falls, following Dixon & Robinson (1998).
- Estimate the multipliers from data and report their uncertainty.

### Step 5 — Red cards

- Multiply both teams' rates after a red card (the 10-man team down, the opponent up).
- Estimate the effect from data; it will be noisy because red cards are rare.

### Step 6 — Evaluation

- No in-play odds are available, so evaluate the probabilities directly:
  - At fixed minutes (0, 15, 30, 45, 60, 75), score the in-play 1X2 probabilities
    against final results with RPS, log loss and calibration (ECE).
  - Compare the plain Poisson model (Step 1) with time-varying rates (Step 3), then
    with score effects (Step 4), then with red cards (Step 5).
  - Use paired block-bootstrap CIs, as in the pre-match work.
- Success criterion, fixed in advance: each added effect must improve RPS at most
  minutes with a 95% CI below 0. Otherwise it is dropped and reported as not helping.
- Use the same splits as the pre-match work. Do not touch the 2023/24+ holdout.

## Data

- football-data.co.uk has **no goal or card minutes**.
- Options:
  - **StatsBomb open data** (github.com/statsbomb/open-data): free event data with goal
    and card minutes, for a subset of competitions.
  - Other free goal-timing sources: check coverage and licence before use.
- Download from my own network; the Claude workspace can't reach these sites.
- Keep raw data out of git (`data/raw/` is already ignored).

## Files to add

- `src/sportsq/inplay.py`: analytic pricer, Monte Carlo simulator, intensity and state
  effects.
- `tests/test_inplay.py`:
  - Monte Carlo prices match analytic prices;
  - at minute 0 the prices equal the pre-match prices;
  - at minute 90 the prices equal the final result;
  - probabilities sum to 1.
- `scripts/inplay_eval.py`: evaluation by minute, writing `reports/inplay_eval.md`.
- A new "In-play" section in `docs/methodology.md`.
- An updated README roadmap.

## References

- Dixon & Robinson (1998). A birth process model for association football matches.
  *JRSS D (The Statistician)* 47(3).
- Dixon & Coles (1997). Modelling association football scores and inefficiencies in
  the football betting market. *JRSS C* 46(2).

## How it will be used in applications

- CV bullet, once results exist: "Extended to in-play pricing with a match-state
  simulator (time-varying rates, score and red-card effects), verified against a
  closed-form model and evaluated by minute."
- Interview story: verify-first (Monte Carlo against analytic), and keep only the
  effects that the evaluation justifies.
