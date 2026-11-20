# In-play extension: plan

Status: Steps 1–4 done (20 Nov 2026); Steps 5–6 to do. Written 20 Nov 2026.

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

### Step 3 — Time-varying goal rates ✅ done

- Goal intensity rises through the match. Fit a piecewise-constant or smooth
  intensity λ(t) from goal-minute data.
- Handle stoppage time: decide how minutes 45+ and 90+ are represented, and say so.
- **Done:** 15-minute bins plus one stoppage bin per half, chosen by cross-validation
  (5-minute bins overfit). States use (period, minute) on the StatsBomb clock.
  Stoppage length is random (empirical joint distribution) and is conditioned on the
  time already played. See `reports/inplay_intensity.md` and the methodology.

### Step 4 — Score effects ✅ done

- Scale each team's rate by the score difference, e.g. the trailing team's rate
  rises and the leading team's falls, following Dixon & Robinson (1998).
- Estimate the multipliers from data and report their uncertainty.
- **Done:** full-likelihood fit with team strengths; tactical vs predictive multipliers
  (price with predictive, i.e. relative to an out-of-fold strength estimate). Effects
  are small: trailing teams score ~8–16% more; leaders ~unchanged once learning about
  strength is included. Neutral out of sample on goal rates; Step 6 decides. See
  `reports/inplay_score_effects.md`.
- **Possible refinement:** use pre-match Pinnacle odds (football-data 2015/16) as the
  strength offset instead of cross-fitted team strengths.

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
- **StatsBomb open data, downloaded 20 Nov 2026:**
  - Run `python scripts/download_statsbomb.py`. It writes two files to `data/processed/`
    (gitignored): `statsbomb_matches.csv` and `statsbomb_events.csv`. It keeps only goals,
    red cards and half lengths, and never stores the raw event files.
  - **Coverage:** the only complete league seasons are the 2015/16 Premier League,
    La Liga, Serie A and Ligue 1 (1,517 matches). Other seasons in the open data cover
    one club (mostly Barcelona) or a tournament, which would bias goal timing, so
    they are excluded by default.
  - **Contents:** 3,997 goals and 413 red cards.
  - **Checks:**
    - Goal events reproduce the final score in all 1,517 matches.
    - The Premier League has 59 red cards, matching the published season total.
  - **Half lengths, including stoppage time:** first half mean 46.3 minutes (max
    54.3); second half ends at minute 93.8 on average (max 101.2).
  - **Clock caveat:** StatsBomb restarts the second half at minute 45, so always use
    `period` with `minute`.
- **Limits and what they mean for Steps 3–6:**
  - This is one season, inside the development split. It is enough for the shape of
    the goal intensity λ(t) (Step 3) and a first estimate of score effects (Step 4).
  - Red cards (Step 5) will be noisy.
  - It **cannot** support the validation/holdout evaluation in Step 6 as planned.
  - For later seasons, use Understat shot data (goal minutes for the top-5 leagues
    since 2014/15, via the `soccerdata` package). It has no red cards, and must be
    downloaded from my own network.
- **Betfair Exchange historical data (BASIC, free), downloaded 20 Nov 2026:**
  - **Coverage:** August 2024 – late January 2025, about 974,000 market files across
    all football worldwide (4 GB tar, `data/BASIC/`, gitignored).
  - **Format:** one bz2 stream file per market, with last-traded prices at about
    1-minute snapshots plus market-definition updates.
  - **Pipeline:**
    - `scripts/betfair_index.py`: resumable index of match-odds and over/under 0.5–4.5
      markets.
    - `scripts/betfair_epl.py`: matches Premier League fixtures (219/229; the archive
      ends before the last January weekend) and rebuilds each match timeline from the
      markets alone. The scorer side comes from the match-odds move, and is fitted to
      the known final score.
  - **Checks against football-data:**
    - Goal count right in 219/219 matches.
    - Exact final score from price moves alone in 89.5%.
    - Half-time score (which tests goal timing) in 96.3%.
  - **Output:** `data/processed/betfair_epl_2425_inplay.csv`, about 22,000
    match-minutes with the score and market 1X2 / over-2.5 probabilities.
  - **Pitfalls found:**
    - Every goal (and VAR checks) suspends *all* markets, so the deciding goal is the
      *last* suspension before a market closes.
    - Some archive files are event-level bundles of every market, so the parser must
      filter by market id.
    - Folder dates are not match dates.
  - **Use in Step 6:** compare the in-play pricer with the market minute by minute,
    for the first half of 2024/25. A full season needs February–May 2025 as well.
- Keep raw data out of git (`data/raw/`, `data/processed/`, `data/BASIC/` and `*.tar`
  are ignored).
- Attribute StatsBomb when publishing results built on this data.

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
