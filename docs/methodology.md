# Methodology

## Question

**Null hypothesis:** the betting market is efficient. Pinnacle's margin-free closing
odds are the best available forecast, and no strategy beats the price available at
bet time once the margin is paid.

**Approach:** treat the market as the prior and model only what can be added to it.
This is market-anchored residual modelling, not out-predicting the bookmaker from
scratch. It follows from three findings:
- Ratings built from odds outperform ratings built from results (Wunderlich & Memmert 2018).
- A model can profit while being *less* accurate than the market if it is decorrelated
  from it (Hubáček et al. 2019; Hubáček & Šír 2023).
- Betting returns are not a valid measure of accuracy (Wunderlich & Memmert 2020).

## Data and splits

The source is football-data.co.uk.
- Pinnacle pre-closing and closing 1X2 odds exist from 2012/13.
- From 2019/20, every bookmaker has an "after market opening" set and a closing set
  (closing columns carry a "C" suffix).
- Closing odds are used **only** to evaluate bets, never as model inputs.

| Split | Seasons | Use |
|---|---|---|
| Warm-up | 2005/06 – 2011/12 | Rating burn-in only |
| Development | 2012/13 – 2018/19 | Fit and tune freely |
| Validation | 2019/20 – 2022/23 | Choose between models (full opening and closing odds) |
| **Locked holdout** | 2023/24 onwards | Touched once, at the end |

Leagues: pooled across divisions, with results reported per league. Market efficiency
varies by league (Angelini & De Angelis 2019; Oikonomidis et al. 2015), so lower
divisions matter.

## Pipeline

0. **Simulation first** (`scripts/sim_study.py`, as in Garnica-Caparrós et al. 2022).
   - Generate synthetic leagues and markets with known truth and inject edges of known size.
   - The pipeline must recover the parameters, keep a low false-positive rate in the
     null world, and find large injected edges.
   - These runs set the betting threshold and the Kelly shrinkage.
1. **Market prior:** remove the margin (multiplicative, power, Shin) and keep the
   best-calibrated method.
2. **Independent signals:**
   - Elo and pi-ratings (Constantinou & Fenton 2013), mapped to 1X2 with an ordered logit
   - Dixon–Coles with time decay; upgrade to a dynamic bivariate Poisson model
     (Koopman & Lit 2015)
   - Shot-based ratings (GAP, Wheatcroft 2020/21) and xG (from Understat; FBref's
     advanced stats were removed in January 2026)
   - Gradient boosting on rating features (the winning recipe of the 2017 Soccer
     Prediction Challenge)
3. **Combine with the market:**
   - A log-linear blend learns how much weight the model earns next to the market.
   - Optional experiment: a decorrelation loss.
4. **Evaluation:**
   - Ranked probability score (main), log loss and Brier score, plus calibration and ECE.
   - Paired block bootstrap by match date for confidence intervals.
   - Select models by calibration, not accuracy (Walsh & Joshi 2024).
5. **Betting:**
   - Bet at opening odds when EV exceeds the threshold.
   - Stake with Kelly on a probability shrunk toward the market (Baker & McHale 2013),
     capped at 2% of bankroll.
   - The edge metric is closing line value; profit and loss is reported last, with
     confidence intervals.
6. **Markets:** 1X2 and over/under 2.5 goals. Wheatcroft found the over/under market
   profitable with shot-based ratings.

## In-play

The in-play extension prices 1X2 and over/under markets at any minute from the current match
state. The plan, with the order of steps and success criteria, is in
[inplay-plan.md](inplay-plan.md). Code: `src/sportsq/inplay.py`.

### Step 1: analytic baseline (done)

- **Inputs:**
  - pre-match goal rates $\lambda_h, \lambda_a$ from Dixon–Coles
  - current score $(h, a)$
  - minutes played $m$
- **Remaining goals:** with $r = (90 - m)/90$, they are
  $X \sim \text{Poisson}(\lambda_h r)$ and $Y \sim \text{Poisson}(\lambda_a r)$,
  independent. The final score is $(h + X, a + Y)$, and prices are read off that
  final-score grid.
- **Dixon–Coles correction in-play.** The low-score factor $\tau$ is defined on the
  *full-match* score.
  - Assume that, given the final counts, goal times are i.i.d. uniform over the match.
    This is the same assumption that justifies constant rates.
  - Then conditioning the full-match Dixon–Coles distribution on the score at minute $m$
    gives exactly
    $P(\text{final}=(x,y)) \propto \text{Pois}(x-h;\lambda_h r)\,\text{Pois}(y-a;\lambda_a r)\,\tau(x,y)$,
    by Bayes' rule with Poisson thinning.
  - So $\tau$ is applied to the *final* score cells, using full-match rates, and the
    grid is renormalised.
  - A unit test confirms this against brute-force conditioning.
  - Consequences: prices at minute 0 equal the pre-match prices exactly; $\tau$ stops
    mattering once a final count above 1 is certain; and $\rho = 0$ reduces to
    independent Poisson.
- **Stoppage time** is not modelled in Step 1: minutes are clipped to [0, 90]. Step 3
  revisits this.
- **Known limitations:**
  - Goal rates are constant within a match (fixed in Step 3).
  - The score doesn't change behaviour (Step 4).
  - Red cards are ignored (Step 5).

### Step 2: Monte Carlo match-state simulator (done)

- **Engine:** goals for the rest of the match are simulated as event times of a point
  process from the current state, using thinning (Lewis & Shedler 1979).
  - Candidate events arrive at a constant bound rate.
  - Each is accepted as a home goal, an away goal or rejected, according to the
    current rates.
  - Rates come from a function of time and match state, which is where Steps 3–5
    plug in. With constant rates the bound equals the total rate, so nothing is
    rejected.
- **Dixon–Coles correction:** reproduced by self-normalised importance weighting. Each
  simulated match is weighted by $\tau$(final score), with full-match rates, which is
  exactly the Step 1 posterior.
  - Standard errors use the delta method.
  - The effective sample size is reported.
- **Verification** (`tests/test_inplay.py`, `reports/inplay_mc_check.md`):
  - 8 states × 2 rate pairs × 3 values of $\rho$, with 200,000 simulations each, compared
    with the analytic prices in units of Monte Carlo standard error.
  - Each run must pass three checks: max |z| < 4.5, |mean z| < 0.6, and
    0.4 < mean z² < 1.8. Since Step 3, cells with fewer than 30 expected hits use an
    exact Poisson count test instead of a z-score.
  - These tolerances were set from 90 runs with other seeds. The z-scores are
    correlated, because outcomes and nested over lines share simulated matches.
  - The checks then passed on 180 further fresh seeds, and still fail if the standard
    errors are off by a factor of 2 (mean z² 4.0 or 0.25).
  - **Negative control:** simulating without the weights while comparing against
    $\rho = -0.15$ prices fails with max |z| above 100.

### Step 3: time-varying goal rates and stoppage time (done)

- **Data:** StatsBomb open data, 1,517 matches from the complete 2015/16 Premier League,
  La Liga, Serie A and Ligue 1 seasons, with 3,997 goals
  (`scripts/download_statsbomb.py`).
- **Clock:** a state is (period, minute) on the StatsBomb clock.
  - First-half stoppage is minute > 45 in period 1; the second half restarts at 45.
  - Internally, both halves are laid end to end on a played-time axis
    $u \in [0, 90 + s_1 + s_2]$.
- **Intensity:** piecewise constant, in 15-minute bins plus one stoppage bin per half.
  - Each team's rate in bin $b$ is $\lambda_\text{team} w_b$.
  - The weights are normalised so that the expected integrated intensity over a match is 1.
    So $\lambda$ stays the expected full-match goals from Dixon–Coles, which is fitted on
    full-time scores including stoppage goals.
  - Per-bin rates are maximum-likelihood estimates (goals ÷ exposure minutes), with exact
    Poisson CIs.
- **Stoppage time:**
  - The lengths $(s_1, s_2)$ follow their empirical joint distribution, as 131 scenarios
    at 0.5-minute resolution (means 1.3 and 3.8 minutes).
  - During stoppage, scenarios are conditioned on the stoppage already played.
  - Prices are the scenario mixture.
- **Dixon–Coles correction:** both teams share one shape, so, given the final counts,
  goal times are i.i.d. with density proportional to the intensity.
  - The Step 1 derivation then holds, with "share of the match played" replaced by
    "share of the integrated intensity used".
  - Scenario grids are mixed before normalising, so the scenario weights are updated by
    the $\tau$ likelihood, exactly as in the Monte Carlo estimator.
- **Findings** (`reports/inplay_intensity.md`):
  - The rate rises from 0.82× the match average (minutes 0–15; 95% CI 0.75–0.90) to
    1.14× (75–90; 1.06–1.23), and about 1.23–1.25× in stoppage time.
  - Home and away goals share the shape: χ² homogeneity test p = 0.57.
  - **Out of sample:** in 5-fold cross-validated log-likelihood of goal times, 15-minute
    bins beat a constant rate by +0.012 per match (95% CI +0.003 to +0.022).
  - 5-minute bins are *worse* than 15-minute bins (−0.007, CI −0.013 to −0.001), because
    they overfit. So 15-minute bins are the default.
  - **Price impact:** at 0–0 after 80 minutes the draw falls from 80% to 72%. At 0–0
    after 90 minutes, Step 1 treats the match as over, while Step 3 still gives a 9%
    chance of a stoppage-time goal.
- **Verification** (`tests/test_inplay.py`):
  - A constant profile reproduces Step 1 exactly.
  - At kick-off, prices equal the pre-match prices exactly when stoppage time is fixed
    at its mean, and are within 0.002 when it's random.
  - The fit recovers a known shape from simulated goal times (within 2–5% across seeds).
  - Monte Carlo pricing agrees with the analytic mixture: 0 failures in 220 fresh-seed
    runs, average mean z² of 0.99.
- **Test fix found while doing this:**
  - Very rare events (expected only a handful of times in 200,000 simulations) made the
    z-score checks unreliable.
  - Cells with fewer than 30 expected hits are now checked with an exact Poisson count
    test instead, in both the Step 2 and Step 3 tests.
- **Caveats:**
  - One season, inside the development split.
  - The late rise partly reflects score effects (trailing teams attack), which Step 4
    will separate out.

### Step 4: score effects (done)

- **Model:** each team's rate is
  $c\,\lambda_\text{team}\,w_b(u)\,m(d)$, where:
  - $m(d)$ is a multiplier for the goal difference from that team's point of view,
    with states ≤−2, −1, 0, +1, ≥+2 and $m(0) = 1$ (Dixon & Robinson 1998);
  - $w_b$ is the time shape net of score effects;
  - $c$ is a per-match calibration so that pre-match expected goals still equal the
    Dixon–Coles $\lambda$s.
- **Estimation:** full Poisson-process likelihood on segments of constant (time bin,
  score), with team attack/defence and home advantage.
  - It's valid even though the score depends on earlier goals, because the intensity
    is predictable.
  - **Two pitfalls, both demonstrated on simulated leagues in the tests:**
    - Leaving out team strength flips the sign: leaders appear to score *more*,
      because strong teams lead.
    - Conditioning on each team's goal total in the match is invalid, because the
      states are created by the team's own goals. It overstated the effects about
      ten-fold (trailing by 2+: 9.6×).
- **Tactical vs predictive:**
  - With in-sample team strengths, the multipliers measure the *tactical* effect:
    ≤−2: 1.17 (1.03–1.33); −1: 1.09; +1: 0.99; ≥+2: 0.93. LR test p = 0.02.
  - But tactical multipliers make held-out goal predictions *worse* (−0.0055 per match,
    CI −0.010 to −0.000). A live pricer only has an estimated strength, and the score
    is evidence about that estimate's error: a leader is probably stronger than thought.
  - **Predictive** multipliers, fitted on out-of-fold strength offsets: ≤−2: 1.16
    (1.02–1.31); −1: 1.08; +1: 1.03; ≥+2: 1.03.
  - Out of sample they are neutral (−0.0005 per match, CI −0.004 to +0.003).
  - The predictive version is the one to price with. A simulation test confirms the
    learning effect: predictive multipliers for leaders sit 4–12% above the tactical
    ones.
  - The time shape barely changes when score effects are removed: the late rise in
    goals is a time effect.
- **Pricing:**
  - Rates now depend on the score, so remaining goals are not Poisson. Exact prices
    come from the continuous-time Markov chain over scores: a matrix exponential per
    time bin per stoppage scenario.
  - Calibration uses the mean-stoppage profile, which is ~100× faster and within about
    1e-5 in expected goals.
  - The Dixon–Coles $\tau$ is kept as an optional final-score reweighting. Its Step 1
    justification no longer holds, so $\rho = 0$ is recommended with score effects.
- **Price impact** with predictive multipliers is small: at most about 1–1.5 points on
  1X2 in the examples (`reports/inplay_score_effects.md`).
- **Verification:**
  - With no effects, the pricer reproduces Step 3 to 1e-6.
  - Calibration preserves expected goals.
  - Monte Carlo matches the Markov chain: 0 failures in 50 fresh-seed runs.
  - The fit recovers known effects on simulated leagues (max |z| below 3 across 6 seeds).
- **Caveats:**
  - One season.
  - Effects are the same for home and away, and constant over time.
  - Red cards are not yet separated out (Step 5).
  - A sharper strength offset would be the pre-match market odds (Pinnacle 2015/16 is
    in football-data). That's a natural refinement.
  - Step 6 decides whether score effects earn their place.

## Pre-registered success criteria (holdout)

- **Forecast:** the blend's RPS is below the opening market's, with a 95% CI below 0.
- **Edge:** mean CLV is above 0, with a 95% CI above 0.
- **Profit:** reported with its CI; not used to choose models.
- A null result is a valid outcome. If the holdout disappoints, it is reported, not re-tuned.

## Guarding against false discoveries

- Log every experiment.
- Run permutation tests (shuffle results).
- Use the simulation null world to calibrate how often the pipeline "finds" an edge
  that does not exist.

## Practical caveat

Kaunitz et al. (2017) profited from mispriced soft-bookmaker odds but were
quickly limited. A backtest edge at soft books is not an executable strategy;
CLV against Pinnacle and exchange prices is the honest test.

## References

- Angelini & De Angelis (2019). Efficiency of online football betting markets. *IJF* 35(2).
- Baker & McHale (2013). Optimal betting under parameter uncertainty: improving the Kelly criterion. *Decision Analysis* 10(3).
- Berrar, Lopes & Dubitzky (2019). Incorporating domain knowledge in machine learning for soccer outcome prediction. *Machine Learning* 108.
- Constantinou & Fenton (2013). Determining the level of ability of football teams by dynamic ratings (pi-ratings). *JQAS* 9(1).
- Dixon & Coles (1997). Modelling association football scores and inefficiencies in the football betting market. *JRSS C*.
- Garnica-Caparrós, Memmert & Wunderlich (2022). Artificial data in sports forecasting. *ISeB* 20.
- Hubáček, Šourek & Železný (2019). Exploiting sports-betting market using machine learning. *IJF* 35(2).
- Hubáček & Šír (2023). Beating the market with a bad predictive model. *IJF* 39(2).
- Kaunitz, Zhong & Kreiner (2017). Beating the bookies with their own numbers. arXiv:1710.02824.
- Koopman & Lit (2015). A dynamic bivariate Poisson model for the English Premier League. *JRSS A* 178(1); (2019) score-driven models, *IJF* 35(2).
- Walsh & Joshi (2024). Machine learning for sports betting: accuracy or calibration? *MLWA* 16.
- Wheatcroft (2020). A profitable model for predicting the over/under market in football. *IJF* 36(3); (2021) Forecasting football matches by predicting match statistics. *JSA* 7(2).
- Lewis & Shedler (1979). Simulation of nonhomogeneous Poisson processes by thinning. *Naval Research Logistics Quarterly* 26(3).
- Dixon & Robinson (1998). A birth process model for association football matches. *JRSS D* 47(3).
- Wunderlich & Memmert (2018). The Betting Odds Rating System. *PLOS ONE* 13(6); (2020) Are betting returns a useful measure of accuracy? *IJF* 36(2).
