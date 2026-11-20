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
    0.4 < mean z² < 1.8.
  - These tolerances were set from 90 runs with other seeds. The z-scores are
    correlated, because outcomes and nested over lines share simulated matches.
  - The checks then passed on 180 further fresh seeds, and still fail if the standard
    errors are off by a factor of 2 (mean z² 4.0 or 0.25).
  - **Negative control:** simulating without the weights while comparing against
    $\rho = -0.15$ prices fails with max |z| above 100.

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
