# In-play Step 4: score effects

Data: StatsBomb open data, 1517 matches (2015/16 Premier League, La Liga, Serie A, Ligue 1). Model: each team's goals follow a Poisson process with log-rate = intercept + home + attack[team] − defence[opponent] + time bin + score state, fitted by maximum likelihood on segments of constant time bin and score.

## 1. Score-state multipliers

Multiplier on a team's scoring rate relative to a level score, by goal difference from that team's point of view (95% CIs).

**Tactical** (team strengths estimated in-sample, i.e. using these same matches):

| goal difference (scoring team's view)   |   exposure_min |   goals |   multiplier |   ci_low |   ci_high |
|:----------------------------------------|---------------:|--------:|-------------:|---------:|----------:|
| ≤ −2                                    |          24253 |     315 |        1.17  |    1.028 |     1.332 |
| −1                                      |          50391 |     655 |        1.093 |    0.995 |     1.201 |
| 0 (level)                               |         139410 |    1770 |        1     |    1     |     1     |
| +1                                      |          50391 |     791 |        0.992 |    0.908 |     1.084 |
| ≥ +2                                    |          24253 |     466 |        0.934 |    0.833 |     1.047 |

Likelihood-ratio test against no score effects: LR = 11.6 on 4 df, p = 0.0208.

**Predictive** (team strengths cross-fitted on other matches, so the effects are measured relative to an *estimated* strength, as a live pricer must use them):

| goal difference (scoring team's view)   |   exposure_min |   goals |   multiplier |   ci_low |   ci_high |
|:----------------------------------------|---------------:|--------:|-------------:|---------:|----------:|
| ≤ −2                                    |          24253 |     315 |        1.155 |    1.019 |     1.308 |
| −1                                      |          50391 |     655 |        1.083 |    0.987 |     1.188 |
| 0 (level)                               |         139410 |    1770 |        1     |    1     |     1     |
| +1                                      |          50391 |     791 |        1.027 |    0.941 |     1.12  |
| ≥ +2                                    |          24253 |     466 |        1.025 |    0.92  |     1.141 |

The score is also evidence about strength: a team that leads is probably better than estimated, which offsets the tactical slowdown of leaders. Trailing teams score more under both.

## 2. Why the estimation method matters

| method                                         |   ≤ −2 |   −1 |   +1 |   ≥ +2 |
|:-----------------------------------------------|-------:|-----:|-----:|-------:|
| **Tactical:** full likelihood + team strengths |   1.17 | 1.09 | 0.99 |   0.93 |
| **Predictive:** + out-of-fold team strengths   |   1.15 | 1.08 | 1.03 |   1.02 |
| Naive: full likelihood, no team strengths      |   0.97 | 0.99 | 1.15 |   1.33 |
| Invalid: conditional on each team's goals      |   9.6  | 2.85 | 0.26 |   0.08 |

- **Naive:** stronger teams both lead more often and score more, so without team strengths 'leading' looks like it *raises* scoring. Pure confounding.
- **Conditional:** conditioning on each team's goal total in the match is invalid here because the score states are created by the team's own goals. A team that scored once scored it while not yet ahead and, by construction, never scores again while leading. The effects come out wildly overstated.
- The simulation tests in `tests/test_inplay.py` reproduce both biases on synthetic leagues with known effects, and show the correct method recovers the truth.

## 3. Time shape before and after removing score effects

Rate in each bin relative to minutes 0–15.

| bin   |   Step 3 (gross) |   Step 4 (net of score effects) |
|:------|-----------------:|--------------------------------:|
| 0-15  |            1     |                           1     |
| 15-30 |            1.118 |                           1.109 |
| 30-45 |            1.16  |                           1.143 |
| 45+   |            1.524 |                           1.485 |
| 45-60 |            1.27  |                           1.257 |
| 60-75 |            1.261 |                           1.237 |
| 75-90 |            1.39  |                           1.366 |
| 90+   |            1.499 |                           1.49  |

Most of the late rise in goals survives: it is a time effect, not a score effect.

## 4. Do score effects help out of sample?

5-fold cross-validation by match. Held-out Poisson log-likelihood of each team's goals, with minus without score effects (positive = score effects help). Bootstrap 95% CI over matches.

| variant                                 |   Δ log-lik per match | 95% CI             |   total |
|:----------------------------------------|----------------------:|:-------------------|--------:|
| tactical (in-sample team strengths)     |               -0.0055 | -0.0105 to -0.0002 |    -8.4 |
| predictive (out-of-fold team strengths) |               -0.0005 | -0.0042 to +0.0031 |    -0.8 |

- **Tactical** effects are significant in-sample but make held-out predictions worse: with team strengths estimated from other matches, the score carries information about strength that the tactical multipliers push the wrong way.
- **Predictive** effects neither help nor hurt goal-rate prediction here. The effects are small relative to one season's noise; Step 6 tests whether they improve in-play forecasts of final results.

## 5. Price impact (pre-match expected goals 1.5 v 1.1, ρ = 0)

Both use the predictive model's time shape; Step 4 adds its multipliers and recalibrates so pre-match expected goals stay 1.5 and 1.1 (scale factors 0.9722, 0.9621).

| state    | Step 3 H/D/A          | Step 4 H/D/A          |   Step 3 over 2.5 |   Step 4 over 2.5 |
|:---------|:----------------------|:----------------------|------------------:|------------------:|
| 1-0, 30' | 0.735 / 0.182 / 0.083 | 0.723 / 0.195 / 0.083 |             0.558 |             0.565 |
| 1-0, 70' | 0.815 / 0.159 / 0.027 | 0.807 / 0.167 / 0.027 |             0.166 |             0.17  |
| 0-1, 70' | 0.052 / 0.216 / 0.732 | 0.053 / 0.229 / 0.719 |             0.166 |             0.17  |
| 2-0, 60' | 0.958 / 0.036 / 0.006 | 0.952 / 0.042 / 0.006 |             0.635 |             0.651 |
| 0-2, 60' | 0.015 / 0.067 / 0.918 | 0.017 / 0.078 / 0.905 |             0.635 |             0.658 |
| 0-0, 80' | 0.189 / 0.678 / 0.134 | 0.184 / 0.687 / 0.129 |             0.01  |             0.01  |
| 1-1, 85' | 0.134 / 0.770 / 0.096 | 0.130 / 0.777 / 0.092 |             0.245 |             0.238 |

## Caveats

- One season of four leagues (2015/16); team strength is one attack and one defence parameter per team for the season.
- Effects are assumed the same for home and away teams and at every minute (trailing late may matter more than trailing early).
- Red cards are not yet modelled (Step 5) and are partly mixed into these estimates.
- Whether score effects improve in-play forecasts of final results is decided in Step 6.
- Data: StatsBomb open data (attribution required when publishing).
