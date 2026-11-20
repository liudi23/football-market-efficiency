# In-play Step 2: Monte Carlo vs analytic

Pre-match rates: home 1.5, away 1.1; Dixon–Coles rho = -0.08; 200,000 simulated matches per state (fixed seeds).

The simulator draws goal times as a point process (thinning) from the current state; the Dixon–Coles correction is applied by importance weighting. z = (MC − analytic) / MC standard error.

|   minute | score   | home (analytic / MC)   | draw            | away            | over 2.5        |   max |z| |
|---------:|:--------|:-----------------------|:----------------|:----------------|:----------------|----------:|
|        0 | 0-0     | 0.4544 / 0.4542        | 0.2773 / 0.2777 | 0.2683 / 0.2681 | 0.4816 / 0.4836 |      1.85 |
|       30 | 0-0     | 0.3911 / 0.3901        | 0.3648 / 0.3661 | 0.2441 / 0.2437 | 0.2510 / 0.2524 |      1.5  |
|       30 | 1-0     | 0.7280 / 0.7267        | 0.1946 / 0.1960 | 0.0774 / 0.0773 | 0.5197 / 0.5185 |      1.49 |
|       45 | 1-0     | 0.7451 / 0.7452        | 0.1974 / 0.1973 | 0.0575 / 0.0575 | 0.3777 / 0.3780 |      0.3  |
|       60 | 0-1     | 0.0686 / 0.0693        | 0.2554 / 0.2545 | 0.6760 / 0.6762 | 0.2228 / 0.2230 |      1.24 |
|       75 | 1-1     | 0.1789 / 0.1793        | 0.6943 / 0.6937 | 0.1268 / 0.1270 | 0.3343 / 0.3351 |      0.78 |
|       45 | 2-0     | 0.9432 / 0.9434        | 0.0472 / 0.0469 | 0.0096 / 0.0096 | 0.7275 / 0.7275 |      0.5  |
|       85 | 0-0     | 0.0627 / 0.0632        | 0.8935 / 0.8929 | 0.0439 / 0.0439 | 0.0004 / 0.0005 |      1.08 |

Across all 32 comparisons: mean z = 0.174, mean z² = 0.645 (≈ 1 if the standard errors are honest), max |z| = 1.85.
