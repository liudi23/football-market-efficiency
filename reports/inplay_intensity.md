# In-play Step 3: time-varying goal intensity

Data: StatsBomb open data, 1517 matches (2015/16 Premier League, La Liga, Serie A, Ligue 1); 3997 goals; 2.632 goals per match.

## 1. Goal intensity by time bin

Relative rate = goals per minute in the bin ÷ goals per minute over all play (incl. stoppage). CIs are exact Poisson intervals.

| bin   |   goals |   exposure_min |   goals_per_90 |   relative_rate |   rel_ci_low |   rel_ci_high |
|:------|--------:|---------------:|---------------:|----------------:|-------------:|--------------:|
| 0-15  |     518 |        22755   |          2.049 |           0.822 |        0.753 |         0.896 |
| 15-30 |     579 |        22755   |          2.29  |           0.919 |        0.846 |         0.997 |
| 30-45 |     601 |        22755   |          2.377 |           0.954 |        0.879 |         1.033 |
| 45+   |      70 |         2017.4 |          3.123 |           1.253 |        0.977 |         1.583 |
| 45-60 |     658 |        22755   |          2.603 |           1.044 |        0.966 |         1.127 |
| 60-75 |     653 |        22755   |          2.583 |           1.036 |        0.958 |         1.119 |
| 75-90 |     720 |        22755   |          2.848 |           1.143 |        1.061 |         1.229 |
| 90+   |     198 |         5801.5 |          3.072 |           1.233 |        1.067 |         1.417 |

Stoppage time: first half mean 1.33 min, second half mean 3.82 min; modelled as 131 joint scenarios (0.5-minute resolution).

## 2. Same shape for home and away goals?

The in-play Dixon–Coles correction assumes both teams share one time shape.

| bin   |   home goals |   home relative rate |   away goals |   away relative rate |
|:------|-------------:|---------------------:|-------------:|---------------------:|
| 0-15  |          290 |                0.805 |          228 |                0.844 |
| 15-30 |          316 |                0.878 |          263 |                0.974 |
| 30-45 |          359 |                0.997 |          242 |                0.896 |
| 45+   |           39 |                1.222 |           31 |                1.295 |
| 45-60 |          392 |                1.089 |          266 |                0.985 |
| 60-75 |          368 |                1.022 |          285 |                1.055 |
| 75-90 |          404 |                1.122 |          316 |                1.17  |
| 90+   |          116 |                1.264 |           82 |                1.191 |

Chi-square test that the home share of goals is the same in every bin: χ² = 5.8 on 7 df, p = 0.565.

## 3. Does time variation help out of sample?

5-fold cross-validated log-likelihood of goal times (Poisson process, both teams pooled, same goals-per-match for every model so only the time shape differs). Positive Δ = better than a constant rate per played minute. Bootstrap CI over matches.

| model vs constant   |   Δ log-lik per match | 95% CI             |   total Δ log-lik |
|:--------------------|----------------------:|:-------------------|------------------:|
| 15-min bins         |                0.0123 | 0.0032 to 0.0216   |              18.7 |
| 5-min bins          |                0.0055 | -0.0053 to 0.0165  |               8.4 |
| 5-min minus 15-min  |               -0.0068 | -0.0130 to -0.0008 |             -10.3 |

## 4. Price impact (pre-match expected goals 1.5 v 1.1, ρ = −0.08)

Step 1 assumes a constant rate over exactly 90 minutes; Step 3 uses the fitted shape and random stoppage time.

| state                 | Step 1 H/D/A          | Step 3 H/D/A          |   Step 1 over 2.5 |   Step 3 over 2.5 |
|:----------------------|:----------------------|:----------------------|------------------:|------------------:|
| 0-0, 30'              | 0.391 / 0.365 / 0.244 | 0.405 / 0.345 / 0.250 |             0.251 |             0.292 |
| 0-0, 45' (1st half)   | 0.342 / 0.438 / 0.220 | 0.366 / 0.402 / 0.232 |             0.142 |             0.189 |
| 0-0, 45+2' (1st half) | 0.342 / 0.438 / 0.220 | 0.364 / 0.405 / 0.231 |             0.142 |             0.185 |
| 0-0, 60'              | 0.271 / 0.549 / 0.180 | 0.300 / 0.504 / 0.196 |             0.056 |             0.083 |
| 0-0, 80'              | 0.117 / 0.802 / 0.081 | 0.166 / 0.720 / 0.114 |             0.003 |             0.01  |
| 1-0, 85'              | 0.937 / 0.062 / 0.002 | 0.890 / 0.104 / 0.006 |             0.01  |             0.036 |
| 0-0, 90'              | 0.000 / 1.000 / 0.000 | 0.055 / 0.906 / 0.039 |             0     |             0     |
| 1-1, 90+2'            | 0.000 / 1.000 / 0.000 | 0.032 / 0.945 / 0.023 |             0     |             0.056 |
| 0-0, 90+5'            | 0.000 / 1.000 / 0.000 | 0.009 / 0.985 / 0.006 |             0     |             0     |

## Caveats

- One season of four leagues (2015/16), inside the development split.
- The late rise in goals partly reflects score effects (trailing teams attack, leaders defend). Step 4 models those explicitly; until then this shape absorbs them.
- Data: StatsBomb open data (attribution required when publishing).
