# In-play Step 6: evaluation by minute — synthetic season (pipeline check)

Matches: 300; evaluation states: 2215. Skipped: {'no kick-off price': 0, 'after goal': 185, 'missing price': 0}. Kick-off market fit (constant-rate model): RMS probability error 0.0036.

Models: **M1** constant rates (Step 1); **M3** time-varying rates + stoppage (Step 3); **M4** M3 + predictive score effects (Step 4); **MKT** Betfair market. Red cards (Step 5) are not modelled.

Each model's pre-match rates are fitted so that its own kick-off prices match the market.

## 1. Accuracy by minute (lower is better)

| minute   |   n |   RPS MKT |   RPS M1 |   RPS M3 |   RPS M4 |   Brier o2.5 MKT |   Brier o2.5 M1 |   Brier o2.5 M3 |   Brier o2.5 M4 |
|:---------|----:|----------:|---------:|---------:|---------:|-----------------:|----------------:|----------------:|----------------:|
| 0'       | 300 |    0.2287 |   0.2285 |   0.2285 |   0.2287 |           0.2505 |          0.2506 |          0.2506 |          0.2506 |
| 15'      | 278 |    0.2084 |   0.2077 |   0.2079 |   0.2083 |           0.2368 |          0.2362 |          0.2368 |          0.237  |
| 30'      | 282 |    0.1935 |   0.1934 |   0.1934 |   0.1935 |           0.2212 |          0.2221 |          0.2223 |          0.223  |
| HT       | 265 |    0.1687 |   0.1697 |   0.1693 |   0.1695 |           0.1715 |          0.1704 |          0.1717 |          0.1724 |
| 60'      | 278 |    0.1352 |   0.1351 |   0.1349 |   0.1352 |           0.1575 |          0.1585 |          0.1581 |          0.1582 |
| 75'      | 266 |    0.1006 |   0.1025 |   0.1009 |   0.1009 |           0.1071 |          0.1056 |          0.1078 |          0.1083 |
| 85'      | 268 |    0.0647 |   0.0675 |   0.065  |   0.0649 |           0.066  |          0.0665 |          0.066  |          0.066  |
| 90'      | 278 |    0.0287 |   0.0306 |   0.0287 |   0.0287 |           0.028  |          0.0288 |          0.0279 |          0.0278 |

## 2. Does each added effect help? (paired, bootstrap by match)

**Decision rule** (revised before any real evaluation data was seen; see docs/inplay-plan.md): keep an effect if the RPS difference pooled over the in-play minutes (15'–90') has a 95% CI below 0 **and** no single minute is significantly worse. The original rule (CI below 0 at *most* minutes) rejected the true model in the synthetic check, because time effects matter only late in matches.

### Step 3: time-varying rates + stoppage (M3 − M1)

| minute   |   n |    mean |   ci_low |   ci_high | better?             |
|:---------|----:|--------:|---------:|----------:|:--------------------|
| 0'       | 300 |  0      |  -0      |    0      | no clear difference |
| 15'      | 278 |  0.0002 |  -0.0002 |    0.0006 | no clear difference |
| 30'      | 282 | -0      |  -0.0008 |    0.0007 | no clear difference |
| HT       | 265 | -0.0004 |  -0.0014 |    0.0007 | no clear difference |
| 60'      | 278 | -0.0003 |  -0.0016 |    0.001  | no clear difference |
| 75'      | 266 | -0.0015 |  -0.0037 |    0.0007 | no clear difference |
| 85'      | 268 | -0.0025 |  -0.0049 |   -0.0004 | yes (CI < 0)        |
| 90'      | 278 | -0.0019 |  -0.0036 |   -0.0003 | yes (CI < 0)        |

Pooled over in-play minutes (300 matches): -0.00098 (95% CI -0.00188 to -0.00009); minutes significantly worse: 0. → **KEEP**

(Original rule, for reference: CI below 0 at 2 of 7 minutes.)

### Step 4: score effects (M4 − M3)

| minute   |   n |    mean |   ci_low |   ci_high | better?             |
|:---------|----:|--------:|---------:|----------:|:--------------------|
| 0'       | 300 |  0.0002 |  -0.0002 |    0.0006 | no clear difference |
| 15'      | 278 |  0.0004 |  -0.0003 |    0.0011 | no clear difference |
| 30'      | 282 |  0.0001 |  -0.0006 |    0.0008 | no clear difference |
| HT       | 265 |  0.0002 |  -0.0005 |    0.0009 | no clear difference |
| 60'      | 278 |  0.0003 |  -0.0002 |    0.0008 | no clear difference |
| 75'      | 266 | -0.0001 |  -0.0004 |    0.0003 | no clear difference |
| 85'      | 268 | -0.0001 |  -0.0003 |    0.0001 | no clear difference |
| 90'      | 278 | -0      |  -0.0001 |    0      | no clear difference |

Pooled over in-play minutes (300 matches): +0.00015 (95% CI -0.00025 to +0.00053); minutes significantly worse: 0. → **DROP (not shown to help)**

(Original rule, for reference: CI below 0 at 0 of 7 minutes.)

## 3. Model versus market

RPS difference model − market (negative = model more accurate than the market).

**M1 − MKT**

| minute   |   n |    mean |   ci_low |   ci_high | better?             |
|:---------|----:|--------:|---------:|----------:|:--------------------|
| 0'       | 300 | -0.0002 |  -0.0005 |    0.0001 | no clear difference |
| 15'      | 278 | -0.0006 |  -0.0023 |    0.001  | no clear difference |
| 30'      | 282 | -0.0001 |  -0.0017 |    0.0015 | no clear difference |
| HT       | 265 |  0.001  |  -0.0008 |    0.0028 | no clear difference |
| 60'      | 278 | -0.0001 |  -0.0017 |    0.0016 | no clear difference |
| 75'      | 266 |  0.0019 |  -0.0004 |    0.0043 | no clear difference |
| 85'      | 268 |  0.0028 |   0.0006 |    0.0052 | worse (CI > 0)      |
| 90'      | 278 |  0.0019 |   0.0002 |    0.0037 | worse (CI > 0)      |

**M3 − MKT**

| minute   |   n |    mean |   ci_low |   ci_high | better?             |
|:---------|----:|--------:|---------:|----------:|:--------------------|
| 0'       | 300 | -0.0002 |  -0.0005 |    0.0001 | no clear difference |
| 15'      | 278 | -0.0005 |  -0.002  |    0.001  | no clear difference |
| 30'      | 282 | -0.0001 |  -0.0014 |    0.0012 | no clear difference |
| HT       | 265 |  0.0007 |  -0.0006 |    0.002  | no clear difference |
| 60'      | 278 | -0.0004 |  -0.0012 |    0.0005 | no clear difference |
| 75'      | 266 |  0.0004 |  -0.0002 |    0.001  | no clear difference |
| 85'      | 268 |  0.0003 |  -0      |    0.0006 | no clear difference |
| 90'      | 278 |  0      |  -0.0001 |    0.0001 | no clear difference |

**M4 − MKT**

| minute   |   n |    mean |   ci_low |   ci_high | better?             |
|:---------|----:|--------:|---------:|----------:|:--------------------|
| 0'       | 300 |  0      |  -0.0005 |    0.0005 | no clear difference |
| 15'      | 278 | -0.0001 |  -0.0017 |    0.0015 | no clear difference |
| 30'      | 282 | -0      |  -0.0015 |    0.0015 | no clear difference |
| HT       | 265 |  0.0009 |  -0.0005 |    0.0022 | no clear difference |
| 60'      | 278 | -0      |  -0.001  |    0.0009 | no clear difference |
| 75'      | 266 |  0.0003 |  -0.0005 |    0.0011 | no clear difference |
| 85'      | 268 |  0.0002 |  -0.0002 |    0.0006 | no clear difference |
| 90'      | 278 | -0      |  -0.0001 |    0.0001 | no clear difference |

Average distance from the market (half the sum of |p_model − p_market| over H/D/A; 0.05 = 5 points of probability moved):

| minute   |     M1 |     M3 |     M4 |
|:---------|-------:|-------:|-------:|
| 0'       | 0.0086 | 0.0086 | 0.0161 |
| 15'      | 0.0206 | 0.0188 | 0.0249 |
| 30'      | 0.0219 | 0.0167 | 0.0227 |
| HT       | 0.0252 | 0.0157 | 0.0207 |
| 60'      | 0.0292 | 0.0131 | 0.0171 |
| 75'      | 0.0442 | 0.0095 | 0.0123 |
| 85'      | 0.0489 | 0.0052 | 0.0065 |
| 90'      | 0.0549 | 0.0028 | 0.0034 |

## Caveats

- The market is last traded prices (not mid-prices), normalised; they can be stale for up to a minute and right after goals (those 3 minutes are skipped).
- Pre-match rates come from the kick-off market, so minute 0 matches it by construction; differences are purely about in-play updating.
- Goal times are rebuilt from market suspensions (±1 minute); red cards are not observed or modelled.
- Time shape and score effects were fitted on 2015/16 StatsBomb data; this evaluation fits nothing on the evaluation season.
