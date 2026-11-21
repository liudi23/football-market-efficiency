# In-play Step 6: evaluation by minute — Premier League 22/23 (Betfair Exchange)

Matches: 370; evaluation states: 2737. Skipped: {'no kick-off price': 0, 'after goal': 218, 'missing price': 0}. Kick-off market fit (constant-rate model): RMS probability error 0.0064.

Models: **M1** constant rates (Step 1); **M3** time-varying rates + stoppage (Step 3); **M4** M3 + predictive score effects (Step 4); **MKT** Betfair market. Red cards (Step 5) are not modelled.

Each model's pre-match rates are fitted so that its own kick-off prices match the market.

## 1. Accuracy by minute (lower is better)

| minute   |   n |   RPS MKT |   RPS M1 |   RPS M3 |   RPS M4 |   Brier o2.5 MKT |   Brier o2.5 M1 |   Brier o2.5 M3 |   Brier o2.5 M4 |
|:---------|----:|----------:|---------:|---------:|---------:|-----------------:|----------------:|----------------:|----------------:|
| 0'       | 367 |    0.1983 |   0.198  |   0.198  |   0.1981 |           0.2379 |          0.2378 |          0.2378 |          0.2379 |
| 15'      | 337 |    0.1827 |   0.1867 |   0.1863 |   0.1861 |           0.2058 |          0.2043 |          0.2047 |          0.2044 |
| 30'      | 349 |    0.1664 |   0.1693 |   0.1688 |   0.1687 |           0.1915 |          0.1897 |          0.19   |          0.19   |
| HT       | 320 |    0.1458 |   0.15   |   0.1492 |   0.1489 |           0.1597 |          0.1572 |          0.1586 |          0.159  |
| 60'      | 344 |    0.1149 |   0.1173 |   0.117  |   0.117  |           0.1152 |          0.1144 |          0.1129 |          0.1135 |
| 75'      | 336 |    0.0825 |   0.0849 |   0.0835 |   0.0835 |           0.0782 |          0.0762 |          0.0744 |          0.0749 |
| 85'      | 343 |    0.054  |   0.0625 |   0.0598 |   0.0598 |           0.0575 |          0.0621 |          0.0561 |          0.0566 |
| 90'      | 341 |    0.0346 |   0.0469 |   0.0428 |   0.0428 |           0.0309 |          0.0381 |          0.0319 |          0.0322 |

## 2. Does each added effect help? (paired, bootstrap by match)

**Decision rule** (revised before any real evaluation data was seen; see docs/inplay-plan.md): keep an effect if the RPS difference pooled over the in-play minutes (15'–90') has a 95% CI below 0 **and** no single minute is significantly worse. The original rule (CI below 0 at *most* minutes) rejected the true model in the synthetic check, because time effects matter only late in matches.

### Step 3: time-varying rates + stoppage (M3 − M1)

| minute   |   n |    mean |   ci_low |   ci_high | better?             |
|:---------|----:|--------:|---------:|----------:|:--------------------|
| 0'       | 367 |  0      |  -0      |    0      | no clear difference |
| 15'      | 337 | -0.0004 |  -0.0007 |    0      | no clear difference |
| 30'      | 349 | -0.0005 |  -0.0013 |    0.0004 | no clear difference |
| HT       | 320 | -0.0008 |  -0.0018 |    0.0003 | no clear difference |
| 60'      | 344 | -0.0003 |  -0.0017 |    0.001  | no clear difference |
| 75'      | 336 | -0.0014 |  -0.0033 |    0.0004 | no clear difference |
| 85'      | 343 | -0.0028 |  -0.005  |   -0.0008 | yes (CI < 0)        |
| 90'      | 341 | -0.0041 |  -0.0063 |   -0.0021 | yes (CI < 0)        |

Pooled over in-play minutes (370 matches): -0.00154 (95% CI -0.00259 to -0.00054); minutes significantly worse: 0. → **KEEP**

(Original rule, for reference: CI below 0 at 2 of 7 minutes.)

### Step 4: score effects (M4 − M3)

| minute   |   n |    mean |   ci_low |   ci_high | better?             |
|:---------|----:|--------:|---------:|----------:|:--------------------|
| 0'       | 367 |  0.0001 |  -0.0002 |    0.0004 | no clear difference |
| 15'      | 337 | -0.0002 |  -0.0008 |    0.0003 | no clear difference |
| 30'      | 349 | -0.0001 |  -0.0007 |    0.0006 | no clear difference |
| HT       | 320 | -0.0003 |  -0.001  |    0.0003 | no clear difference |
| 60'      | 344 |  0      |  -0.0005 |    0.0005 | no clear difference |
| 75'      | 336 | -0      |  -0.0004 |    0.0003 | no clear difference |
| 85'      | 343 | -0      |  -0.0002 |    0.0001 | no clear difference |
| 90'      | 341 | -0      |  -0.0001 |    0      | no clear difference |

Pooled over in-play minutes (370 matches): -0.00009 (95% CI -0.00042 to +0.00024); minutes significantly worse: 0. → **DROP (not shown to help)**

(Original rule, for reference: CI below 0 at 0 of 7 minutes.)

## 3. Model versus market

RPS difference model − market (negative = model more accurate than the market).

**M1 − MKT**

| minute   |   n |    mean |   ci_low |   ci_high | better?             |
|:---------|----:|--------:|---------:|----------:|:--------------------|
| 0'       | 367 | -0.0002 |  -0.0006 |    0.0002 | no clear difference |
| 15'      | 337 |  0.0039 |   0.0005 |    0.0077 | worse (CI > 0)      |
| 30'      | 349 |  0.0029 |  -0.0014 |    0.0072 | no clear difference |
| HT       | 320 |  0.0041 |  -0.0004 |    0.009  | no clear difference |
| 60'      | 344 |  0.0024 |  -0.0025 |    0.0076 | no clear difference |
| 75'      | 336 |  0.0024 |  -0.0038 |    0.0091 | no clear difference |
| 85'      | 343 |  0.0085 |   0.0004 |    0.0182 | worse (CI > 0)      |
| 90'      | 341 |  0.0123 |   0.0028 |    0.0236 | worse (CI > 0)      |

**M3 − MKT**

| minute   |   n |    mean |   ci_low |   ci_high | better?             |
|:---------|----:|--------:|---------:|----------:|:--------------------|
| 0'       | 367 | -0.0002 |  -0.0005 |    0.0002 | no clear difference |
| 15'      | 337 |  0.0036 |   0.0004 |    0.0071 | worse (CI > 0)      |
| 30'      | 349 |  0.0024 |  -0.0015 |    0.0062 | no clear difference |
| HT       | 320 |  0.0033 |  -0.0006 |    0.0076 | no clear difference |
| 60'      | 344 |  0.0021 |  -0.002  |    0.0065 | no clear difference |
| 75'      | 336 |  0.001  |  -0.0041 |    0.0065 | no clear difference |
| 85'      | 343 |  0.0058 |  -0.0008 |    0.0141 | no clear difference |
| 90'      | 341 |  0.0082 |  -0.0001 |    0.0179 | no clear difference |

**M4 − MKT**

| minute   |   n |    mean |   ci_low |   ci_high | better?             |
|:---------|----:|--------:|---------:|----------:|:--------------------|
| 0'       | 367 | -0.0001 |  -0.0003 |    0      | no clear difference |
| 15'      | 337 |  0.0034 |   0.0004 |    0.0066 | worse (CI > 0)      |
| 30'      | 349 |  0.0023 |  -0.0013 |    0.0059 | no clear difference |
| HT       | 320 |  0.003  |  -0.0006 |    0.007  | no clear difference |
| 60'      | 344 |  0.0021 |  -0.0018 |    0.0065 | no clear difference |
| 75'      | 336 |  0.001  |  -0.0041 |    0.0065 | no clear difference |
| 85'      | 343 |  0.0058 |  -0.0008 |    0.014  | no clear difference |
| 90'      | 341 |  0.0081 |  -0.0001 |    0.0178 | no clear difference |

Average distance from the market (half the sum of |p_model − p_market| over H/D/A; 0.05 = 5 points of probability moved):

| minute   |     M1 |     M3 |     M4 |
|:---------|-------:|-------:|-------:|
| 0'       | 0.0151 | 0.0151 | 0.0054 |
| 15'      | 0.0343 | 0.0343 | 0.0291 |
| 30'      | 0.0474 | 0.0434 | 0.0386 |
| HT       | 0.0532 | 0.0462 | 0.0402 |
| 60'      | 0.07   | 0.0516 | 0.0495 |
| 75'      | 0.0871 | 0.0533 | 0.0523 |
| 85'      | 0.106  | 0.0622 | 0.062  |
| 90'      | 0.1187 | 0.0704 | 0.0702 |

## Caveats

- The market is last traded prices (not mid-prices), normalised; they can be stale for up to a minute and right after goals (those 3 minutes are skipped).
- Over 2.5: once three goals are scored the market is settled and closed, so its probability is set to 1 (its last traded price is stale).
- Pre-match rates come from the kick-off market, so minute 0 matches it by construction; differences are purely about in-play updating.
- Goal times are rebuilt from market suspensions (±1 minute); red cards are not observed or modelled.
- Time shape and score effects were fitted on 2015/16 StatsBomb data; this evaluation fits nothing on the evaluation season.
