# In-play Step 6: evaluation by minute — synthetic season (pipeline check, seed 1)

Matches: 380; evaluation states: 2820. Skipped: {'no kick-off price': 0, 'after goal': 220, 'missing price': 0}. Kick-off market fit (constant-rate model): RMS probability error 0.0035.

Models: **M1** constant rates (Step 1); **M3** time-varying rates + stoppage (Step 3); **M4** M3 + predictive score effects (Step 4); **MKT** Betfair market. Red cards (Step 5) are not modelled.

Each model's pre-match rates are fitted so that its own kick-off prices match the market.

## 1. Accuracy by minute (lower is better)

| minute   |   n |   RPS MKT |   RPS M1 |   RPS M3 |   RPS M4 |   Brier o2.5 MKT |   Brier o2.5 M1 |   Brier o2.5 M3 |   Brier o2.5 M4 |
|:---------|----:|----------:|---------:|---------:|---------:|-----------------:|----------------:|----------------:|----------------:|
| 0'       | 380 |    0.2099 |   0.2099 |   0.2099 |   0.2097 |           0.2303 |          0.2303 |          0.2303 |          0.2302 |
| 15'      | 358 |    0.2033 |   0.2035 |   0.2035 |   0.2032 |           0.2184 |          0.222  |          0.2195 |          0.2196 |
| 30'      | 354 |    0.1804 |   0.1799 |   0.1796 |   0.1791 |           0.1898 |          0.1925 |          0.1906 |          0.1908 |
| HT       | 331 |    0.1544 |   0.1548 |   0.1545 |   0.1542 |           0.1713 |          0.1755 |          0.1712 |          0.1709 |
| 60'      | 351 |    0.1223 |   0.123  |   0.1223 |   0.1221 |           0.1399 |          0.1448 |          0.1399 |          0.1397 |
| 75'      | 353 |    0.0842 |   0.0851 |   0.0843 |   0.0844 |           0.0951 |          0.101  |          0.095  |          0.0951 |
| 85'      | 343 |    0.0546 |   0.0564 |   0.0546 |   0.0546 |           0.0604 |          0.0645 |          0.0603 |          0.0607 |
| 90'      | 350 |    0.0281 |   0.03   |   0.0282 |   0.0282 |           0.0371 |          0.0429 |          0.0371 |          0.0372 |

## 2. Does each added effect help? (paired, bootstrap by match)

**Decision rule** (revised before any real evaluation data was seen; see docs/inplay-plan.md): keep an effect if the RPS difference pooled over the in-play minutes (15'–90') has a 95% CI below 0 **and** no single minute is significantly worse. The original rule (CI below 0 at *most* minutes) rejected the true model in the synthetic check, because time effects matter only late in matches.

### Step 3: time-varying rates + stoppage (M3 − M1)

| minute   |   n |    mean |   ci_low |   ci_high | better?             |
|:---------|----:|--------:|---------:|----------:|:--------------------|
| 0'       | 380 | -0      |  -0      |    0      | no clear difference |
| 15'      | 358 |  0      |  -0.0003 |    0.0003 | no clear difference |
| 30'      | 354 | -0.0002 |  -0.001  |    0.0005 | no clear difference |
| HT       | 331 | -0.0003 |  -0.0011 |    0.0006 | no clear difference |
| 60'      | 351 | -0.0006 |  -0.0019 |    0.0006 | no clear difference |
| 75'      | 353 | -0.0008 |  -0.0025 |    0.0009 | no clear difference |
| 85'      | 343 | -0.0018 |  -0.0038 |    0      | no clear difference |
| 90'      | 350 | -0.0018 |  -0.0034 |   -0.0004 | yes (CI < 0)        |

Pooled over in-play minutes (380 matches): -0.00095 (95% CI -0.00183 to -0.00010); minutes significantly worse: 0. → **KEEP**

(Original rule, for reference: CI below 0 at 1 of 7 minutes.)

### Step 4: score effects (M4 − M3)

| minute   |   n |    mean |   ci_low |   ci_high | better?             |
|:---------|----:|--------:|---------:|----------:|:--------------------|
| 0'       | 380 | -0.0002 |  -0.0006 |    0.0002 | no clear difference |
| 15'      | 358 | -0.0003 |  -0.0009 |    0.0003 | no clear difference |
| 30'      | 354 | -0.0005 |  -0.0012 |    0.0001 | no clear difference |
| HT       | 331 | -0.0003 |  -0.001  |    0.0003 | no clear difference |
| 60'      | 351 | -0.0002 |  -0.0007 |    0.0002 | no clear difference |
| 75'      | 353 |  0.0001 |  -0.0002 |    0.0004 | no clear difference |
| 85'      | 343 |  0      |  -0.0001 |    0.0002 | no clear difference |
| 90'      | 350 |  0      |  -0      |    0.0001 | no clear difference |

Pooled over in-play minutes (380 matches): -0.00019 (95% CI -0.00053 to +0.00015); minutes significantly worse: 0. → **DROP (not shown to help)**

(Original rule, for reference: CI below 0 at 0 of 7 minutes.)

## 3. Model versus market

RPS difference model − market (negative = model more accurate than the market).

**M1 − MKT**

| minute   |   n |    mean |   ci_low |   ci_high | better?             |
|:---------|----:|--------:|---------:|----------:|:--------------------|
| 0'       | 380 | -0      |  -0.0003 |    0.0002 | no clear difference |
| 15'      | 358 |  0.0003 |  -0.001  |    0.0017 | no clear difference |
| 30'      | 354 | -0.0005 |  -0.0017 |    0.0008 | no clear difference |
| HT       | 331 |  0.0004 |  -0.0009 |    0.0017 | no clear difference |
| 60'      | 351 |  0.0007 |  -0.0007 |    0.0022 | no clear difference |
| 75'      | 353 |  0.0009 |  -0.0009 |    0.0028 | no clear difference |
| 85'      | 343 |  0.0017 |  -0.0002 |    0.0038 | no clear difference |
| 90'      | 350 |  0.0019 |   0.0005 |    0.0035 | worse (CI > 0)      |

**M3 − MKT**

| minute   |   n |    mean |   ci_low |   ci_high | better?             |
|:---------|----:|--------:|---------:|----------:|:--------------------|
| 0'       | 380 | -0      |  -0.0003 |    0.0002 | no clear difference |
| 15'      | 358 |  0.0003 |  -0.001  |    0.0017 | no clear difference |
| 30'      | 354 | -0.0007 |  -0.0019 |    0.0004 | no clear difference |
| HT       | 331 |  0.0002 |  -0.0009 |    0.0013 | no clear difference |
| 60'      | 351 |  0.0001 |  -0.0007 |    0.0009 | no clear difference |
| 75'      | 353 |  0.0001 |  -0.0005 |    0.0007 | no clear difference |
| 85'      | 343 | -0.0001 |  -0.0004 |    0.0002 | no clear difference |
| 90'      | 350 |  0.0001 |  -0      |    0.0002 | no clear difference |

**M4 − MKT**

| minute   |   n |    mean |   ci_low |   ci_high | better?             |
|:---------|----:|--------:|---------:|----------:|:--------------------|
| 0'       | 380 | -0.0002 |  -0.0007 |    0.0003 | no clear difference |
| 15'      | 358 | -0      |  -0.0015 |    0.0016 | no clear difference |
| 30'      | 354 | -0.0013 |  -0.0027 |    0.0001 | no clear difference |
| HT       | 331 | -0.0002 |  -0.0014 |    0.0011 | no clear difference |
| 60'      | 351 | -0.0001 |  -0.0011 |    0.0008 | no clear difference |
| 75'      | 353 |  0.0002 |  -0.0004 |    0.0009 | no clear difference |
| 85'      | 343 | -0      |  -0.0004 |    0.0003 | no clear difference |
| 90'      | 350 |  0.0001 |  -0      |    0.0002 | no clear difference |

Average distance from the market (half the sum of |p_model − p_market| over H/D/A; 0.05 = 5 points of probability moved):

| minute   |     M1 |     M3 |     M4 |
|:---------|-------:|-------:|-------:|
| 0'       | 0.0084 | 0.0084 | 0.0184 |
| 15'      | 0.0197 | 0.0188 | 0.0253 |
| 30'      | 0.0199 | 0.0163 | 0.0233 |
| HT       | 0.0227 | 0.0154 | 0.02   |
| 60'      | 0.0291 | 0.0126 | 0.0166 |
| 75'      | 0.0436 | 0.0099 | 0.0127 |
| 85'      | 0.0522 | 0.0054 | 0.0061 |
| 90'      | 0.0561 | 0.0028 | 0.0032 |

## Caveats

- The market is last traded prices (not mid-prices), normalised; they can be stale for up to a minute and right after goals (those 3 minutes are skipped).
- Pre-match rates come from the kick-off market, so minute 0 matches it by construction; differences are purely about in-play updating.
- Goal times are rebuilt from market suspensions (±1 minute); red cards are not observed or modelled.
- Time shape and score effects were fitted on 2015/16 StatsBomb data; this evaluation fits nothing on the evaluation season.
