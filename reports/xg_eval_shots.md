# shot-based xG proxy ratings vs goals Dixon-Coles vs the market (big-5 leagues)

Matches: development 5475, validation 7203

## 1. Choose alpha on development seasons (RPS)

- goals DC: 0.19675
- xG alpha=0.25: 0.19579
- xG alpha=0.5: 0.19576
- xG alpha=0.75: 0.19674
- xG alpha=1: 0.19884
- chosen: **xG alpha=0.5**

## 2. Validation seasons

| model | RPS | log loss | distance to market (pts) |
|---|---|---|---|
| market (Pinnacle close) | 0.19768 | 0.9753 | 0 |
| goals DC | 0.20509 | 1.0094 | 7.53 |
| xG alpha=0.25 | 0.20393 | 0.9968 | 7.03 |
| xG alpha=0.5 | 0.20355 | 0.9948 | 6.99 |
| xG alpha=0.75 | 0.20403 | 0.9961 | 7.48 |
| xG alpha=1 | 0.20546 | 1.0007 | 8.41 |

- RPS xG alpha=0.5 minus goals DC: -0.00154 (95% CI -0.00229 to -0.00084)
- RPS xG alpha=0.5 minus market: +0.00587 (95% CI +0.00449 to +0.00720)
- RPS goals DC minus market: +0.00742 (95% CI +0.00587 to +0.00904)
- distance to market: goals DC 7.53 pts -> xG alpha=0.5 6.99 pts (7% closer)

## 3. Does the xG model add anything to the market? (log-linear blend)

- goals DC: weights fitted on dev market 1.053 / model -0.009; on validation 1.016 / -0.031. Blend (dev weights) minus market RPS on validation +0.00004 (CI -0.00007 to +0.00014)
- xG alpha=0.5: weights fitted on dev market 1.105 / model -0.077; on validation 1.032 / -0.054. Blend (dev weights) minus market RPS on validation +0.00002 (CI -0.00011 to +0.00016)

## 4. By league (validation): RPS and distance to market

| league | market | goals DC | xG alpha=0.5 | dist DC | dist xG |
|---|---|---|---|---|---|
| D1 | 0.20210 | 0.20956 | 0.20774 | 6.95 | 6.69 |
| E0 | 0.19903 | 0.20697 | 0.20481 | 7.94 | 7.21 |
| F1 | 0.20137 | 0.20850 | 0.20791 | 7.64 | 7.43 |
| I1 | 0.19226 | 0.19903 | 0.19745 | 7.18 | 6.55 |
| SP1 | 0.19472 | 0.20249 | 0.20094 | 7.84 | 7.05 |
