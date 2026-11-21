# shot-based xG proxy (calibrated to StatsBomb xG) ratings vs goals Dixon-Coles vs the market (big-5 leagues)

Matches: development 5475, validation 7203

## 1. Choose alpha on development seasons (RPS)

- goals DC: 0.19675
- xG alpha=0.25: 0.19571
- xG alpha=0.5: 0.19590
- xG alpha=0.75: 0.19756
- xG alpha=1: 0.20101
- chosen: **xG alpha=0.25**

## 2. Validation seasons

| model | RPS | log loss | distance to market (pts) |
|---|---|---|---|
| market (Pinnacle close) | 0.19768 | 0.9753 | 0 |
| goals DC | 0.20509 | 1.0094 | 7.53 |
| xG alpha=0.25 | 0.20388 | 0.9961 | 7.06 |
| xG alpha=0.5 | 0.20367 | 0.9949 | 7.22 |
| xG alpha=0.75 | 0.20466 | 0.9980 | 8.08 |
| xG alpha=1 | 0.20710 | 1.0059 | 9.55 |

- RPS xG alpha=0.25 minus goals DC: -0.00121 (95% CI -0.00162 to -0.00082)
- RPS xG alpha=0.25 minus market: +0.00621 (95% CI +0.00479 to +0.00756)
- RPS goals DC minus market: +0.00742 (95% CI +0.00587 to +0.00904)
- distance to market: goals DC 7.53 pts -> xG alpha=0.25 7.06 pts (6% closer)

## 3. Does the xG model add anything to the market? (log-linear blend)

- goals DC: weights fitted on dev market 1.053 / model -0.009; on validation 1.016 / -0.031. Blend (dev weights) minus market RPS on validation +0.00004 (CI -0.00007 to +0.00014)
- xG alpha=0.25: weights fitted on dev market 1.105 / model -0.072; on validation 1.034 / -0.054. Blend (dev weights) minus market RPS on validation +0.00002 (CI -0.00011 to +0.00016)

## 4. By league (validation): RPS and distance to market

| league | market | goals DC | xG alpha=0.25 | dist DC | dist xG |
|---|---|---|---|---|---|
| D1 | 0.20210 | 0.20956 | 0.20795 | 6.95 | 6.65 |
| E0 | 0.19903 | 0.20697 | 0.20532 | 7.94 | 7.26 |
| F1 | 0.20137 | 0.20850 | 0.20759 | 7.64 | 7.24 |
| I1 | 0.19226 | 0.19903 | 0.19820 | 7.18 | 6.78 |
| SP1 | 0.19472 | 0.20249 | 0.20139 | 7.84 | 7.30 |

## 5. Calibrated proxy (alpha 0.25) minus first proxy (alpha 0.5), validation RPS: +0.00033 (CI -0.00002 to +0.00070); n=7203
