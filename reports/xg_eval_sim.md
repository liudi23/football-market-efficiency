# xG model on SIMULATED data (synthetic xG = 0.11 x shots)

Matches: development 1140, validation 1520

## 1. Choose alpha on development seasons (RPS)

- goals DC: 0.22527
- xG alpha=0.25: 0.22424
- xG alpha=0.5: 0.22377
- xG alpha=0.75: 0.22382
- xG alpha=1: 0.22441
- chosen: **xG alpha=0.5**

## 2. Validation seasons

| model | RPS | log loss | distance to market (pts) |
|---|---|---|---|
| market (Pinnacle close) | 0.20717 | 1.0100 | 0 |
| goals DC | 0.21704 | 1.0412 | 7.86 |
| xG alpha=0.25 | 0.21572 | 1.0368 | 7.17 |
| xG alpha=0.5 | 0.21482 | 1.0340 | 6.71 |
| xG alpha=0.75 | 0.21440 | 1.0326 | 6.53 |
| xG alpha=1 | 0.21446 | 1.0329 | 6.68 |

- RPS xG alpha=0.5 minus goals DC: -0.00222 (95% CI -0.00353 to -0.00095)
- RPS xG alpha=0.5 minus market: +0.00765 (95% CI +0.00469 to +0.01064)
- RPS goals DC minus market: +0.00987 (95% CI +0.00620 to +0.01358)
- distance to market: goals DC 7.86 pts -> xG alpha=0.5 6.71 pts (15% closer)

## 3. Does the xG model add anything to the market? (log-linear blend)

- goals DC: weights fitted on dev market 0.874 / model -0.023; on validation 1.250 / -0.209. Blend (dev weights) minus market RPS on validation +0.00089 (CI +0.00015 to +0.00161)
- xG alpha=0.5: weights fitted on dev market 0.925 / model -0.089; on validation 1.283 / -0.256. Blend (dev weights) minus market RPS on validation +0.00082 (CI +0.00004 to +0.00158)

## 4. By league (validation): RPS and distance to market

| league | market | goals DC | xG alpha=0.5 | dist DC | dist xG |
|---|---|---|---|---|---|
| SIM | 0.20717 | 0.21704 | 0.21482 | 7.86 | 6.71 |
