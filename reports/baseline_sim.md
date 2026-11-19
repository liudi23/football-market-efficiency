# Baseline (sim data)

Leagues: ['SIM']; matches before 2023-07-01: 3420

## 1. Margin removal (Pinnacle closing odds)

|                |    rps |   log_loss |   brier |    n |    ece |
|:---------------|-------:|-----------:|--------:|-----:|-------:|
| multiplicative | 0.2084 |     1.0164 |  0.6095 | 3420 | 0.0103 |
| power          | 0.2084 |     1.0165 |  0.6096 | 3420 | 0.0144 |
| shin           | 0.2084 |     1.0164 |  0.6096 | 3420 | 0.0137 |

Mean closing overround: 0.0250

## 2. Ratings vs market (validation seasons)

|              |    rps |   log_loss |   brier |    n |
|:-------------|-------:|-----------:|--------:|-----:|
| elo          | 0.219  |     1.0461 |  0.6298 | 1520 |
| pi           | 0.2204 |     1.0505 |  0.6327 | 1520 |
| market_open  | 0.2125 |     1.024  |  0.6153 | 1520 |
| market_close | 0.2101 |     1.0167 |  0.6096 | 1520 |

## 3. Walk-forward Dixon-Coles and market blend (second half of validation)

|             |    rps |   log_loss |   brier |   n |
|:------------|-------:|-----------:|--------:|----:|
| dixon_coles | 0.2254 |     1.0688 |  0.6455 | 760 |
| market_open | 0.2203 |     1.0513 |  0.6345 | 760 |
| blend       | 0.2215 |     1.0553 |  0.6372 | 760 |

Blend weights: market 1.270, model -0.264

Blend minus opening RPS: 0.00118 (95% CI -0.00023 to 0.00254)

### Value betting at opening odds

|                     | value              |
|:--------------------|:-------------------|
| n_bets              | 370.0              |
| roi_flat            | -0.0849            |
| roi_ci              | (-0.2128, 0.0553)  |
| mean_clv            | -0.0591            |
| clv_ci              | (-0.0726, -0.0462) |
| share_beating_close | 0.3324             |

By league:

| league   |   n |   mean_clv |     roi |
|:---------|----:|-----------:|--------:|
| SIM      | 370 |    -0.0591 | -0.0849 |
