# Baseline (real data)

Leagues: ['D1', 'E0', 'E1', 'F1', 'I1', 'SP1']; matches before 2023-07-01: 42703

## 1. Margin removal (Pinnacle closing odds)

|                |    rps |   log_loss |   brier |     n |    ece |
|:---------------|-------:|-----------:|--------:|------:|-------:|
| multiplicative | 0.1989 |     0.9812 |  0.5844 | 26051 | 0.0043 |
| power          | 0.1989 |     0.9812 |  0.5844 | 26051 | 0.0024 |
| shin           | 0.1989 |     0.9812 |  0.5844 | 26051 | 0.0025 |

Mean closing overround: 0.0236

## 2. Ratings vs market (validation seasons)

|              |    rps |   log_loss |   brier |    n |
|:-------------|-------:|-----------:|--------:|-----:|
| elo          | 0.2107 |     1.0175 |  0.6091 | 9411 |
| pi           | 0.2092 |     1.0133 |  0.6058 | 9411 |
| market_open  | 0.2032 |     0.9941 |  0.5927 | 9401 |
| market_close | 0.2025 |     0.9919 |  0.5913 | 9409 |

## 3. Walk-forward Dixon-Coles and market blend (second half of validation)

|             |    rps |   log_loss |   brier |    n |
|:------------|-------:|-----------:|--------:|-----:|
| dixon_coles | 0.2097 |     1.0249 |  0.6069 | 4728 |
| market_open | 0.2028 |     0.9921 |  0.5918 | 4728 |
| blend       | 0.2028 |     0.9921 |  0.5918 | 4728 |

Blend weights: market 0.974, model -0.006

Blend minus opening RPS: 0.00002 (95% CI -0.00008 to 0.00012)

### Value betting at opening odds

|                     | value             |
|:--------------------|:------------------|
| n_bets              | 3.0               |
| roi_flat            | 0.4267            |
| roi_ci              | (-1.0, 3.28)      |
| mean_clv            | -0.0457           |
| clv_ci              | (-0.1381, 0.0323) |
| share_beating_close | 0.3333            |

By league:

| league   |   n |   mean_clv |   roi |
|:---------|----:|-----------:|------:|
| E0       |   2 |     0.0004 |  1.14 |
| F1       |   1 |    -0.1381 | -1    |
