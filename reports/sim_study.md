# Stage 0: simulation study

Worlds: {'null': {'market_lag': 0, 'open_noise': 0.03}, 'slow': {'market_lag': 4, 'open_noise': 0.05}, 'very_slow': {'market_lag': 10, 'open_noise': 0.08}}. Walk-forward Dixon-Coles from 2017-08-01; blend fitted on the first half of out-of-sample matches, everything below scored on the second half.

## World: null

|    |   rps_truth |   rps_close |   rps_open |   rps_dixon_coles |   rps_blend |   blend_minus_open |   ci_low |   ci_high |   w_market |   w_model |   n_bets |      roi |   mean_clv |   clv_ci_low |
|---:|------------:|------------:|-----------:|------------------:|------------:|-------------------:|---------:|----------:|-----------:|----------:|---------:|---------:|-----------:|-------------:|
|  0 |      0.2146 |      0.2148 |     0.2149 |            0.2202 |      0.2151 |             0.0002 |  -0      |    0.0005 |     1.0349 |   -0.0399 |        0 | nan      |   nan      |     nan      |
|  1 |      0.2082 |      0.2083 |     0.2082 |            0.2178 |      0.2082 |            -0      |  -0.0007 |    0.0006 |     1.1318 |   -0.0761 |       81 |  -0.0216 |    -0.0225 |      -0.0279 |
|  2 |      0.2069 |      0.2068 |     0.2067 |            0.2166 |      0.2065 |            -0.0002 |  -0.0009 |    0.0006 |     1.1806 |   -0.1192 |      221 |  -0.0233 |    -0.0187 |      -0.0223 |
|  3 |      0.2128 |      0.213  |     0.2128 |            0.2226 |      0.2129 |             0.0002 |  -0.0005 |    0.0008 |     0.9037 |   -0.003  |       90 |  -0.041  |    -0.0426 |      -0.0533 |
|  4 |      0.2145 |      0.2147 |     0.2146 |            0.2235 |      0.2145 |            -0.0001 |  -0.0004 |    0.0003 |     0.9497 |    0.0028 |        0 | nan      |   nan      |     nan      |

- Edge detected by forecast score (blend beats opening RPS, 95% CI below 0): **0%** of 5 seeds
- Edge detected by closing line value (CLV 95% CI above 0): **0%**
- Seeds with positive flat ROI (not a test, shown for contrast): 0%

## World: slow

|    |   rps_truth |   rps_close |   rps_open |   rps_dixon_coles |   rps_blend |   blend_minus_open |   ci_low |   ci_high |   w_market |   w_model |   n_bets |      roi |   mean_clv |   clv_ci_low |
|---:|------------:|------------:|-----------:|------------------:|------------:|-------------------:|---------:|----------:|-----------:|----------:|---------:|---------:|-----------:|-------------:|
|  0 |      0.2146 |      0.2148 |     0.2156 |            0.2202 |      0.2153 |            -0.0002 |  -0.0007 |    0.0002 |     0.9595 |   -0.017  |        0 | nan      |   nan      |     nan      |
|  1 |      0.2082 |      0.2083 |     0.2093 |            0.2178 |      0.2093 |             0      |  -0.0005 |    0.0005 |     1.093  |   -0.0559 |        8 |  -0.1681 |    -0.0769 |      -0.1071 |
|  2 |      0.2069 |      0.2068 |     0.2066 |            0.2166 |      0.2065 |            -0.0002 |  -0.0005 |    0.0001 |     1.0786 |   -0.0516 |        0 | nan      |   nan      |     nan      |
|  3 |      0.2128 |      0.213  |     0.2147 |            0.2226 |      0.2148 |             0.0002 |  -0.0007 |    0.001  |     0.8493 |    0.0325 |      244 |   0.0812 |     0.0133 |      -0.0038 |
|  4 |      0.2145 |      0.2147 |     0.2175 |            0.2235 |      0.2171 |            -0.0003 |  -0.0008 |    0.0002 |     0.9418 |   -0.0205 |        0 | nan      |   nan      |     nan      |

- Edge detected by forecast score (blend beats opening RPS, 95% CI below 0): **0%** of 5 seeds
- Edge detected by closing line value (CLV 95% CI above 0): **0%**
- Seeds with positive flat ROI (not a test, shown for contrast): 20%

## World: very_slow

|    |   rps_truth |   rps_close |   rps_open |   rps_dixon_coles |   rps_blend |   blend_minus_open |   ci_low |   ci_high |   w_market |   w_model |   n_bets |     roi |   mean_clv |   clv_ci_low |
|---:|------------:|------------:|-----------:|------------------:|------------:|-------------------:|---------:|----------:|-----------:|----------:|---------:|--------:|-----------:|-------------:|
|  0 |      0.2146 |      0.2148 |     0.216  |            0.2202 |      0.215  |            -0.001  |  -0.0027 |    0.0007 |     0.7059 |    0.1783 |      595 |  0.1286 |     0.0395 |       0.0232 |
|  1 |      0.2082 |      0.2083 |     0.2116 |            0.2178 |      0.2116 |             0      |  -0.0013 |    0.0014 |     0.7713 |    0.2214 |      303 | -0.0446 |     0.0528 |       0.0318 |
|  2 |      0.2069 |      0.2068 |     0.2091 |            0.2166 |      0.2097 |             0.0006 |  -0.0004 |    0.0014 |     0.7395 |    0.2375 |      328 | -0.0086 |     0.0156 |      -0.0027 |
|  3 |      0.2128 |      0.213  |     0.2155 |            0.2226 |      0.2163 |             0.0007 |  -0.0007 |    0.0022 |     0.6847 |    0.1701 |      593 |  0.0049 |     0.0589 |       0.0423 |
|  4 |      0.2145 |      0.2147 |     0.2205 |            0.2235 |      0.2195 |            -0.001  |  -0.0021 |    0      |     0.7977 |    0.0714 |      471 |  0.1279 |     0.0766 |       0.0576 |

- Edge detected by forecast score (blend beats opening RPS, 95% CI below 0): **0%** of 5 seeds
- Edge detected by closing line value (CLV 95% CI above 0): **80%**
- Seeds with positive flat ROI (not a test, shown for contrast): 60%

Reading: in the null world a detection rate is the false-positive rate; in the other worlds it is the power. Ranked probability score: lower is better. Seeds with 0 bets mean the blend gave the model too little weight to clear the EV threshold, which is the correct behaviour when there is no edge.