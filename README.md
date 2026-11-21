# sports-trading

Market-anchored football forecasting and betting research. The project treats the betting
market as the null hypothesis, checks on simulated markets that the pipeline finds known
edges (and only those), and then tests on real data with time-ordered splits and a locked
holdout. Full method and references: [docs/methodology.md](docs/methodology.md).

## Findings so far

**The market is the best forecaster, and no model here beats it.** Our models summarise
public information well; the closing price already contains that information and more.

**Best pre-match model:** Dixon–Coles with time decay, with team ratings fitted to an equal
mix of goals and a shot-based xG proxy (0.31 × shots on target), refitted weekly
(`src/sportsq/xg.py`). Big-5 leagues, validation 2019/20–2022/23, 7,203 matches:

| Forecast | RPS (lower is better) | Gap to market | Distance from market (pts) |
|---|---|---|---|
| **Market (Pinnacle close, margin removed)** | **0.1977** | n/a | 0 |
| Goals + shot-proxy Dixon–Coles | 0.2036 | +0.0059 (CI +0.0045 to +0.0072) | 7.0 |
| Goals-only Dixon–Coles | 0.2051 | +0.0074 | 7.5 |

- Shot data improves the goals model in every league (−0.0015 RPS, CI −0.0023 to −0.0008)
  and closes about 21% of the gap to the market.
- In a log-linear blend the market gets all the weight (model weight ≈ 0).
- A proxy calibrated to real StatsBomb xG tracks xG better (correlation 0.73 vs 0.68) but
  forecasts no better: the goal-heavy proxy keeps finishing information that pure xG drops.
- Real Understat xG is the next test (downloader ready; the site was down).

**In play:** time-varying goal rates with realistic stoppage time (Step 3) beat a constant-rate
model; score effects add nothing. The live Betfair market beats every model at every minute.

**Market efficiency and alternative data** (all against the sharp closing price):

| Test | Result |
|---|---|
| Consensus of other bookmakers vs Pinnacle | Some books (BetVictor, Interwetten) move first, but the signal is too small to beat the margin |
| Weather (wind, rain, temperature) vs over/under 2.5 | No effect; rules out more than ~0.07 log-odds per 10 km/h of wind |
| Crowd-free football 2020/21 | The market cut home advantage almost immediately |
| Popular clubs | Soft bookmakers overcharge big-club backers by ~1 pt; Pinnacle prices them fairly |
| In-play match odds vs over/under 2.5 | Match odds carry information the over/under price lacks (weight 0.34 ± 0.15); trade not significant; 2021/22 confirmation pre-registered |

**Where an edge could exist:** speed (line-ups, live events), market structure (consistency
between related markets, order books) and soft bookmakers (who restrict winning accounts),
not better use of slow public information. Details: [docs/methodology.md](docs/methodology.md).

## Setup

Keep the virtual environment outside Dropbox so it doesn't sync:

```bash
cd ~/Library/CloudStorage/Dropbox/DS_projects/sports-trading
python3 -m venv ~/.venvs/sports-trading && source ~/.venvs/sports-trading/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Quick start

```bash
python scripts/sim_study.py --seeds 5          # Stage 0: injection-recovery on synthetic markets
python scripts/download_data.py                # football-data.co.uk CSVs -> data/raw/
python scripts/baseline_real.py                # Stages 1-3 on real data -> reports/baseline_real.md
python scripts/baseline_real.py --sim          # same pipeline on simulated data
python scripts/inplay_mc_check.py              # in-play: Monte Carlo vs analytic -> reports/inplay_mc_check.md
python scripts/download_statsbomb.py           # goal/red-card minutes (StatsBomb open data) -> data/processed/
python scripts/inplay_intensity.py             # fit time-varying goal rates -> reports/inplay_intensity.md
python scripts/inplay_score_effects.py         # fit score effects -> reports/inplay_score_effects.md
python scripts/betfair_index.py --budget 150   # index Betfair BASIC files in data/BASIC (resumable)
python scripts/betfair_epl.py --season 2425    # Premier League in-play market prices + rebuilt timelines
python scripts/inplay_eval.py --synthetic      # Step 6 pipeline check on a simulated season
python scripts/inplay_eval.py --season 2223    # Step 6 evaluation (2023/24+ is locked holdout)

# pre-match xG ratings (big-5 leagues; see "Findings so far")
python scripts/xg_proxy_calibrate.py           # calibrate a shot-count proxy to StatsBomb 2015/16 xG
python scripts/xg_eval.py fit E0 shots         # walk-forward per league: shots | shots2 | understat
python scripts/xg_eval.py report shots         # -> reports/xg_eval_shots.md
python scripts/fetch_understat.py              # real xG (run on your own machine; Understat was down 21 Nov 2026)

# market-efficiency and alternative-data tests (all on seasons before the holdout)
python scripts/consensus_leaders.py            # do other bookmakers lead Pinnacle? -> reports/consensus_leaders.txt
python scripts/fetch_weather.py                # Open-Meteo weather per ground (run on your own machine)
python scripts/weather_ou.py                   # weather vs over/under 2.5 -> reports/weather_ou.txt
python scripts/ghost_games.py                  # crowd-free 2020/21 -> reports/ghost_games.txt
python scripts/popular_teams.py                # big-club pricing bias -> reports/popular_teams.txt
python scripts/inplay_consistency.py fit 2223  # match odds vs over/under in play
python scripts/inplay_consistency.py report 2223
```

## Layout

```
src/sportsq/
  data.py          download + standardise football-data.co.uk (one schema for real and simulated data)
  implied.py       margin removal: multiplicative, power, Shin
  metrics.py       RPS, log loss, Brier, calibration/ECE, paired block bootstrap
  ratings.py       Elo, pi-ratings (pre-match, no lookahead), ordered logit -> 1X2
  dixon_coles.py   Dixon-Coles with time decay; 1X2 and over/under from the score grid
  simulate.py      synthetic leagues + markets with injectable inefficiencies
  backtest.py      walk-forward refits, market blend
  staking.py       value-bet selection, shrunk Kelly, bankroll, closing line value
  betfair.py       Betfair historical stream files: parse markets, rebuild match timelines
  inplay.py        in-play 1X2 and over/under prices from score, minute and stoppage time
                   (analytic + Monte Carlo; time-varying goal rates; score effects)
  xg.py            team ratings on a goals/xG mix (Poisson quasi-likelihood), walk-forward,
                   automatic team-name matching between data sources
scripts/           simulation, data download, pre-match and in-play evaluation, xG,
                   market-efficiency tests (see Quick start)
tests/             unit tests (no lookahead, parameter recovery, scoring rules)
reports/           generated results
docs/              methodology, in-play plan
```

## Simulation check (5 seeds per world)

| World | Injected edge | Detected by RPS | Detected by CLV | Seeds with ROI > 0 |
|---|---|---|---|---|
| null | none | 0% | 0% | 0% |
| slow | opening prices 4 weeks stale | 0% | 0% | 20% |
| very_slow | 10 weeks stale, noisier opening | 0% | **80%** | 60% |

- No false positives in the null world.
- Closing line value detects a real edge long before the forecast score or profit does.
- ROI is too noisy to judge on its own.
- A results-only Dixon–Coles model can't see a small edge.

All of this matches the methodology. See `reports/sim_study.md`.

## Roadmap

- [x] Stage 0: simulation framework and study
- [x] Stages 1–3 baseline: margin removal, Elo/pi-ratings, walk-forward Dixon–Coles, blend, CLV
- [x] Run on real data: market beats every model (see Findings so far)
- [ ] Dynamic bivariate Poisson (state-space / score-driven, Koopman & Lit)
- [x] Shot-based xG proxy ratings (calibrated to StatsBomb xG): close ~21% of the gap to the market
- [ ] Understat xG features (downloader ready; site down on 21 Nov 2026)
- [x] Market-efficiency tests: bookmaker consensus, weather, crowd-free games, popular clubs
- [ ] In-play match odds vs over/under: confirm on 2021/22 (pre-registered; needs the download)
- [ ] Gradient boosting on rating features; decorrelation loss
- [ ] Favourite–longshot bias and opening-vs-closing efficiency analysis (partly done in the consensus test)
- [ ] C# (.NET) port of the score-grid pricer and backtest, checked against Python
- [ ] Locked-holdout evaluation and write-up
- [ ] In-play pricing ([plan](docs/inplay-plan.md)):
  - [x] Step 1: analytic pricer from score and minute (exact in-play Dixon–Coles correction)
  - [x] Step 2: Monte Carlo match-state simulator, checked against Step 1 ([report](reports/inplay_mc_check.md))
  - [x] Step 3: time-varying goal rates and stoppage time ([report](reports/inplay_intensity.md))
  - [x] Step 4: score effects ([report](reports/inplay_score_effects.md))
  - [x] Betfair in-play market data for 2024/25 (Aug–Jan), with timelines checked against results
  - [x] Step 6: evaluation on 2022/23 vs results and the Betfair market
    ([report](reports/inplay_eval_2223.md)): keep time-varying rates, drop score effects
  - [ ] Final one-off test on 2024/25 (locked holdout)
  - [ ] Step 5: red cards (needs a red-card data source)
