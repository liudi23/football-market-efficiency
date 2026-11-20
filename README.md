# sports-trading

Market-anchored football forecasting and betting research. The project treats the betting
market as the null hypothesis, checks on simulated markets that the pipeline finds known
edges (and only those), and then tests on real data with time-ordered splits and a locked
holdout. Full method and references: [docs/methodology.md](docs/methodology.md).

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
scripts/           sim_study.py, download_data.py, baseline_real.py, inplay_mc_check.py,
                   download_statsbomb.py, inplay_intensity.py, inplay_score_effects.py,
                   betfair_index.py, betfair_epl.py
tests/             unit tests (no lookahead, parameter recovery, scoring rules)
reports/           generated results
docs/              methodology, in-play plan
```

## First results (simulation, 5 seeds per world)

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
- [ ] Run on real data (download needs your own network)
- [ ] Dynamic bivariate Poisson (state-space / score-driven, Koopman & Lit)
- [ ] Shot-based GAP ratings and Understat xG features; over/under 2.5 market
- [ ] Gradient boosting on rating features; decorrelation loss
- [ ] Favourite–longshot bias and opening-vs-closing efficiency analysis
- [ ] C# (.NET) port of the score-grid pricer and backtest, checked against Python
- [ ] Locked-holdout evaluation and write-up
- [ ] In-play pricing ([plan](docs/inplay-plan.md)):
  - [x] Step 1: analytic pricer from score and minute (exact in-play Dixon–Coles correction)
  - [x] Step 2: Monte Carlo match-state simulator, checked against Step 1 ([report](reports/inplay_mc_check.md))
  - [x] Step 3: time-varying goal rates and stoppage time ([report](reports/inplay_intensity.md))
  - [x] Step 4: score effects ([report](reports/inplay_score_effects.md))
  - [ ] Step 5: red cards
  - [x] Betfair in-play market data for 2024/25 (Aug–Jan), with timelines checked against results
  - [ ] Step 6: evaluation by minute, against final results and against the market
