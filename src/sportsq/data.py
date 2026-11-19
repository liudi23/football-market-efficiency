"""Download and standardise football-data.co.uk CSVs.

Data notes (from football-data.co.uk):
- Pre-closing odds exist for all seasons; Pinnacle (PS*) from 2012/13.
- Since 2019/20 two sets of odds are collected: one after market opening
  (no suffix) and the closing odds (extra "C", e.g. PSCH, B365CH).
- Pinnacle closing 1X2 odds are also available back to 2012/13.

Output schema (one row per match), shared with `simulate.py` so every
downstream module runs unchanged on real and synthetic data:

    date, season, league, home, away, hg, ag, result,
    hs, as_, hst, ast,                      # shots / shots on target
    open_h, open_d, open_a,                 # price you could bet into (Pinnacle pre-closing)
    close_h, close_d, close_a,              # Pinnacle closing (benchmark only, never a feature)
    max_h, max_d, max_a, avg_h, avg_d, avg_a,
    ou_over, ou_under, ou_close_over, ou_close_under
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# Main-league codes on football-data.co.uk
LEAGUES = {
    "E0": "England Premier League", "E1": "England Championship",
    "E2": "England League One", "E3": "England League Two",
    "SC0": "Scotland Premiership", "D1": "Germany Bundesliga 1",
    "D2": "Germany Bundesliga 2", "I1": "Italy Serie A", "I2": "Italy Serie B",
    "SP1": "Spain La Liga", "SP2": "Spain Segunda", "F1": "France Ligue 1",
    "F2": "France Ligue 2", "N1": "Netherlands Eredivisie",
    "B1": "Belgium Pro League", "P1": "Portugal Primeira",
    "T1": "Turkey Super Lig", "G1": "Greece Super League",
}

# first-choice column, then fallbacks (column names changed over the years)
COLUMN_MAP = {
    "open_h": ["PSH", "PH"], "open_d": ["PSD", "PD"], "open_a": ["PSA", "PA"],
    "close_h": ["PSCH"], "close_d": ["PSCD"], "close_a": ["PSCA"],
    "max_h": ["MaxH", "BbMxH"], "max_d": ["MaxD", "BbMxD"], "max_a": ["MaxA", "BbMxA"],
    "avg_h": ["AvgH", "BbAvH"], "avg_d": ["AvgD", "BbAvD"], "avg_a": ["AvgA", "BbAvA"],
    "ou_over": ["P>2.5", "BbMx>2.5"], "ou_under": ["P<2.5", "BbMx<2.5"],
    "ou_close_over": ["PC>2.5"], "ou_close_under": ["PC<2.5"],
    "hs": ["HS"], "as_": ["AS"], "hst": ["HST"], "ast": ["AST"],
}


def season_code(start_year: int) -> str:
    """2019 -> '1920'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def download(leagues, start_years, raw_dir="data/raw", overwrite=False):
    """Download raw CSVs into raw_dir/{league}_{season}.csv. Returns list of paths."""
    import requests

    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    paths = []
    for lg in leagues:
        for y in start_years:
            s = season_code(y)
            path = raw / f"{lg}_{s}.csv"
            if path.exists() and not overwrite:
                paths.append(path)
                continue
            r = requests.get(BASE_URL.format(season=s, league=lg), timeout=30)
            if r.status_code != 200:
                print(f"skip {lg} {s}: HTTP {r.status_code}")
                continue
            path.write_bytes(r.content)
            paths.append(path)
    return paths


def _first(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def standardise(raw: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], format="mixed", dayfirst=True, errors="coerce"),
        "season": season,
        "league": league,
        "home": raw["HomeTeam"].str.strip(),
        "away": raw["AwayTeam"].str.strip(),
        "hg": raw["FTHG"].astype(int),
        "ag": raw["FTAG"].astype(int),
    })
    for col, names in COLUMN_MAP.items():
        out[col] = _first(raw, names).values
    out["result"] = np.select([out.hg > out.ag, out.hg == out.ag], ["H", "D"], "A")
    return out


def load(raw_dir="data/raw", leagues=None) -> pd.DataFrame:
    """Load every downloaded CSV into one standardised, date-sorted frame."""
    frames = []
    for path in sorted(Path(raw_dir).glob("*.csv")):
        lg, s = path.stem.split("_")
        if leagues and lg not in leagues:
            continue
        text = path.read_bytes().decode("latin-1")
        raw = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
        frames.append(standardise(raw, lg, s))
    if not frames:
        raise FileNotFoundError(f"No CSVs in {raw_dir}; run scripts/download_data.py")
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["date", "league", "home"]).reset_index(drop=True)
