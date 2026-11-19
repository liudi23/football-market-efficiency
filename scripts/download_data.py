"""Download football-data.co.uk CSVs into data/raw/.

Usage:
    python scripts/download_data.py                       # default leagues, 2005/06 onwards
    python scripts/download_data.py --leagues E0 E1 --from 2012
"""
import argparse
from datetime import date

from sportsq.data import download

DEFAULT_LEAGUES = ["E0", "E1", "E2", "E3", "SC0", "D1", "D2", "I1", "I2",
                   "SP1", "SP2", "F1", "F2", "N1", "B1", "P1", "T1", "G1"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES)
    ap.add_argument("--from", dest="start", type=int, default=2005)
    ap.add_argument("--to", dest="end", type=int,
                    default=date.today().year - (date.today().month < 8))
    a = ap.parse_args()
    paths = download(a.leagues, range(a.start, a.end + 1))
    print(f"{len(paths)} files in data/raw")
