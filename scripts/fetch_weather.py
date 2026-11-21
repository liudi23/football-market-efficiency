"""Download hourly weather for every ground in data/external/grounds.csv from
the free Open-Meteo archive (ERA5 reanalysis; no API key).  Standard library
only, so it runs with any python3.  Resumable: grounds already saved are
skipped.  Run from the project folder:

    python3 scripts/fetch_weather.py

Output: data/external/weather/<team>.csv  (time in Europe/London local time)
Open-Meteo's free tier is for non-commercial use (CC BY 4.0, cite Open-Meteo).
"""
import csv, json, ssl, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "weather"
# one request per ground per season: Open-Meteo counts every 2 weeks of data
# as one call (free limits about 5,000 calls an hour and 10,000 a day), so the
# whole download is ~8,500 calls and is paced to take about two hours.
SEASONS = [("2019-08-01", "2020-08-02"), ("2020-09-11", "2021-05-24"),
           ("2021-08-05", "2022-05-30"), ("2022-07-28", "2023-05-30")]
PAUSE = 20  # seconds between requests
VARS = ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m"]
URL = "https://archive-api.open-meteo.com/v1/archive"


def _ssl_context():
    """python.org builds on macOS ship without CA certificates; use certifi's
    bundle when it is installed (pip install certifi)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


CTX = _ssl_context()


def fetch(lat, lon, start, end, tries=8):
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": ",".join(VARS), "timezone": "Europe/London",
        "wind_speed_unit": "kmh"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(f"{URL}?{q}", timeout=60, context=CTX) as r:
                return json.load(r)
        except urllib.error.URLError as e:
            if isinstance(e.reason, ssl.SSLError):         # retrying won't help
                sys.exit("SSL certificate error. Run:  pip install certifi\n"
                         "then start this script again.")
            wait = 120 * (k + 1)
            print(f"   retry in {wait}s ({e})", flush=True)
            time.sleep(wait)
        except Exception as e:                       # rate limit or network
            wait = 120 * (k + 1)                     # hourly limit resets
            print(f"   retry in {wait}s ({e})", flush=True)
            time.sleep(wait)
    raise RuntimeError("giving up")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grounds = list(csv.DictReader(open(ROOT / "data/external/grounds.csv")))
    for i, g in enumerate(grounds, 1):
        path = OUT / (g["team"].replace("'", "").replace(" ", "_") + ".csv")
        if path.exists():
            continue
        left = len(grounds) - i + 1
        print(f"[{i}/{len(grounds)}] {g['team']}  (~{left * len(SEASONS) * PAUSE / 60:.0f} min left)",
              flush=True)
        rows = []
        for start, end in SEASONS:
            h = fetch(g["lat"], g["lon"], start, end)["hourly"]
            rows += list(zip(h["time"], *[h[v] for v in VARS]))
            time.sleep(PAUSE)
        tmp = path.with_suffix(".part")
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time"] + VARS)
            w.writerows(rows)
        tmp.rename(path)
    n = len(list(OUT.glob("*.csv")))
    print(f"done: {n}/{len(grounds)} grounds in {OUT}")


if __name__ == "__main__":
    sys.exit(main())
