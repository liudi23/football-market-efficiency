"""Download match-level xG from Understat for the big-5 leagues, 2014/15-2022/23
(the 2023/24+ holdout is deliberately not fetched).  Standard library + certifi
only.  Run from the project folder on your own machine:

    python3 scripts/fetch_understat.py

Output: data/external/understat_matches.csv (one row per match).
Understat data is for personal, non-commercial research; cite understat.com.
"""
import codecs, csv, gzip, http.cookiejar, json, re, ssl, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external/understat_matches.csv"
LEAGUES = {"EPL": "E0", "La_liga": "SP1", "Bundesliga": "D1", "Serie_A": "I1", "Ligue_1": "F1"}
YEARS = range(2014, 2023)          # season start years: 2014/15 .. 2022/23
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"


def ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


CTX = ctx()
# keep cookies: the JSON endpoint may expect the session cookie set by the page
OPENER = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
    urllib.request.HTTPSHandler(context=CTX))


def get(url, xhr=False, referer=None):
    h = {"User-Agent": UA, "Accept-Encoding": "gzip", "Accept-Language": "en-GB,en;q=0.9"}
    if xhr:
        h.update({"X-Requested-With": "XMLHttpRequest",
                  "Accept": "application/json, text/javascript, */*; q=0.01"})
    if referer:
        h["Referer"] = referer
    req = urllib.request.Request(url, headers=h)
    with OPENER.open(req, timeout=60) as r:
        body = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
    return body.decode("utf-8", "replace")


def matches(league, year):
    """Understat's JSON endpoint first; fall back to the data embedded in the page."""
    errors = []
    page_url = f"https://understat.com/league/{league}/{year}"
    html = ""
    try:
        html = get(page_url)                      # sets cookies; older format embeds data
        m = re.search(r"datesData\s*=\s*JSON\.parse\('(.*?)'\)", html)
        if m:
            return json.loads(codecs.decode(m.group(1), "unicode_escape"))
    except Exception as e:
        errors.append(f"page: {e}")
    body = ""
    try:
        body = get(f"https://understat.com/getLeagueData/{league}/{year}", xhr=True, referer=page_url)
        d = json.loads(body)
        if isinstance(d, dict) and "dates" in d:
            return d["dates"]
        errors.append(f"json keys {list(d)[:8] if isinstance(d, dict) else type(d)}")
    except Exception as e:
        errors.append(f"json endpoint: {e}; first 300 chars of reply: {body[:300]!r}")
        # show what data the page loads, to adapt the script if Understat changed again
        hints = sorted(set(re.findall(r"""['"](/get[A-Za-z]+Data[^'"]*)['"]""", html)))
        errors.append(f"data URLs mentioned in the page: {hints[:10]}")
    raise RuntimeError("\n".join(errors))


def main():
    rows = []
    for league, code in LEAGUES.items():
        for y in YEARS:
            ms = matches(league, y)
            n = 0
            for m in ms:
                if not m.get("isResult"):
                    continue
                rows.append([code, f"{y % 100:02d}{(y + 1) % 100:02d}", m["datetime"],
                             m["h"]["title"], m["a"]["title"], m["goals"]["h"], m["goals"]["a"],
                             m["xG"]["h"], m["xG"]["a"]])
                n += 1
            print(f"{league} {y}/{y + 1 - 2000:02d}: {n} matches", flush=True)
            time.sleep(1.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["league", "season", "date", "home_us", "away_us", "hg_us", "ag_us", "hxg", "axg"])
        w.writerows(rows)
    print(f"saved {len(rows)} matches to {OUT}")


if __name__ == "__main__":
    try:
        main()
    except ssl.SSLError as e:
        sys.exit(f"SSL error ({e}). Run:  python3 -m pip install certifi   then retry.")
