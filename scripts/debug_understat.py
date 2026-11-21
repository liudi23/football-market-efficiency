"""Save Understat's league page and its scripts so the data request can be found."""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fetch_understat import get

out = Path(__file__).resolve().parents[1] / "data/external/understat_debug"
out.mkdir(parents=True, exist_ok=True)
page = "https://understat.com/league/EPL/2019"
html = get(page)
(out / "league_page.html").write_text(html)
srcs = re.findall(r'<script[^>]+src="([^"]+)"', html)
print("scripts:", srcs)
for s in srcs:
    url = s if s.startswith("http") else "https://understat.com" + ("" if s.startswith("/") else "/") + s
    try:
        (out / (url.rsplit("/", 1)[-1].split("?")[0] or "script.js")).write_text(get(url, referer=page))
    except Exception as e:
        print("failed", url, e)
print("saved to", out)
