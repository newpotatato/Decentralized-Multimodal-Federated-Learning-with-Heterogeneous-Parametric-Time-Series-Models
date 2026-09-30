#!/usr/bin/env python3
"""
Collect Reuters articles (URL, headline slug, publication day) for 2017-12 .. 2021-08
from the GDELT 1.0 daily event exports (http://data.gdeltproject.org/events/).

Each daily export lists the events added that day together with a SOURCEURL.
We keep rows whose SOURCEURL is on reuters.com, deduplicate by URL (first day seen)
and recover the headline from the URL slug, e.g.
  .../article/us-canada-cannabis/health-canada-to-allow-some-edible-...-idUKKCN1TF2B4
  -> "health canada to allow some edible ..."

Output: real_data_integration/reuters_2018_2021/reuters_articles_gdelt.csv
        columns: date, url, headline, gdelt_avgtone
Daily parts are cached in .../parts/ so the download can be resumed.
"""

import argparse
import io
import re
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

GDELT_URL = "https://data.gdeltproject.org/events/{day}.export.CSV.zip"
# GDELT 1.0 event export: 1 = SQLDATE, 34 = AvgTone, 56 = DATEADDED, 57 = SOURCEURL
USECOLS = [34, 56, 57]
NAMES = ["gdelt_avgtone", "date_added", "url"]
SLUG_RE = re.compile(r"/article/(?:[^/?#]+/)?([A-Za-z0-9-]{8,}?)-id[A-Z0-9]{6,}")


def headline_from_url(url: str):
    m = SLUG_RE.search(url)
    if not m:
        return None
    text = m.group(1).replace("-", " ").strip()
    return text or None


def fetch_day(day: str, parts_dir: Path, retries: int = 3) -> str:
    out = parts_dir / f"{day}.csv"
    if out.exists():
        # A part interrupted mid-write lacks the trailing newline; fetch it again.
        data = out.read_bytes()
        if data.endswith(b"\n") and data.startswith(b"date,url"):
            return f"{day}: cached"
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(GDELT_URL.format(day=day), timeout=120) as resp:
                payload = resp.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                raw = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")
            # Plain substring filter first: only a small share of rows cite reuters.com.
            tones = {}
            for line in raw.splitlines():
                if "reuters.com" not in line:
                    continue
                fields = line.split("\t")
                if len(fields) <= USECOLS[2]:
                    continue
                url = fields[USECOLS[2]].strip()
                if "reuters.com" not in url:
                    continue
                try:
                    tones.setdefault(url, []).append(float(fields[USECOLS[0]]))
                except ValueError:
                    continue
            df = pd.DataFrame({
                "url": list(tones),
                "gdelt_avgtone": [sum(v) / len(v) for v in tones.values()],
            })
            df.insert(0, "date", pd.to_datetime(day, format="%Y%m%d").date())
            df["headline"] = df["url"].map(headline_from_url)
            tmp = out.with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(out)
            return f"{day}: {len(df)} reuters urls"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return f"{day}: MISSING (404)"
            last_err = e
        except Exception as e:  # network hiccups, truncated zips
            last_err = e
        time.sleep(2 * (attempt + 1))
    return f"{day}: FAILED ({type(last_err).__name__}: {last_err})"


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2017-12-01")
    p.add_argument("--end", default="2021-08-14")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out-dir", default=str(here / "real_data_integration" / "reuters_2018_2021"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    days = [d.strftime("%Y%m%d") for d in pd.date_range(args.start, args.end, freq="D")]

    problems = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_day, d, parts_dir): d for d in days}
        for i, fut in enumerate(as_completed(futures), 1):
            msg = fut.result()
            if "MISSING" in msg or "FAILED" in msg:
                problems.append(msg)
            if i % 50 == 0 or i == len(days):
                print(f"[{i}/{len(days)}] {msg}", flush=True)

    frames = [pd.read_csv(f) for f in sorted(parts_dir.glob("*.csv"))]
    allrows = pd.concat(frames, ignore_index=True).sort_values(["date", "url"])
    # An article may be re-reported on later days: keep the first day it appeared.
    allrows = allrows.drop_duplicates("url", keep="first")
    out_csv = out_dir / "reuters_articles_gdelt.csv"
    allrows.to_csv(out_csv, index=False)

    print(f"days requested: {len(days)}, parts on disk: {len(frames)}")
    print(f"unique reuters articles: {len(allrows)}, with headline: {allrows['headline'].notna().sum()}")
    print(f"saved: {out_csv}")
    if problems:
        print("problems:")
        for m in problems:
            print("  ", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
