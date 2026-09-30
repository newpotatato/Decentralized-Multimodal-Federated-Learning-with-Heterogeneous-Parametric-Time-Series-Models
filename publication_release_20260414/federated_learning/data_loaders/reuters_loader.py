#!/usr/bin/env python3
"""
Lightweight Reuters loader for exogenous features.

Reads Reuters-21578 text files (ApteMod) from `real_data_integration/reuters/reuters/{training,test}`
and builds a daily sentiment series aligned to a target date index (e.g., MCC dates).

Design choices:
- No heavyweight NLP deps: simple lexicon-based sentiment.
- Safe temporal alignment: use article dates parsed from text; never positional mapping.
- Caching: saves parquet to `artifacts/cache/reuters_daily.parquet` to avoid recompute.
"""

import re
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

POS_WORDS = {
    "good", "strong", "growth", "gain", "rise", "rises", "improve", "surge", "positive", "profit",
    "record", "beat", "boost", "advance", "increase", "up", "bull", "optimistic", "support",
}
NEG_WORDS = {
    "bad", "weak", "loss", "drop", "falls", "fall", "decline", "negative", "cut", "cuts", "down",
    "bear", "pessimistic", "risk", "warn", "warning", "slow", "slump", "recession", "deficit",
}


def _lexicon_sentiment(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    pos = sum(1 for w in words if any(p in w for p in POS_WORDS))
    neg = sum(1 for w in words if any(n in w for n in NEG_WORDS))
    return (pos - neg) / max(len(words), 1)


def _iter_reuters_files(root: Path) -> Iterable[Path]:
    for split in ("training", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for fp in sorted(split_dir.iterdir(), key=lambda p: p.name):
            if fp.is_file():
                yield fp


# Daily sentiment of 2017-2021 Reuters headlines (experiments_datafusion2023/data_prep).
# Preferred over the 1987 Reuters-21578 corpus, whose dates do not overlap the MCC period.
REUTERS_DAILY_CSV = Path("reuters_2018_2021") / "reuters_daily_sentiment.csv"


def load_reuters_daily_csv(base_path: Path) -> Optional[pd.Series]:
    """Daily Reuters sentiment indexed by date, or None if the CSV is absent."""
    path = Path(base_path) / REUTERS_DAILY_CSV
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    series = df.set_index("date")["reuters_sentiment"].astype(float).sort_index()
    return series[~series.index.duplicated()]


_MONTH_RE = (
    r"(?:JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|"
    r"JUL(?:Y)?|AUG(?:UST)?|SEP(?:T(?:EMBER)?)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)"
)


def _extract_article_date(text: str) -> Optional[pd.Timestamp]:
    """
    Try to extract a publish-like date from Reuters article text.
    Returns normalized timestamp or None if no reliable date is found.
    """
    patterns = [
        # e.g. APRIL 17 1987 / Apr 17, 1987
        rf"\b{_MONTH_RE}\s+\d{{1,2}}(?:,)?\s+\d{{4}}\b",
        # e.g. 17 APRIL 1987
        rf"\b\d{{1,2}}\s+{_MONTH_RE}\s+\d{{4}}\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        ts = pd.to_datetime(m.group(0), errors="coerce")
        if pd.notna(ts):
            return pd.Timestamp(ts).normalize()
    return None


def build_reuters_daily(base_path: Path, target_dates: pd.Series, cache_dir: Optional[Path] = None, max_files: int = 500) -> pd.DataFrame:
    """
    Build a daily sentiment series aligned to target_dates.

    Args:
        base_path: path to real_data_integration folder
        target_dates: pd.Series of datetime64 dates to align to (e.g., MCC date column)
        cache_dir: optional cache directory (default: artifacts/cache)
        max_files: limit number of files to process for faster runtime
    Returns:
        DataFrame with columns ['date', 'reuters_sentiment'] aligned to target_dates length.
    """
    if cache_dir is None:
        cache_dir = base_path / "artifacts" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Versioned cache name to avoid reusing earlier positional-alignment artifacts.
    cache_file = cache_dir / "reuters_daily_strict.parquet"

    target_dates = pd.to_datetime(target_dates, errors="coerce")
    if isinstance(target_dates, pd.Series):
        target_dates = target_dates.dt.normalize()
    else:
        target_dates = pd.DatetimeIndex(target_dates).normalize()

    if cache_file.exists():
        try:
            cached = pd.read_parquet(cache_file)
            if len(cached) == len(target_dates) and "reuters_sentiment" in cached.columns:
                return cached
        except Exception:
            pass

    corpus_root = base_path / "reuters" / "reuters"
    if not corpus_root.exists():
        raise FileNotFoundError(f"Reuters corpus not found at {corpus_root}")

    dated_rows = []
    file_count = 0
    for fp in _iter_reuters_files(corpus_root):
        if file_count >= max_files:
            break
        try:
            with open(fp, 'r', encoding='latin-1', errors='ignore') as f:
                text = f.read()
            article_date = _extract_article_date(text)
            if article_date is not None:
                dated_rows.append((article_date, _lexicon_sentiment(text)))
            file_count += 1
        except Exception:
            file_count += 1

    if not dated_rows:
        raise ValueError(
            "Reuters articles have no parseable dates; refusing positional alignment to avoid temporal leakage."
        )

    dated_df = pd.DataFrame(dated_rows, columns=["date", "sentiment"])
    daily = dated_df.groupby("date", as_index=True)["sentiment"].mean().sort_index()

    aligned = (
        daily.reindex(target_dates)
        .ffill()
        .bfill()
        .fillna(0.0)
    )

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(target_dates.values),
            "reuters_sentiment": aligned.values,
        }
    )

    try:
        df.to_parquet(cache_file, index=False)
    except Exception:
        pass

    return df


if __name__ == "__main__":
    base = Path(__file__).parent
    # Minimal self-test with synthetic dates
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = build_reuters_daily(base, pd.Series(dates))
    print(df.head())
