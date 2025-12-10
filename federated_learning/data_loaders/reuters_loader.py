#!/usr/bin/env python3
"""
Lightweight Reuters loader for exogenous features.

Reads Reuters-21578 text files (ApteMod) from `real_data_integration/reuters/reuters/{training,test}`
and builds a daily sentiment series aligned to a target date index (e.g., MCC dates).

Design choices:
- No heavyweight NLP deps: simple lexicon-based sentiment.
- Deterministic ordering: sort filenames and map sequentially to provided date index.
- Caching: saves parquet to `artifacts/cache/reuters_daily.parquet` to avoid recompute.
"""

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
    cache_file = cache_dir / "reuters_daily.parquet"

    if cache_file.exists():
        try:
            cached = pd.read_parquet(cache_file)
            if len(cached) == len(target_dates):
                return cached
        except Exception:
            pass

    corpus_root = base_path / "reuters" / "reuters"
    if not corpus_root.exists():
        raise FileNotFoundError(f"Reuters corpus not found at {corpus_root}")

    sentiments = []
    file_count = 0
    for fp in _iter_reuters_files(corpus_root):
        if file_count >= max_files:
            break
        try:
            with open(fp, 'r', encoding='latin-1', errors='ignore') as f:
                text = f.read()
            sentiments.append(_lexicon_sentiment(text))
            file_count += 1
        except Exception:
            sentiments.append(0.0)
            file_count += 1

    if not sentiments:
        sentiments = [0.0]

    # Map sequentially to target dates
    sentiments_series = pd.Series(sentiments[: len(target_dates)], index=target_dates.index)
    # If fewer sentiments than dates, pad with last value
    if len(sentiments_series) < len(target_dates):
        last = sentiments_series.iloc[-1] if len(sentiments_series) else 0.0
        pad_len = len(target_dates) - len(sentiments_series)
        sentiments_series = pd.concat([
            sentiments_series,
            pd.Series([last] * pad_len, index=target_dates.index[len(sentiments_series):]),
        ])

    df = pd.DataFrame({
        "date": pd.to_datetime(target_dates.values),
        "reuters_sentiment": sentiments_series.values,
    })

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
