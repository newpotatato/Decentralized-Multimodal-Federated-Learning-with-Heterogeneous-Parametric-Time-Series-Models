#!/usr/bin/env python3
"""
Per-article sentiment for the exogenous news channels.

  fontanka: Russian news leads   -> cointegrated/rubert-tiny-sentiment-balanced
  reuters : Reuters headlines    -> ProsusAI/finbert (FinBERT, Araci 2019)

Score of an article = P(positive) - P(negative) in [-1, 1].

Outputs (read by federated_learning/data_loaders):
  02_data_fontanka/fontanka_news_sentiment.csv                    date, sentiment_score, ...
  real_data_integration/reuters_2018_2021/reuters_daily_sentiment.csv  date, reuters_sentiment, n_articles

Requires torch + transformers (kept out of the main requirements; use a separate venv).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parents[1]
MODELS = {
    "fontanka": "cointegrated/rubert-tiny-sentiment-balanced",
    "reuters": "ProsusAI/finbert",
}
# Reuters URL slugs often start with wire-service markers ("update 2 ...", "rpt ...").
_WIRE_PREFIX = re.compile(r"^(?:(?:update|rpt|refile|corrected|wrapup|exclusive)\s*\d*\s+)+")


def score_texts(texts, model_name, batch_size, max_length):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    labels = {i: lab.lower() for i, lab in model.config.id2label.items()}
    pos = [i for i, lab in labels.items() if lab.startswith("pos")]
    neg = [i for i, lab in labels.items() if lab.startswith("neg")]
    if len(pos) != 1 or len(neg) != 1:
        raise ValueError(f"unexpected labels for {model_name}: {labels}")
    pos, neg = pos[0], neg[0]

    scores = np.empty(len(texts), dtype=float)
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1).numpy()
            scores[start:start + len(batch)] = probs[:, pos] - probs[:, neg]
            if (start // batch_size) % 50 == 0:
                print(f"  {start + len(batch)}/{len(texts)}", flush=True)
    return scores


def run_fontanka(args):
    src = HERE / "02_data_fontanka" / "news_clustered_final.csv"
    df = pd.read_csv(src, usecols=["date", "text"], encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[(df["date"] >= args.start) & (df["date"] <= args.end)].dropna(subset=["date"])
    texts = df["text"].fillna("").astype(str).str.strip().tolist()
    print(f"fontanka: {len(texts)} articles, {df['date'].dt.date.nunique()} days")
    df["sentiment_score"] = score_texts(texts, MODELS["fontanka"], args.batch_size, 256)
    out = HERE / "02_data_fontanka" / "fontanka_news_sentiment.csv"
    df[["date", "sentiment_score"]].to_csv(out, index=False, encoding="utf-8")
    print(f"saved {out}")


def run_reuters(args):
    folder = HERE / "real_data_integration" / "reuters_2018_2021"
    df = pd.read_csv(folder / "reuters_articles_gdelt.csv", parse_dates=["date"])
    df = df[(df["date"] >= args.start) & (df["date"] <= args.end)]
    df = df.dropna(subset=["headline"])
    df["headline"] = df["headline"].astype(str).str.strip().str.replace(_WIRE_PREFIX, "", regex=True)
    df = df[df["headline"].str.split().str.len() >= 3]
    if args.max_per_day > 0:
        # CPU budget: a fixed-seed sample of headlines per day is enough for a daily mean.
        df = (
            df.groupby(df["date"].dt.normalize(), group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), args.max_per_day), random_state=42))
            .sort_values(["date", "url"])
        )
    print(f"reuters: {len(df)} headlines, {df['date'].dt.date.nunique()} days")
    df["sentiment_score"] = score_texts(df["headline"].tolist(), MODELS["reuters"], args.batch_size, 48)
    df[["date", "url", "headline", "sentiment_score", "gdelt_avgtone"]].to_csv(
        folder / "reuters_headlines_sentiment.csv", index=False
    )
    daily = df.groupby(df["date"].dt.normalize()).agg(
        reuters_sentiment=("sentiment_score", "mean"), n_articles=("sentiment_score", "size")
    )
    daily.index.name = "date"
    out = folder / "reuters_daily_sentiment.csv"
    daily.reset_index().to_csv(out, index=False)
    print(f"saved {out} ({len(daily)} days)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("source", choices=["fontanka", "reuters"])
    p.add_argument("--start", default="2017-12-01")
    p.add_argument("--end", default="2021-08-14")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--max-per-day", type=int, default=50, help="Reuters headlines sampled per day (0 = all)")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    run_fontanka(args) if args.source == "fontanka" else run_reuters(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
