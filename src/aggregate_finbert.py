"""Aggregate per-tweet FinBERT scores to a daily table.

Produces the same shape of columns the dissertation's VADER pipeline did
(mean, std, median, min, max, plus pos/neu/neg percentages) so downstream
training code sees a drop-in replacement, just with FinBERT instead of VADER.

Positive/negative/neutral class percentages are derived from the FinBERT
argmax, which matches the native FinBERT output more cleanly than imposing
a VADER-style ±0.05 threshold on the compound.

Usage:
    python -m src.aggregate_finbert \\
        --input artifacts/raw/sept_nov_2021_finbert.csv \\
        --output artifacts/raw/sept_nov_2021_finbert_daily.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _argmax_label(row) -> str:
    labels = ["positive", "negative", "neutral"]
    values = [row["sentiment_pos"], row["sentiment_neg"], row["sentiment_neu"]]
    return labels[int(max(range(3), key=lambda i: values[i]))]


def run(input_path: Path, output_path: Path) -> int:
    df = pd.read_csv(input_path, on_bad_lines="skip", engine="python")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"loaded {len(df):,} per-tweet rows across {df['date'].nunique()} days")

    df["label"] = df.apply(_argmax_label, axis=1)

    agg = df.groupby("date").agg(
        tweet_count=("sentiment_finbert", "count"),
        sentiment_finbert_mean=("sentiment_finbert", "mean"),
        sentiment_finbert_std=("sentiment_finbert", "std"),
        sentiment_finbert_median=("sentiment_finbert", "median"),
        sentiment_finbert_min=("sentiment_finbert", "min"),
        sentiment_finbert_max=("sentiment_finbert", "max"),
    ).reset_index()

    # Class percentages per day
    def pct(label):
        return (
            df.groupby("date")["label"]
            .apply(lambda s: (s == label).mean() * 100)
            .rename(f"pct_{label}_finbert")
        )

    for label in ("positive", "neutral", "negative"):
        agg = agg.merge(pct(label), on="date", how="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_path, index=False)
    print(f"wrote {len(agg)} daily rows -> {output_path}")
    return len(agg)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.input, args.output)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
