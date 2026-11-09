"""Clean raw tweet text for FinBERT (and VADER) scoring.

Mirrors the dissertation preprocessing (`reference/dissertation-code/
tweet_preprocessing.py`) so the Phase 2 FinBERT runs are directly comparable
to the submitted Model A/B/C numbers:

  - lowercase
  - strip retweet markers, URLs, @mentions
  - strip the `#` character but keep the hashtag word
  - collapse whitespace, drop rows with empty cleaned text
  - deduplicate on (datetime, text, username)

Usage:
    python -m src.preprocess \\
        --input artifacts/raw/sept_nov_2021.csv \\
        --output artifacts/raw/sept_nov_2021_cleaned.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


URL_RE = re.compile(r"http\S+|www\S+|https\S+", flags=re.MULTILINE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
RT_RE = re.compile(r"\brt\b")
WS_RE = re.compile(r"\s+")


def clean(text) -> str:
    if pd.isna(text):
        return ""
    t = str(text).lower()
    t = RT_RE.sub("", t)
    t = URL_RE.sub("", t)
    t = MENTION_RE.sub("", t)
    t = HASHTAG_RE.sub(r"\1", t)
    return WS_RE.sub(" ", t).strip()


def run(input_path: Path, output_path: Path) -> int:
    df = pd.read_csv(input_path, on_bad_lines="skip", engine="python")
    print(f"loaded {len(df):,} rows from {input_path}")

    df = df.dropna(subset=["datetime", "text"])
    before = len(df)
    df = df.drop_duplicates(subset=["datetime", "text", "username"])
    print(f"  dedup: {before:,} -> {len(df):,}")

    df["text_cleaned"] = df["text"].map(clean)
    before = len(df)
    df = df[df["text_cleaned"].str.len() > 0].copy()
    print(f"  dropped empty-cleaned: {before:,} -> {len(df):,}")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])
    df["date"] = df["datetime"].dt.date

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[["datetime", "date", "username", "text", "text_cleaned"]].to_csv(
        output_path, index=False
    )
    print(f"wrote {len(df):,} rows -> {output_path}")
    return len(df)


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
