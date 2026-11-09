"""Build a Phase 2 LSTM training CSV from FinBERT daily aggregates + BTC prices.

Produces the same column layout the dissertation's Phase 1 CSV used, except
the three VADER sentiment families (a, b, c) are replaced by a single
`sentiment_finbert_*` family. That keeps the training/evaluation pipeline
unchanged for Models D and E.

Usage:
    # Build Model D training frame (Sept-Nov 2021)
    python -m src.data_finbert \\
        --daily artifacts/raw/sept_nov_2021_finbert_daily.csv \\
        --prices reference/data/NEW_bitcoin_prices_sept_nov_2021.csv \\
        --output artifacts/phase2/finbert_2021_training.csv

    # Build Model E training frame (May-June 2022 bear window)
    python -m src.data_finbert \\
        --daily artifacts/raw/may_nov_2022_finbert_daily.csv \\
        --prices artifacts/raw/btc_prices_may_nov_2022.csv \\
        --output artifacts/phase2/finbert_2022_training.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _normalise_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    keep = ["date", "open", "high", "low", "close", "volume"]
    return df[keep]


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["price_change"] = df["close"].diff()
    df["price_change_pct"] = df["close"].pct_change() * 100
    df["volume_change_pct"] = df["volume"].pct_change() * 100
    df["price_ma_7"] = df["close"].rolling(window=7, min_periods=1).mean()
    df["price_volatility_7"] = df["close"].rolling(window=7, min_periods=1).std()
    df["price_momentum"] = df["close"] - df["price_ma_7"]
    return df


def build(daily_path: Path, prices_path: Path, output_path: Path) -> int:
    daily = pd.read_csv(daily_path)
    daily["date"] = pd.to_datetime(daily["date"]).dt.date

    prices = pd.read_csv(prices_path)
    prices = _normalise_prices(prices)

    merged = daily.merge(prices, on="date", how="inner")
    print(f"daily: {len(daily)}  prices: {len(prices)}  merged: {len(merged)}")

    merged = _add_derived(merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"wrote {len(merged)} rows x {len(merged.columns)} cols -> {output_path}")
    return len(merged)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--daily", type=Path, required=True)
    p.add_argument("--prices", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build(args.daily, args.prices, args.output)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
