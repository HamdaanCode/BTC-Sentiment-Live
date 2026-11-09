"""Aggregate per-tweet VADER scores into the daily LSTM training CSV.

Takes the output of src.score_vader (per-tweet scores for Models A, B, C) and
daily BTC OHLCV, produces the same training CSV format the dissertation's
LSTM pipeline consumes.

Output schema (identical to reference/data/NEW_lstm_training_data.csv):

  date, tweet_count,
  sentiment_{a,b,c}_{mean,std,median,min,max},
  pct_{positive,neutral,negative}_{a,b,c}   (VADER author thresholds ±0.05),
  open, high, low, close, volume,
  price_change, price_change_pct, volume_change_pct,
  price_ma_7, price_volatility_7, price_momentum

price_momentum is defined as (close - price_ma_7), matching the dissertation's
training CSV (verified against row 2 of NEW_lstm_training_data.csv).

Usage:
  python -m src.aggregate_vader \\
      --scored-tweets artifacts/raw/2022_scored.csv \\
      --prices        artifacts/raw/bitcoin_prices_2022.csv \\
      --output        reference/data/NEW_lstm_training_data_2022.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MODELS = ("a", "b", "c")
POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05


def _daily_sentiment_stats(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-day mean/std/median/min/max + pos/neu/neg % for each model."""
    scored = scored.copy()
    if "date" not in scored.columns:
        if "datetime" not in scored.columns:
            raise SystemExit("scored tweets CSV needs 'date' or 'datetime' column")
        scored["date"] = pd.to_datetime(scored["datetime"], utc=True, errors="coerce").dt.date
    scored["date"] = pd.to_datetime(scored["date"])

    frames = []
    counts = scored.groupby("date").size().rename("tweet_count").reset_index()
    frames.append(counts)

    for m in MODELS:
        col = f"sentiment_model_{m}"
        if col not in scored.columns:
            raise SystemExit(
                f"scored tweets CSV missing {col}; "
                "run src.score_vader first"
            )
        agg = scored.groupby("date")[col].agg(
            ["mean", "std", "median", "min", "max"]
        )
        agg.columns = [f"sentiment_{m}_{s}" for s in agg.columns]
        agg = agg.reset_index()
        frames.append(agg)

        # Positive/neutral/negative percentages per model.
        s = scored[col]
        scored[f"_pos_{m}"] = (s > POS_THRESHOLD).astype(int)
        scored[f"_neu_{m}"] = ((s >= NEG_THRESHOLD) & (s <= POS_THRESHOLD)).astype(int)
        scored[f"_neg_{m}"] = (s < NEG_THRESHOLD).astype(int)

    pct_frames = []
    for m in MODELS:
        pcts = (
            scored.groupby("date")[[f"_pos_{m}", f"_neu_{m}", f"_neg_{m}"]]
            .mean()
            .mul(100)
            .rename(columns={
                f"_pos_{m}": f"pct_positive_{m}",
                f"_neu_{m}": f"pct_neutral_{m}",
                f"_neg_{m}": f"pct_negative_{m}",
            })
            .reset_index()
        )
        pct_frames.append(pcts)

    out = frames[0]
    for df_ in frames[1:] + pct_frames:
        out = out.merge(df_, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)


def _price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Derive the dissertation's price features from daily OHLCV."""
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(prices.columns.str.lower())
    if missing:
        raise SystemExit(f"prices CSV missing columns: {sorted(missing)}")

    p = prices.copy()
    p.columns = [c.lower() for c in p.columns]
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values("date").reset_index(drop=True)

    p["price_change"] = p["close"] - p["close"].shift(1)
    p["price_change_pct"] = p["close"].pct_change() * 100.0
    p["volume_change_pct"] = p["volume"].pct_change() * 100.0
    p["price_ma_7"] = p["close"].rolling(window=7, min_periods=1).mean()
    p["price_volatility_7"] = p["close"].rolling(window=7, min_periods=2).std()
    p["price_momentum"] = p["close"] - p["price_ma_7"]

    return p[
        [
            "date", "open", "high", "low", "close", "volume",
            "price_change", "price_change_pct", "volume_change_pct",
            "price_ma_7", "price_volatility_7", "price_momentum",
        ]
    ]


def run(scored_path: Path, prices_path: Path, output_path: Path) -> int:
    print(f"loading scored tweets: {scored_path}")
    scored = pd.read_csv(scored_path, on_bad_lines="skip", engine="python")
    print(f"  {len(scored):,} tweets")

    print(f"loading BTC prices: {prices_path}")
    prices = pd.read_csv(prices_path)
    print(f"  {len(prices):,} days")

    daily_sent = _daily_sentiment_stats(scored)
    price_feats = _price_features(prices)

    merged = daily_sent.merge(price_feats, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    # Canonical column order matches reference/data/NEW_lstm_training_data.csv.
    cols = ["date", "tweet_count"]
    for m in MODELS:
        cols += [f"sentiment_{m}_{s}" for s in ("mean", "std", "median", "min", "max")]
    for m in MODELS:
        cols += [f"pct_positive_{m}", f"pct_neutral_{m}", f"pct_negative_{m}"]
    cols += [
        "open", "high", "low", "close", "volume",
        "price_change", "price_change_pct", "volume_change_pct",
        "price_ma_7", "price_volatility_7", "price_momentum",
    ]
    merged = merged[cols]
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(
        f"wrote {len(merged):,} rows -> {output_path} "
        f"({merged['date'].iloc[0]} -> {merged['date'].iloc[-1]})"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scored-tweets", type=Path, required=True)
    p.add_argument("--prices", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args.scored_tweets, args.prices, args.output)


if __name__ == "__main__":
    sys.exit(main())
