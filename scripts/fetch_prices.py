"""Fetch BTC OHLCV for an arbitrary date window as a CSV.

Thin wrapper around yfinance that handles the MultiIndex column quirk
and writes a flat CSV in the exact shape src.aggregate_vader expects:
columns = date, open, high, low, close, volume.

Use this for one-off historical pulls (e.g. the 2022 bear market window
for the L2 retrain). For the live daily cron use scripts/daily_btc_ingest.py
instead; that one maintains an append-only CSV.

Usage:
  python scripts/fetch_prices.py --start 2022-01-01 --end 2023-01-01 \\
                                 --output artifacts/raw/btc_prices_2022.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKER = "BTC-USD"


def fetch(start: str, end: str, ticker: str = TICKER) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        raise SystemExit(f"yfinance returned no rows for {ticker} {start} -> {end}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"]).dt.strftime("%Y-%m-%d"),
        "open": raw["Open"].round(2),
        "high": raw["High"].round(2),
        "low": raw["Low"].round(2),
        "close": raw["Close"].round(2),
        "volume": raw["Volume"].astype("int64"),
    })
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", required=True, help="inclusive start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="exclusive end date YYYY-MM-DD")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ticker", default=TICKER)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    df = fetch(args.start, args.end, args.ticker)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(
        f"wrote {len(df):,} rows -> {args.output} "
        f"({df['date'].iloc[0]} -> {df['date'].iloc[-1]})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
