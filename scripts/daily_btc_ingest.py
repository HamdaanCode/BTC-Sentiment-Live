"""Daily BTC price ingest.

Appends today's BTC-USD OHLCV row to data/daily/btc_prices.csv.
On first run (or when the CSV is empty) backfills from 2024-01-10,
the day the first spot Bitcoin ETFs began trading in the US.
Safe to re-run: duplicate dates are skipped.

Runs in GitHub Actions (see .github/workflows/daily-ingest.yml) and
can also be run locally: `python scripts/daily_btc_ingest.py`.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "daily" / "btc_prices.csv"
JSON_PATH = REPO_ROOT / "data" / "daily" / "latest_btc.json"
BACKFILL_START = "2024-01-10"
TICKER = "BTC-USD"

COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def fetch_range(start: str, end: str) -> pd.DataFrame:
    """Return OHLCV rows for [start, end) as a tidy DataFrame."""
    raw = yf.download(TICKER, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    # yfinance returns a MultiIndex column frame when one ticker is passed; flatten.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    raw["date"] = pd.to_datetime(raw["Date"]).dt.strftime("%Y-%m-%d")
    out = pd.DataFrame({
        "date": raw["date"],
        "open": raw["Open"].round(2),
        "high": raw["High"].round(2),
        "low": raw["Low"].round(2),
        "close": raw["Close"].round(2),
        "volume": raw["Volume"].astype("int64"),
    })
    return out


def load_existing() -> pd.DataFrame:
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        existing = pd.read_csv(CSV_PATH)
        if not existing.empty and "date" in existing.columns:
            return existing
    return pd.DataFrame(columns=COLUMNS)


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing()

    today = date.today().isoformat()
    end_exclusive = (date.today() + pd.Timedelta(days=1)).isoformat()

    if existing.empty:
        print(f"[btc] CSV empty, backfilling {BACKFILL_START} -> {today}")
        fresh = fetch_range(BACKFILL_START, end_exclusive)
    else:
        last_date = existing["date"].max()
        start = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if start > today:
            print(f"[btc] already up to date at {last_date}, refreshing snapshot JSON only")
            fresh = pd.DataFrame(columns=COLUMNS)
        else:
            print(f"[btc] appending {start} -> {today}")
            fresh = fetch_range(start, end_exclusive)

    if fresh.empty and existing.empty:
        print("[btc] yfinance returned no rows and CSV is empty, nothing to write")
        return 1

    if fresh.empty:
        combined = existing
        added = 0
    else:
        combined = fresh if existing.empty else pd.concat([existing, fresh], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        combined.to_csv(CSV_PATH, index=False)
        added = len(combined) - len(existing)

    latest = combined.iloc[-1]
    recent = combined.tail(30).to_dict(orient="records")
    JSON_PATH.write_text(json.dumps({
        "date": latest["date"],
        "open": float(latest["open"]),
        "high": float(latest["high"]),
        "low": float(latest["low"]),
        "close": float(latest["close"]),
        "volume": int(latest["volume"]),
        "rows_total": int(len(combined)),
        "backfill_start": BACKFILL_START,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recent": recent,
    }, indent=2))

    print(f"[btc] rows total = {len(combined)}, +{added} new, latest = {combined['date'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
