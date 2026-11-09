"""Extract a date-filtered tweet window from a large raw CSV.

Reads the Kaggle Bitcoin Tweets CSV in chunks (memory-safe for 3GB+ files),
filters by `[start_date, end_date)` on the `datetime` column (UTC), and writes
the filtered rows to a destination CSV.

Usage:
    python -m src.ingest \\
        --input "/path/to/bitcoin-tweets-2021.csv" \\
        --output artifacts/raw/sept_nov_2021.csv \\
        --start 2021-09-01 --end 2021-12-01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def extract(
    input_path: Path,
    output_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_size: int = 500_000,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    first_chunk = True
    for i, chunk in enumerate(
        pd.read_csv(
            input_path,
            chunksize=chunk_size,
            on_bad_lines="skip",
            engine="python",
            encoding="utf-8",
            quoting=1,
        ),
        start=1,
    ):
        total_in += len(chunk)
        chunk["datetime"] = pd.to_datetime(chunk["datetime"], errors="coerce", utc=True)
        chunk = chunk.dropna(subset=["datetime", "text"])
        mask = (chunk["datetime"] >= start) & (chunk["datetime"] < end)
        kept = chunk.loc[mask, ["datetime", "username", "text"]]
        total_out += len(kept)

        if len(kept):
            kept.to_csv(
                output_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )
            first_chunk = False
        pct = (total_out / total_in * 100) if total_in else 0
        print(
            f"  chunk {i}: processed {total_in:,} rows, kept {total_out:,} ({pct:.2f}%)",
            flush=True,
        )
    return total_out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD exclusive")
    p.add_argument("--chunk-size", type=int, default=500_000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Window: {start.date()} to {(end - pd.Timedelta(days=1)).date()}")
    print()

    kept = extract(args.input, args.output, start, end, chunk_size=args.chunk_size)
    print()
    print(f"Done. Extracted {kept:,} rows to {args.output}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
