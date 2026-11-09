"""Score a preprocessed tweet CSV with FinBERT on MPS.

Streams per-tweet probabilities and a VADER-compatible compound score
(`P(pos) - P(neg)`) into a destination CSV. Writes checkpoints every N
batches so a crashed run can resume from the last flushed row.

Usage:
    python -m src.score_finbert \\
        --input artifacts/raw/sept_nov_2021_cleaned.csv \\
        --output artifacts/raw/sept_nov_2021_finbert.csv \\
        --batch-size 64
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.finbert import FinBERTScorer


def run(
    input_path: Path,
    output_path: Path,
    batch_size: int = 64,
    device: str = "mps",
    flush_every: int = 50,
) -> None:
    df = pd.read_csv(input_path, on_bad_lines="skip", engine="python")
    df = df.dropna(subset=["text_cleaned"]).reset_index(drop=True)
    print(f"loaded {len(df):,} rows from {input_path}")

    # Pre-extract columns to lists so the hot loop never touches pandas.iloc,
    # which is O(log n) per lookup and dominates runtime at this scale.
    texts = df["text_cleaned"].tolist()
    datetimes = df["datetime"].tolist()
    dates = df["date"].tolist()
    usernames = df["username"].tolist()
    total = len(df)
    del df  # free memory

    scorer = FinBERTScorer(device=device, batch_size=batch_size)
    print(f"FinBERT loaded on {device}, batch_size={batch_size}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_buf: list[list] = []
    wrote_header = False
    t0 = time.time()
    last_report = t0
    n_written = 0

    def flush():
        nonlocal rows_buf, wrote_header, n_written
        if not rows_buf:
            return
        pd.DataFrame(
            rows_buf,
            columns=[
                "datetime",
                "date",
                "username",
                "sentiment_pos",
                "sentiment_neg",
                "sentiment_neu",
                "sentiment_finbert",
            ],
        ).to_csv(
            output_path,
            mode="w" if not wrote_header else "a",
            header=not wrote_header,
            index=False,
        )
        wrote_header = True
        n_written += len(rows_buf)
        rows_buf = []

    for i, pos, neg, neu, compound in scorer.iter_scores(texts):
        rows_buf.append(
            [datetimes[i], dates[i], usernames[i], pos, neg, neu, compound]
        )
        if len(rows_buf) >= batch_size * flush_every:
            flush()
            now = time.time()
            if now - last_report > 30:
                elapsed = now - t0
                rate = n_written / elapsed
                remaining = (total - n_written) / rate if rate > 0 else 0
                print(
                    f"  {n_written:,}/{total:,} ({n_written/total*100:.1f}%) "
                    f"| {rate:.0f} tw/s | ~{remaining/60:.1f} min remaining",
                    flush=True,
                )
                last_report = now
    flush()

    elapsed = time.time() - t0
    print(
        f"done. {n_written:,} rows -> {output_path} "
        f"in {elapsed/60:.1f} min ({n_written/elapsed:.0f} tw/s)"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--flush-every", type=int, default=50, help="flush every N batches")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(
        args.input,
        args.output,
        batch_size=args.batch_size,
        device=args.device,
        flush_every=args.flush_every,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
