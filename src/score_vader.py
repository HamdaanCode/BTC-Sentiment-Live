"""VADER sentiment scoring: Models A, B, C in one pass.

Port of the dissertation's vader_model_{a,b,c}.py scripts, parameterised
and consolidated so any cleaned-tweet CSV (2021, 2022, or future) can be
scored in one command with identical methodology.

Models (same definitions as the dissertation):

  A  baseline VADER, compound score, no lexicon changes
  B  VADER + 35-term hand-scored crypto lexicon (18 bullish / 17 bearish
     on VADER's [-4, 4] scale)
  C  Model B + intensity modifiers:
       - ALL-CAPS: if the ORIGINAL text has >=2 all-caps words, multiply
         the base score by 1.5
       - intensifier words: +0.3 boost
       - exclamations: +0.1 per "!", capped at 3
       - word repetitions (non-stopword, >=2): +0.05 each
     boost is sign-preserving (negative scores get more negative, not less)
     final score is clamped to [-1, 1]

Input CSV must have columns:
  text         raw tweet text (needed for CAPS detection in Model C)
  text_cleaned lowercased/URL-stripped text produced by src.preprocess

Usage:
  python -m src.score_vader --input artifacts/raw/2022_cleaned.csv \\
                            --output artifacts/raw/2022_scored.csv

Adds columns sentiment_model_a, sentiment_model_b, sentiment_model_c.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---- Dissertation's exact crypto lexicon (35 terms) ----
CRYPTO_LEXICON = {
    # 18 bullish
    "hodl": 2.0, "moon": 2.5, "mooning": 2.5, "bullish": 2.0,
    "rally": 1.5, "pump": 1.5, "breakout": 1.5, "accumulate": 0.5,
    "buy the dip": 1.0, "btfd": 1.0, "diamond hands": 2.0, "ath": 2.0,
    "green candle": 1.0, "lambo": 2.0, "to the moon": 2.5, "rocket": 1.5,
    "gains": 1.5, "bull run": 2.0,
    # 17 bearish
    "bearish": -2.0, "dump": -2.0, "crash": -2.5, "rekt": -2.5,
    "fud": -1.5, "shitcoin": -2.0, "scam": -2.5, "ponzi": -2.5,
    "bubble": -1.5, "paper hands": -2.0, "bagholder": -1.5, "red candle": -1.0,
    "capitulation": -2.5, "bear trap": -1.0, "dead cat bounce": -1.5,
    "rugpull": -2.5, "dip": -0.5,
}

# ---- Model C intensity knobs (dissertation values) ----
INTENSIFIERS = {"very", "extremely", "super", "insanely", "absolutely", "incredibly"}
CAPS_MULTIPLIER = 1.5
INTENSIFIER_BOOST = 0.3
EXCLAMATION_BOOST = 0.1  # per "!", capped at 3
REPETITION_BOOST = 0.05
STOPWORDS = {"the", "a", "an", "to", "of", "and", "or", "in", "on", "is", "it", "for"}


def _count_caps_words(text: str) -> int:
    return sum(1 for w in text.split() if len(w) >= 2 and w.isupper())


def _has_intensifier(text_lower: str) -> bool:
    return any(w in text_lower for w in INTENSIFIERS)


def _count_repetitions(text_lower: str) -> int:
    words = [w for w in text_lower.split() if w not in STOPWORDS]
    if len(words) < 2:
        return 0
    return sum(1 for c in Counter(words).values() if c >= 2)


def apply_intensity_modifiers(base_score: float, cleaned_text: str, original_text) -> float:
    """Model C's intensity layer. Matches dissertation exactly:
    sign-preserving boost, caps-multiplier on base, final clamp to [-1, 1].
    """
    if pd.isna(cleaned_text) or len(cleaned_text) == 0:
        return base_score

    score = base_score
    boost = 0.0

    if pd.notna(original_text) and _count_caps_words(original_text) >= 2:
        score = base_score * CAPS_MULTIPLIER
        boost += 0.2

    text_lower = cleaned_text.lower()
    if _has_intensifier(text_lower):
        boost += INTENSIFIER_BOOST

    exclamations = min(cleaned_text.count("!"), 3)
    if exclamations:
        boost += exclamations * EXCLAMATION_BOOST

    reps = _count_repetitions(text_lower)
    if reps:
        boost += reps * REPETITION_BOOST

    # Preserve sign when adding the boost so negatives stay negative.
    if score > 0:
        score += boost
    elif score < 0:
        score -= boost

    return max(-1.0, min(1.0, score))


def _print_distribution(label: str, s: pd.Series) -> None:
    n = len(s)
    pos = (s > 0.05).sum()
    neu = ((s >= -0.05) & (s <= 0.05)).sum()
    neg = (s < -0.05).sum()
    print(
        f"  {label}: mean {s.mean():+.4f} std {s.std():.4f} "
        f"median {s.median():+.4f} range [{s.min():+.4f}, {s.max():+.4f}]"
    )
    print(
        f"       positive {pos:,} ({pos/n*100:.1f}%) | "
        f"neutral {neu:,} ({neu/n*100:.1f}%) | "
        f"negative {neg:,} ({neg/n*100:.1f}%)"
    )


def run(input_path: Path, output_path: Path) -> int:
    df = pd.read_csv(input_path, on_bad_lines="skip", engine="python")
    print(f"loaded {len(df):,} rows from {input_path}")

    if "text_cleaned" not in df.columns or "text" not in df.columns:
        raise SystemExit(
            f"input {input_path} must have columns 'text' and 'text_cleaned'; "
            "run src.preprocess first"
        )

    # Model A: baseline VADER (fresh analyzer so no lexicon leakage).
    print("scoring Model A (baseline VADER)...")
    analyzer_a = SentimentIntensityAnalyzer()

    def score_a(text):
        if pd.isna(text) or len(text) == 0:
            return 0.0
        return analyzer_a.polarity_scores(text)["compound"]

    df["sentiment_model_a"] = df["text_cleaned"].apply(score_a)
    _print_distribution("A", df["sentiment_model_a"])

    # Model B: VADER + crypto lexicon.
    print("scoring Model B (VADER + crypto lexicon)...")
    analyzer_b = SentimentIntensityAnalyzer()
    analyzer_b.lexicon.update(CRYPTO_LEXICON)

    def score_b(text):
        if pd.isna(text) or len(text) == 0:
            return 0.0
        return analyzer_b.polarity_scores(text)["compound"]

    df["sentiment_model_b"] = df["text_cleaned"].apply(score_b)
    _print_distribution("B", df["sentiment_model_b"])

    # Model C: Model B base + intensity modifiers using the ORIGINAL text for CAPS.
    print("scoring Model C (Model B + intensity modifiers)...")

    def score_c(row):
        cleaned = row["text_cleaned"]
        if pd.isna(cleaned) or len(cleaned) == 0:
            return 0.0
        base = analyzer_b.polarity_scores(cleaned)["compound"]
        return apply_intensity_modifiers(base, cleaned, row["text"])

    df["sentiment_model_c"] = df.apply(score_c, axis=1)
    _print_distribution("C", df["sentiment_model_c"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"wrote {len(df):,} rows -> {output_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True,
                   help="cleaned tweet CSV (output of src.preprocess)")
    p.add_argument("--output", type=Path, required=True,
                   help="where to write the scored per-tweet CSV")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args.input, args.output)


if __name__ == "__main__":
    sys.exit(main())
