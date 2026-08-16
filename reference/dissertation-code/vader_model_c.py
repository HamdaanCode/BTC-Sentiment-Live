import os
from collections import Counter
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

INPUT_FILE = 'bitcoin_sept_nov_2021_NEW_MODEL_B.csv'
OUTPUT_FILE = 'bitcoin_sept_nov_2021_NEW_MODEL_C.csv'

# same crypto lexicon as model B — compact form here for readability
crypto_lexicon = {
    'hodl': 2.0, 'moon': 2.5, 'mooning': 2.5, 'bullish': 2.0,
    'rally': 1.5, 'pump': 1.5, 'breakout': 1.5, 'accumulate': 0.5,
    'buy the dip': 1.0, 'btfd': 1.0, 'diamond hands': 2.0, 'ath': 2.0,
    'green candle': 1.0, 'lambo': 2.0, 'to the moon': 2.5, 'rocket': 1.5,
    'gains': 1.5, 'bull run': 2.0,
    'bearish': -2.0, 'dump': -2.0, 'crash': -2.5, 'rekt': -2.5,
    'fud': -1.5, 'shitcoin': -2.0, 'scam': -2.5, 'ponzi': -2.5,
    'bubble': -1.5, 'paper hands': -2.0, 'bagholder': -1.5, 'red candle': -1.0,
    'capitulation': -2.5, 'bear trap': -1.0, 'dead cat bounce': -1.5,
    'rugpull': -2.5, 'dip': -0.5,
}

# intensity knobs: ALL-CAPS, intensifier words, "!", and word repetition
INTENSIFIERS = {'very', 'extremely', 'super', 'insanely', 'absolutely', 'incredibly'}
CAPS_MULTIPLIER = 1.5
INTENSIFIER_BOOST = 0.3
EXCLAMATION_BOOST = 0.1  # per "!", capped at 3
REPETITION_BOOST = 0.05
STOPWORDS = {'the', 'a', 'an', 'to', 'of', 'and', 'or', 'in', 'on', 'is', 'it', 'for'}


def count_caps_words(text):
    return sum(1 for w in text.split() if len(w) >= 2 and w.isupper())


def has_intensifier(text_lower):
    return any(w in text_lower for w in INTENSIFIERS)


def count_repetitions(text_lower):
    words = [w for w in text_lower.split() if w not in STOPWORDS]
    if len(words) < 2:
        return 0
    return sum(1 for c in Counter(words).values() if c >= 2)


def apply_intensity_modifiers(base_score, text, original_text):
    if pd.isna(text) or len(text) == 0:
        return base_score

    score = base_score
    boost = 0.0

    if pd.notna(original_text) and count_caps_words(original_text) >= 2:
        score = base_score * CAPS_MULTIPLIER
        boost += 0.2

    text_lower = text.lower()
    if has_intensifier(text_lower):
        boost += INTENSIFIER_BOOST

    exclamations = min(text.count('!'), 3)
    if exclamations:
        boost += exclamations * EXCLAMATION_BOOST

    reps = count_repetitions(text_lower)
    if reps:
        boost += reps * REPETITION_BOOST

    # preserve sign when adding the boost so negatives stay negative
    if score > 0:
        score += boost
    elif score < 0:
        score -= boost

    return max(-1.0, min(1.0, score))


df = pd.read_csv(INPUT_FILE)
print(f"loaded {len(df):,} tweets (already scored by models A & B)")

analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(crypto_lexicon)


def score_row(row):
    text = row['text_cleaned']
    if pd.isna(text) or len(text) == 0:
        return 0.0
    base = analyzer.polarity_scores(text)['compound']
    return apply_intensity_modifiers(base, text, row['text'])


df['sentiment_model_c'] = df.apply(score_row, axis=1)

s = df['sentiment_model_c']
print(f"model C: mean {s.mean():+.4f}, std {s.std():.4f}, "
      f"median {s.median():+.4f}, range [{s.min():+.4f}, {s.max():+.4f}]")

mean_a = df['sentiment_model_a'].mean()
mean_b = df['sentiment_model_b'].mean()
mean_c = s.mean()
print(f"overall means: A {mean_a:+.4f} | B {mean_b:+.4f} "
      f"({(mean_b-mean_a)/abs(mean_a)*100:+.1f}%) | "
      f"C {mean_c:+.4f} ({(mean_c-mean_a)/abs(mean_a)*100:+.1f}%)")

df_crypto = df[df['crypto_term_count'] > 0]
ca, cb, cc = (df_crypto[f'sentiment_model_{m}'].mean() for m in ('a', 'b', 'c'))
print(f"crypto-tweet means: A {ca:+.4f} | B {cb:+.4f} | C {cc:+.4f}")

# tweets where the intensity modifiers moved the score most vs model B
diff = df['sentiment_model_c'] - df['sentiment_model_b']
boosted = df.assign(_d=diff).loc[diff > 0.1].nlargest(3, '_d')
print("\ntop intensity boosts (model C > model B):")
for _, row in boosted.iterrows():
    print(f"  B {row['sentiment_model_b']:+.4f} -> C {row['sentiment_model_c']:+.4f}")
    print(f"    {row['text_cleaned'][:120]}")

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nwrote {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)/(1024*1024):.1f} MB)")
