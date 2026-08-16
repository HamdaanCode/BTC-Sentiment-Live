import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

INPUT_FILE = 'bitcoin_sept_nov_2021_NEW_MODEL_A.csv'
OUTPUT_FILE = 'bitcoin_sept_nov_2021_NEW_MODEL_B.csv'

# 18 bullish / 17 bearish crypto-twitter terms, scored by hand on VADER's [-4, 4] scale
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

df = pd.read_csv(INPUT_FILE)
print(f"loaded {len(df):,} tweets (already scored by model A)")

analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(crypto_lexicon)

def get_sentiment(text):
    if pd.isna(text) or len(text) == 0:
        return 0.0
    return analyzer.polarity_scores(text)['compound']

df['sentiment_model_b'] = df['text_cleaned'].apply(get_sentiment)

s = df['sentiment_model_b']
print(f"model B: mean {s.mean():+.4f}, std {s.std():.4f}, "
      f"median {s.median():+.4f}, range [{s.min():+.4f}, {s.max():+.4f}]")

n = len(df)
pos = (s > 0.05).sum()
neu = ((s >= -0.05) & (s <= 0.05)).sum()
neg = (s < -0.05).sum()
print(f"  positive: {pos:,} ({pos/n*100:.1f}%) | neutral: {neu:,} ({neu/n*100:.1f}%) | "
      f"negative: {neg:,} ({neg/n*100:.1f}%)")

df_crypto = df[df['crypto_term_count'] > 0]
df_non_crypto = df[df['crypto_term_count'] == 0]
print(f"crypto tweets:     A {df_crypto['sentiment_model_a'].mean():+.4f}  "
      f"B {df_crypto['sentiment_model_b'].mean():+.4f}")
print(f"non-crypto tweets: A {df_non_crypto['sentiment_model_a'].mean():+.4f}  "
      f"B {df_non_crypto['sentiment_model_b'].mean():+.4f}")

# which tweets moved the most between A and B? gives a quick smell-test of the lexicon
df_crypto_diff = df_crypto.assign(
    sentiment_diff=df_crypto['sentiment_model_b'] - df_crypto['sentiment_model_a']
)
top_improvements = df_crypto_diff.nlargest(3, 'sentiment_diff')
print("\nbiggest A->B positive shifts (crypto tweets only):")
for _, row in top_improvements.iterrows():
    print(f"  A {row['sentiment_model_a']:+.4f} -> B {row['sentiment_model_b']:+.4f}  "
          f"terms={row['crypto_terms_string']}")
    print(f"    {row['text_cleaned'][:110]}")

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nwrote {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)/(1024*1024):.1f} MB)")
