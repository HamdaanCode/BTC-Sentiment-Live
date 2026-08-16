import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# model A = baseline: plain VADER, no lexicon changes
INPUT_FILE = 'bitcoin_sept_nov_2021_NEW_CLEANED.csv'
OUTPUT_FILE = 'bitcoin_sept_nov_2021_NEW_MODEL_A.csv'

df = pd.read_csv(INPUT_FILE)
print(f"loaded {len(df):,} tweets from {INPUT_FILE}")

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if pd.isna(text) or len(text) == 0:
        return 0.0
    return analyzer.polarity_scores(text)['compound']

df['sentiment_model_a'] = df['text_cleaned'].apply(get_sentiment)

s = df['sentiment_model_a']
print(f"sentiment: mean {s.mean():+.4f}, std {s.std():.4f}, "
      f"median {s.median():+.4f}, range [{s.min():+.4f}, {s.max():+.4f}]")

positive = (s > 0.05).sum()
neutral = ((s >= -0.05) & (s <= 0.05)).sum()
negative = (s < -0.05).sum()
n = len(df)
print(f"  positive >0.05:  {positive:>7,}  ({positive/n*100:.1f}%)")
print(f"  neutral:         {neutral:>7,}  ({neutral/n*100:.1f}%)")
print(f"  negative <-0.05: {negative:>7,}  ({negative/n*100:.1f}%)")

df_crypto = df[df['crypto_term_count'] > 0]
df_non_crypto = df[df['crypto_term_count'] == 0]
print(f"crypto tweets avg:     {df_crypto['sentiment_model_a'].mean():+.4f}")
print(f"non-crypto tweets avg: {df_non_crypto['sentiment_model_a'].mean():+.4f}")

# spot-check the extremes to make sure VADER is behaving
for label, sample in [('most positive', df.nlargest(3, 'sentiment_model_a')),
                      ('most negative', df.nsmallest(3, 'sentiment_model_a'))]:
    print(f"\n{label}:")
    for _, row in sample.iterrows():
        print(f"  {row['sentiment_model_a']:+.4f}  {row['text_cleaned'][:120]}")

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nwrote {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)/(1024*1024):.1f} MB)")

