import os
import re
from collections import Counter

import emoji
import pandas as pd

INPUT_FILE = 'bitcoin_sept_nov_2021_NEW_EXTRACTED.csv'
OUTPUT_FILE = 'bitcoin_sept_nov_2021_NEW_CLEANED.csv'

# 18 bullish / 17 bearish terms used both here (detection) and later in VADER (scoring)
positive_terms = [
    'hodl', 'moon', 'mooning', 'bullish', 'rally', 'pump', 'breakout',
    'accumulate', 'buy the dip', 'btfd', 'diamond hands', 'ath',
    'green candle', 'lambo', 'to the moon', 'rocket', 'gains', 'bull run',
]
negative_terms = [
    'bearish', 'dump', 'crash', 'rekt', 'fud', 'shitcoin', 'scam',
    'ponzi', 'bubble', 'paper hands', 'bagholder', 'red candle',
    'capitulation', 'bear trap', 'dead cat bounce', 'rugpull', 'dip',
]
all_crypto_terms = positive_terms + negative_terms

# multi-word phrases must be checked first; otherwise "moon" would swallow "to the moon"
multi_word_terms = [
    'buy the dip', 'diamond hands', 'green candle', 'to the moon',
    'bull run', 'paper hands', 'red candle', 'bear trap', 'dead cat bounce',
]
single_word_terms = [t for t in all_crypto_terms if t not in multi_word_terms]


def clean_tweet(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\brt\b', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)  # strip # but keep the word
    return re.sub(r'\s+', ' ', text).strip()


def extract_emojis(text):
    if pd.isna(text):
        return []
    return [c for c in text if c in emoji.EMOJI_DATA]


def detect_crypto_terms(text):
    if pd.isna(text):
        return []
    t = text.lower()
    hits = [p for p in multi_word_terms if p in t]
    hits += [w for w in single_word_terms if re.search(r'\b' + re.escape(w) + r'\b', t)]
    return hits


df = pd.read_csv(INPUT_FILE, on_bad_lines='skip', engine='python')
print(f"loaded {len(df):,} tweets")

before = len(df)
df = df.dropna(subset=['datetime', 'text'])
print(f"dropped {before - len(df):,} rows missing datetime/text -> {len(df):,} left")

before = len(df)
df = df.drop_duplicates(subset=['datetime', 'text', 'username'])
print(f"dropped {before - len(df):,} duplicates -> {len(df):,} left")

df['text_cleaned'] = df['text'].apply(clean_tweet)
before = len(df)
df = df[df['text_cleaned'].str.len() > 0]
print(f"dropped {before - len(df):,} tweets that were empty after cleaning -> {len(df):,} left")

df['emojis_detected'] = df['text_cleaned'].apply(extract_emojis)
df['emojis_string'] = df['emojis_detected'].apply(''.join)
df['emoji_count'] = df['emojis_detected'].apply(len)

emoji_tweets = (df['emoji_count'] > 0).sum()
print(f"emojis: {emoji_tweets:,} tweets contain {df['emoji_count'].sum():,} emojis "
      f"({emoji_tweets/len(df)*100:.1f}% of tweets)")

df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
print(f"date range: {df['datetime'].min()} -> {df['datetime'].max()}")

df['crypto_terms'] = df['text_cleaned'].apply(detect_crypto_terms)
df['crypto_term_count'] = df['crypto_terms'].apply(len)
df['crypto_terms_string'] = df['crypto_terms'].apply(', '.join)

crypto_tweets = (df['crypto_term_count'] > 0).sum()
total_hits = df['crypto_term_count'].sum()
print(f"crypto-term hits: {total_hits:,} across {crypto_tweets:,} tweets "
      f"({crypto_tweets/len(df)*100:.1f}%)")

term_counts = Counter(t for terms in df['crypto_terms'] for t in terms)
print("top 15 terms:")
for term, count in term_counts.most_common(15):
    tag = 'pos' if term in positive_terms else 'neg'
    print(f"  {term:<20} {tag}  {count:,}")

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nwrote {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)/(1024*1024):.1f} MB)")
