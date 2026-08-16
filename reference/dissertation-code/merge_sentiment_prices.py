
import pandas as pd
import numpy as np

SENTIMENT_FILE = 'daily_sentiment_aggregated_NEW.csv'
PRICE_FILE = 'bitcoin_prices_sept_nov_2021_NEW.csv'
OUTPUT_FILE = 'lstm_training_data_NEW.csv'

df_sentiment = pd.read_csv(SENTIMENT_FILE)
df_sentiment['date'] = pd.to_datetime(df_sentiment['date']).dt.date

df_prices = pd.read_csv(PRICE_FILE)
df_prices['date'] = pd.to_datetime(df_prices['date']).dt.date

df = pd.merge(df_sentiment, df_prices, on='date', how='outer')
print(f"merged: {len(df)} rows")

missing_sentiment = df['sentiment_a_mean'].isna().sum()
missing_price = df['close'].isna().sum()
if missing_sentiment:
    print(f"warning: {missing_sentiment} days missing sentiment")
if missing_price:
    print(f"warning: {missing_price} days missing price")

df = df.sort_values('date').reset_index(drop=True)

# derived features: raw + % price change, volume change, 7d MA, 7d vol, momentum vs MA7
df['price_change'] = df['close'].diff()
df['price_change_pct'] = df['close'].pct_change() * 100
df['volume_change_pct'] = df['volume'].pct_change() * 100
df['price_ma_7'] = df['close'].rolling(window=7, min_periods=1).mean()
df['price_volatility_7'] = df['close'].rolling(window=7, min_periods=1).std()
df['price_momentum'] = df['close'] - df['price_ma_7']

# STATISTICS
print(f"\nDataset summary:")
print(f"  Total days: {len(df)}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print(f"  Total columns: {len(df.columns)}")

print(f"\nBitcoin price statistics:")
print(f"  Mean close: ${df['close'].mean():,.2f}")
print(f"  Min close:  ${df['close'].min():,.2f}")
print(f"  Max close:  ${df['close'].max():,.2f}")
print(f"  Std:        ${df['close'].std():,.2f}")

print(f"\nSentiment statistics (all 3 models):")
print(f"  Model A mean: {df['sentiment_a_mean'].mean():+.4f}")
print(f"  Model B mean: {df['sentiment_b_mean'].mean():+.4f}")
print(f"  Model C mean: {df['sentiment_c_mean'].mean():+.4f}")

print(f"\nDerived features statistics:")
print(f"  Avg daily price change: {df['price_change_pct'].mean():+.2f}%")
print(f"  Avg volume change: {df['volume_change_pct'].mean():+.2f}%")
print(f"  Avg 7-day volatility: ${df['price_volatility_7'].mean():,.2f}")

print(f"\nPearson correlation with Bitcoin close price:")
corr_a = df['sentiment_a_mean'].corr(df['close'])
corr_b = df['sentiment_b_mean'].corr(df['close'])
corr_c = df['sentiment_c_mean'].corr(df['close'])

print(f"  Model A: {corr_a:+.4f}")
print(f"  Model B: {corr_b:+.4f}")
print(f"  Model C: {corr_c:+.4f}")

# Next day price prediction correlation
df['next_day_price'] = df['close'].shift(-1)
df['next_day_price_change_pct'] = ((df['next_day_price'] - df['close']) / df['close'] * 100)

corr_a_next = df['sentiment_a_mean'].corr(df['next_day_price_change_pct'])
corr_b_next = df['sentiment_b_mean'].corr(df['next_day_price_change_pct'])
corr_c_next = df['sentiment_c_mean'].corr(df['next_day_price_change_pct'])

print(f"\nCorrelation with NEXT DAY price change %:")
print(f"  Model A: {corr_a_next:+.4f}")
print(f"  Model B: {corr_b_next:+.4f}")
print(f"  Model C: {corr_c_next:+.4f}")

display_cols = ['date', 'close', 'sentiment_a_mean', 'sentiment_b_mean',
                'sentiment_c_mean', 'price_change_pct', 'price_ma_7']
print("\nFirst 10 days:\n" + df[display_cols].head(10).to_string(index=False))
print("\nLast 10 days:\n" + df[display_cols].tail(10).to_string(index=False))

import os
df_to_save = df.drop(['next_day_price', 'next_day_price_change_pct'], axis=1, errors='ignore')
df_to_save.to_csv(OUTPUT_FILE, index=False)
print(f"\nwrote {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)/1024:.1f} KB, "
      f"{len(df_to_save)} rows x {len(df_to_save.columns)} cols)")

