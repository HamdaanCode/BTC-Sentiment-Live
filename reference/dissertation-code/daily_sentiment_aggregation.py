
import pandas as pd
import numpy as np

INPUT_FILE = 'bitcoin_sept_nov_2021_NEW_MODEL_C.csv'
OUTPUT_FILE = 'daily_sentiment_aggregated_NEW.csv'

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} tweets")
print(f"  Columns: {len(df.columns)}")

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
df['date'] = df['datetime'].dt.date

print(f"\nDate range:")
print(f"  First tweet: {df['datetime'].min()}")
print(f"  Last tweet: {df['datetime'].max()}")
print(f"  Total days: {df['date'].nunique()}")

# Define sentiment thresholds
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

# Group by date and calculate statistics
daily_stats = df.groupby('date').agg(
    # Basic stats
    tweet_count=('text', 'count'),

    # Model A statistics
    sentiment_a_mean=('sentiment_model_a', 'mean'),
    sentiment_a_std=('sentiment_model_a', 'std'),
    sentiment_a_median=('sentiment_model_a', 'median'),
    sentiment_a_min=('sentiment_model_a', 'min'),
    sentiment_a_max=('sentiment_model_a', 'max'),

    # Model B statistics
    sentiment_b_mean=('sentiment_model_b', 'mean'),
    sentiment_b_std=('sentiment_model_b', 'std'),
    sentiment_b_median=('sentiment_model_b', 'median'),
    sentiment_b_min=('sentiment_model_b', 'min'),
    sentiment_b_max=('sentiment_model_b', 'max'),

    # Model C statistics
    sentiment_c_mean=('sentiment_model_c', 'mean'),
    sentiment_c_std=('sentiment_model_c', 'std'),
    sentiment_c_median=('sentiment_model_c', 'median'),
    sentiment_c_min=('sentiment_model_c', 'min'),
    sentiment_c_max=('sentiment_model_c', 'max'),
).reset_index()

# Calculate percentage distributions for each model
for date in daily_stats['date']:
    day_tweets = df[df['date'] == date]

    # Model A percentages
    pct_pos_a = (day_tweets['sentiment_model_a'] > POSITIVE_THRESHOLD).sum() / len(day_tweets) * 100
    pct_neu_a = ((day_tweets['sentiment_model_a'] >= NEGATIVE_THRESHOLD) &
                 (day_tweets['sentiment_model_a'] <= POSITIVE_THRESHOLD)).sum() / len(day_tweets) * 100
    pct_neg_a = (day_tweets['sentiment_model_a'] < NEGATIVE_THRESHOLD).sum() / len(day_tweets) * 100

    # Model B percentages
    pct_pos_b = (day_tweets['sentiment_model_b'] > POSITIVE_THRESHOLD).sum() / len(day_tweets) * 100
    pct_neu_b = ((day_tweets['sentiment_model_b'] >= NEGATIVE_THRESHOLD) &
                 (day_tweets['sentiment_model_b'] <= POSITIVE_THRESHOLD)).sum() / len(day_tweets) * 100
    pct_neg_b = (day_tweets['sentiment_model_b'] < NEGATIVE_THRESHOLD).sum() / len(day_tweets) * 100

    # Model C percentages
    pct_pos_c = (day_tweets['sentiment_model_c'] > POSITIVE_THRESHOLD).sum() / len(day_tweets) * 100
    pct_neu_c = ((day_tweets['sentiment_model_c'] >= NEGATIVE_THRESHOLD) &
                 (day_tweets['sentiment_model_c'] <= POSITIVE_THRESHOLD)).sum() / len(day_tweets) * 100
    pct_neg_c = (day_tweets['sentiment_model_c'] < NEGATIVE_THRESHOLD).sum() / len(day_tweets) * 100

    # Add to dataframe
    daily_stats.loc[daily_stats['date'] == date, 'pct_positive_a'] = pct_pos_a
    daily_stats.loc[daily_stats['date'] == date, 'pct_neutral_a'] = pct_neu_a
    daily_stats.loc[daily_stats['date'] == date, 'pct_negative_a'] = pct_neg_a

    daily_stats.loc[daily_stats['date'] == date, 'pct_positive_b'] = pct_pos_b
    daily_stats.loc[daily_stats['date'] == date, 'pct_neutral_b'] = pct_neu_b
    daily_stats.loc[daily_stats['date'] == date, 'pct_negative_b'] = pct_neg_b

    daily_stats.loc[daily_stats['date'] == date, 'pct_positive_c'] = pct_pos_c
    daily_stats.loc[daily_stats['date'] == date, 'pct_neutral_c'] = pct_neu_c
    daily_stats.loc[daily_stats['date'] == date, 'pct_negative_c'] = pct_neg_c

print(f"\nDataset summary:")
print(f"  Total days: {len(daily_stats)}")
print(f"  Total tweets: {daily_stats['tweet_count'].sum():,}")
print(f"  Avg tweets/day: {daily_stats['tweet_count'].mean():,.0f}")
print(f"  Min tweets/day: {daily_stats['tweet_count'].min():,}")
print(f"  Max tweets/day: {daily_stats['tweet_count'].max():,}")

print(f"\nOverall sentiment means:")
print(f"  Model A: {daily_stats['sentiment_a_mean'].mean():+.4f}")
print(f"  Model B: {daily_stats['sentiment_b_mean'].mean():+.4f}")
print(f"  Model C: {daily_stats['sentiment_c_mean'].mean():+.4f}")

# Monthly breakdown
daily_stats['month'] = pd.to_datetime(daily_stats['date']).dt.to_period('M')

print(f"\nMonthly breakdown:")
for month in daily_stats['month'].unique():
    month_data = daily_stats[daily_stats['month'] == month]
    print(f"\n{month.strftime('%B %Y')}:")
    print(f"  Days: {len(month_data)}")
    print(f"  Total tweets: {month_data['tweet_count'].sum():,}")
    print(f"  Avg tweets/day: {month_data['tweet_count'].mean():,.0f}")
    print(f"  Model A mean: {month_data['sentiment_a_mean'].mean():+.4f}")
    print(f"  Model B mean: {month_data['sentiment_b_mean'].mean():+.4f}")
    print(f"  Model C mean: {month_data['sentiment_c_mean'].mean():+.4f}")

display_df = daily_stats.drop('month', axis=1)
display_cols = ['date', 'tweet_count', 'sentiment_a_mean', 'sentiment_b_mean', 'sentiment_c_mean']
print("\nFirst 10 days:\n" + display_df[display_cols].head(10).to_string(index=False))
print("\nLast 10 days:\n" + display_df[display_cols].tail(10).to_string(index=False))

import os
daily_stats_to_save = daily_stats.drop('month', axis=1)
daily_stats_to_save.to_csv(OUTPUT_FILE, index=False)
print(f"\nwrote {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)/1024:.1f} KB)")

