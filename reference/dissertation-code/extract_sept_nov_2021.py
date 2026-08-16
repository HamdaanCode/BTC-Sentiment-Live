
import pandas as pd
from datetime import datetime
import os

# input/output paths for the extraction step
INPUT_FILE = 'bitcoin-tweets-2021.csv'
OUTPUT_FILE = 'bitcoin_sept_nov_2021_NEW_EXTRACTED.csv'

# bounds for the date filter, kept UTC-aware so they compare cleanly with the tweet timestamps
START_DATE = pd.Timestamp('2021-09-01', tz='UTC')
END_DATE = pd.Timestamp('2021-12-01', tz='UTC')  # exclusive, so the window ends at Nov 30

print(f"Input: {INPUT_FILE} -> {OUTPUT_FILE}")
print(f"Window: {START_DATE.date()} to {(END_DATE - pd.Timedelta(days=1)).date()}")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"{INPUT_FILE} not found in cwd")

# the source file has roughly 15.6M rows so we read it in chunks to stay within memory
chunk_size = 500000
chunks_filtered = []
total_rows = 0
filtered_rows = 0

for i, chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=chunk_size,
                                       on_bad_lines='skip',
                                       engine='python',
                                       encoding='utf-8',
                                       quoting=1), 1):  # quoting=1 is QUOTE_ALL
    total_rows += len(chunk)

    # parse the timestamp, coercing malformed values to NaT so they can be dropped
    chunk['datetime'] = pd.to_datetime(chunk['datetime'], errors='coerce', utc=True)

    chunk = chunk.dropna(subset=['datetime'])

    # keep only the rows that fall inside our Sept-Nov window
    filtered_chunk = chunk[(chunk['datetime'] >= START_DATE) &
                           (chunk['datetime'] < END_DATE)]

    filtered_rows += len(filtered_chunk)

    if len(filtered_chunk) > 0:
        chunks_filtered.append(filtered_chunk)

    print(f"  chunk {i}: {total_rows:,} read, {filtered_rows:,} kept "
          f"({filtered_rows/total_rows*100:.1f}%)")

df = pd.concat(chunks_filtered, ignore_index=True)
print(f"\nextracted {len(df):,} of {total_rows:,} rows")
print(f"first tweet: {df['datetime'].min()}  last tweet: {df['datetime'].max()}")

# quick null audit so we know which columns have gaps
missing = df.isnull().sum()
if missing.any():
    print("missing values per column:")
    for col, count in missing[missing > 0].items():
        print(f"  {col}: {count:,}")

# monthly counts for sanity-checking coverage
df['month'] = df['datetime'].dt.to_period('M')
for month, count in df.groupby('month').size().items():
    print(f"  {month.strftime('%B %Y')}: {count:,} tweets")

df = df.drop('month', axis=1)
df.to_csv(OUTPUT_FILE, index=False)
print(f"wrote {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)/(1024*1024):.1f} MB)")

