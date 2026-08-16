
import os
import pandas as pd
import yfinance as yf

TICKER = 'BTC-USD'
START_DATE = '2021-09-01'
END_DATE = '2021-12-01'  # exclusive, so window goes up to Nov 30
OUTPUT_FILE = 'bitcoin_prices_sept_nov_2021_NEW.csv'

btc = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
if btc.empty:
    raise RuntimeError(f"yfinance returned no rows for {TICKER} {START_DATE}..{END_DATE}")

btc.reset_index(inplace=True)

# yfinance sometimes returns a MultiIndex on columns, flatten it
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = [col[0] if isinstance(col, tuple) else col for col in btc.columns]

btc.columns = [str(c).lower() for c in btc.columns]
btc = btc[['date', 'open', 'high', 'low', 'close', 'volume']]
btc['date'] = pd.to_datetime(btc['date']).dt.date

print(f"{TICKER}: {len(btc)} days, {btc['date'].min()} -> {btc['date'].max()}")
print(f"close: min ${btc['close'].min():,.2f}, max ${btc['close'].max():,.2f}, "
      f"mean ${btc['close'].mean():,.2f}")

# highest close in the window
ath_row = btc.loc[btc['close'].idxmax()]
print(f"ATH in window: {ath_row['date']} @ ${ath_row['close']:,.2f}")

# month-by-month averages as a sanity check
btc['month'] = pd.to_datetime(btc['date']).dt.to_period('M')
for month, month_data in btc.groupby('month'):
    first, last = month_data['close'].iloc[0], month_data['close'].iloc[-1]
    pct = (last - first) / first * 100
    print(f"  {month.strftime('%B %Y')}: {len(month_data)}d, "
          f"avg ${month_data['close'].mean():,.2f}, change {pct:+.2f}%")

btc.drop('month', axis=1).to_csv(OUTPUT_FILE, index=False)
print(f"wrote {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)/1024:.1f} KB)")

