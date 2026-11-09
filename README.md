# btc-sentiment-live

[![E2E tests](https://github.com/HamdaanCode/BTC-Sentiment-Live/actions/workflows/e2e.yml/badge.svg)](https://github.com/HamdaanCode/BTC-Sentiment-Live/actions/workflows/e2e.yml)

An ongoing personal research project on Bitcoin price prediction with sentiment features. The sentiment-method side of this work started as my BSc dissertation at Heriot-Watt (Machine Learning for Bitcoin Price Prediction); the live site at [btc-sentiment-live.vercel.app](https://btc-sentiment-live.vercel.app) extends it into 2026 with a running daily pipeline.

The dissertation trained three LSTMs on daily Bitcoin data plus tweet-derived sentiment across the Sept–Nov 2021 all-time high window. Best model achieved **R² = 0.787, MAE = $1,300, MAPE = 2.16%** — a 5.9% MAE reduction over baseline VADER using a custom 35-term crypto lexicon.

Phase 2 asks two questions in parallel:
1. **Does the pipeline generalise across market regimes?** (same sentiment method, different time period)
2. **Does sentiment from a different community still carry signal in a post-ETF market?** (same sentiment method, different source)

## Research design

VADER with my crypto lexicon stays fixed as the sentiment method across Phase 2 so time-period and source variables stay isolated. A transformer ablation (FinBERT / CryptoBERT) is deferred as future work.

### Experiment A — Time regime generalisation
- **Varies:** time period (2021 Sept–Nov all-time high vs 2022 bear market)
- **Controls:** sentiment method (VADER + crypto lexicon), source (Twitter)
- **Asks:** does the pipeline still work when the market mood flips?
- **Status:** 2022 retrain pipeline ported; running locally next.

### Experiment B — Live 2026 Reddit pipeline
- **Varies:** source (Reddit instead of Twitter), time period (post-ETF era)
- **Controls:** sentiment method (VADER + crypto lexicon)
- **Asks:** does sentiment from a different community still carry signal in a post-ETF market?
- **Status:** daily ingest live via GitHub Actions cron; accumulating.

Note on the source confound: the live Reddit pipeline is framed as a separate experiment from the 2021 Twitter work. Direct time-period comparisons use the 2022 Twitter dataset in Experiment A, which controls for the sentiment-source variable.

## Roadmap

- [x] Reproduce the 2021 baseline (Model B) end-to-end from a clean venv
- [x] Daily ingest cron: BTC OHLCV + r/Bitcoin sentiment, committed to repo
- [x] Live "Today's signals" block on the homepage
- [x] Port the dissertation's VADER scoring into `src/` for the 2022 retrain
- [ ] Run Experiment A: train Model B2 on 2022 bear-market Twitter data
- [ ] Rolling directional accuracy on the live page (needs daily model inference)
- [ ] L5: upvote-weighted Reddit sentiment
- [ ] On-chain metrics ingest (transaction count, active addresses, exchange flow)
- [ ] L1: transformer ablation (FinBERT / CryptoBERT) — deferred future work

## Repo layout

```
app/                         Next.js 15 (App Router, TS, Tailwind) — the site
├─ page.tsx                  Landing + Phase 2 (research design, today's signals, L1–L5)
├─ DissertationCharts.tsx    Recharts client components (2021 results)
└─ dissertationData.ts       Generated from reference/data/*.csv

src/                         Phase 1 + Phase 2 Python
├─ config.py                 Paths, hyperparams, model → feature map (A, B, C, B2…)
├─ preprocess.py             Cleans raw tweet CSVs (lowercases, strips URLs/mentions)
├─ score_vader.py            VADER Models A/B/C in one pass, methodology mirrors
│                            dissertation exactly (crypto lexicon, intensity modifiers)
├─ aggregate_vader.py        Per-tweet scores + daily OHLCV → LSTM training CSV
├─ data.py                   Windowing + temporal train/test split
├─ model.py                  LSTM architecture
└─ train.py                  Train one or all models (A/B/C/B2/…)

scripts/                     Live pipeline + one-off fetchers
├─ daily_btc_ingest.py       GitHub Actions cron: daily BTC price → CSV + JSON
├─ daily_reddit_sentiment.py GitHub Actions cron: r/Bitcoin top 25 → VADER → CSV + JSON
├─ fetch_prices.py           One-off historical OHLCV pull (e.g. for 2022 retrain)
└─ requirements-ingest.txt   Minimal deps for the daily cron runners

data/
└─ daily/                    Append-only CSVs maintained by the cron
    ├─ btc_prices.csv        Daily BTC OHLCV, backfilled from 2024-01-10 (ETF day)
    ├─ reddit_sentiment.csv  Daily r/Bitcoin sentiment aggregates
    ├─ latest_btc.json       Snapshot the site imports at build time
    └─ latest_reddit.json    Same, for Reddit

reference/
├─ dissertation-code/        Original dissertation Python + notebook (frozen)
└─ data/                     Processed CSVs from the dissertation pipeline
```

## Stack

- **Site:** Next.js 15 · TypeScript · Tailwind CSS 4 · Recharts
- **Hosting:** Vercel (auto-deploys on push)
- **Python:** 3.12 · TensorFlow/Keras · VADER (`vaderSentiment`) · pandas · scikit-learn · yfinance · feedparser
- **CI:** GitHub Actions — Playwright E2E on every push; daily ingest cron

## Running the 2022 retrain (Experiment A)

```bash
source .venv/bin/activate

# 1. Clean the 2022 raw tweets (paths relative to your local setup)
python -m src.preprocess \
  --input  "/path/to/bitcoin-tweets-2022.csv" \
  --output artifacts/raw/2022_cleaned.csv

# 2. Score Models A, B, C with VADER (+ crypto lexicon + intensity modifiers)
python -m src.score_vader \
  --input  artifacts/raw/2022_cleaned.csv \
  --output artifacts/raw/2022_scored.csv

# 3. Pull 2022 BTC OHLCV (one-off historical fetch)
python scripts/fetch_prices.py \
  --start 2022-01-01 --end 2023-01-01 \
  --output artifacts/raw/btc_prices_2022.csv

# 4. Join per-tweet scores + daily prices → LSTM training CSV
python -m src.aggregate_vader \
  --scored-tweets artifacts/raw/2022_scored.csv \
  --prices        artifacts/raw/btc_prices_2022.csv \
  --output        reference/data/NEW_lstm_training_data_2022.csv

# 5. Train Model B2 (VADER + crypto lexicon on 2022 data)
python -m src.train --model B2 --seed 42
```

## Local dev

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Testing & QA

End-to-end coverage runs on every push and PR across **Chromium, Firefox, and WebKit** via GitHub Actions. See the badge above for the latest status.

```bash
npm run test:e2e            # headless run against a local dev server
npm run test:e2e:ui         # interactive Playwright UI mode
npm run test:e2e:report     # open the last HTML report
```

Point the suite at a deployed URL:

```bash
PLAYWRIGHT_BASE_URL=https://btc-sentiment-live.vercel.app npm run test:e2e
```

CI uploads the full HTML report as a workflow artifact on every run, plus traces on failure.

## Background

- Related repo (frozen dissertation code): [Bitcoin-LSTM-code-and-pipeline](https://github.com/HamdaanCode/Bitcoin-LSTM-code-and-pipeline)
