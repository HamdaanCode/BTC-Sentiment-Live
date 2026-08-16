# btc-sentiment-live

Live extension of my BSc dissertation on **Machine Learning for Bitcoin Price Prediction** (Heriot-Watt, First Class, 84%).

The dissertation trained three LSTM models on daily Bitcoin data plus tweet-derived sentiment across the Sept–Nov 2021 all-time high window. Best model achieved **R² = 0.787, MAE = $1,300, MAPE = 2.16%** — a 5.9% MAE reduction over baseline VADER using a custom crypto lexicon.

This repo takes that work live and asks the harder question: **how does a 2021-trained sentiment model actually hold up on 2026 markets?**

## Roadmap

1. **Retrain the 2021 baseline** — reproduce the winning model (Model B, crypto-lexicon VADER + LSTM) locally, save weights.
2. **Live data pipeline** — daily BTC OHLCV + Reddit sentiment via GitHub Actions cron.
3. **Ship v0 predictions publicly** — accuracy log with every prediction vs. actual close, updated daily.
4. **Train a fresh 2026 model** — once enough live data is collected, train alongside the 2021 baseline for a direct A/B.
5. **Transformer sentiment** — replace VADER with a fine-tuned RoBERTa-style model. Compare against both LSTM baselines.

## Repo layout

```
app/                       Next.js 15 (App Router, TS, Tailwind) — the site
├─ page.tsx                Landing page: pitch, results, roadmap
├─ DissertationCharts.tsx  Recharts client components
└─ dissertationData.ts     Generated from reference/data/*.csv
reference/
├─ dissertation-code/      Original dissertation Python scripts + notebook
└─ data/                   Processed CSVs from the dissertation pipeline
```

## Stack

- **Site:** Next.js 15 · TypeScript · Tailwind CSS 4 · Recharts
- **Hosting:** Vercel
- **Original dissertation code:** Python 3, TensorFlow/Keras, VADER, pandas, scikit-learn
- **Coming for Phase 2:** Python + FastAPI, GitHub Actions cron, PostgreSQL or SQLite, HuggingFace transformers

## Local dev

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Background

- Dissertation title: *Machine Learning for Bitcoin Price Prediction — LSTM models with sentiment features*
- Institution: Heriot-Watt University
- Related repo (frozen dissertation code): [Bitcoin-LSTM-code-and-pipeline](https://github.com/HamdaanCode/Bitcoin-LSTM-code-and-pipeline)
