# Project Progress

Rolling status doc. Updated as work is completed. Newest at top of each section.

Last updated: 2026-11-11 (L2 closed with H1 2022 experimental results published to site)

## Environment

- pyenv 2.8.4 installed to `~/.pyenv`; init lines added to `~/.zshrc`
- Python 3.12.14 compiled via pyenv, pinned in this repo via `.python-version`
- `.venv/` created at repo root, pip 26.2.1
- Caveat: pyenv build warned `_lzma` module missing (no `xz` at build time). Not a blocker for training; only matters for `.xz` files.

To activate:
```
cd ~/Documents/btc-sentiment
source .venv/bin/activate
```

## Roadmap

Anchored to the dissertation's Phase 2 goals and the L1–L5 limitations named in `hwu_cs_dissertation_2025_H00350285.docx`.

### Phase 2 — reproducible retraining pipeline
- [x] **Env setup** — pyenv + Py3.12 + .venv (done 2026-08-17)
- [x] **3a — Skeleton + requirements + data module + seeds** (done 2026-08-17)
  - `requirements.txt` — TF 2.17.0, numpy 1.26.4, pandas 2.3.3, sklearn 1.9.0, scipy 1.17.1, matplotlib 3.11.1 (all installed clean into `.venv`)
  - `src/config.py` — paths, hyperparams, `set_global_seed()` (seeds python/numpy/tf, `TF_DETERMINISTIC_OPS=1`)
  - `src/data.py` — `load_scaled()`, `create_windows()`, `temporal_split()`, `prepare(model_key)`, `inverse_price()`
  - Smoke test: `prepare('A')` yields X_train (61, 3, 5) / X_test (27, 3, 5), matches dissertation shapes
- [x] **3b — Model + training + weight persistence** (done 2026-08-17)
  - `src/model.py` — `build_lstm()` (Input → LSTM 32 → Dropout → LSTM 16 → Dropout → Dense), Keras 3 clean
  - `src/train.py` — CLI (`--model A|B|C|all`, `--seed`, `--epochs`, `--batch-size`, `--patience`)
  - Artifacts per run → `artifacts/{model}_seed{seed}/`: `model.keras`, `scaler.pkl`, `metrics.json`, `history.json`
  - Summary → `artifacts/summary_seed{seed}.json`
  - Verified reproducibility: two seed=42 runs of model A both give `best_val_loss=0.004391`
  - Baseline seed=42 numbers: A best_val_loss=0.004391, B=0.004625, C=0.004646 (all three ~equivalent, matching dissertation's finding)
- [x] **3c — Evaluate + plots** (done 2026-08-17)
  - `src/evaluate.py` — loads `model.keras` + `scaler.pkl`, computes RMSE / MAE / MAPE / R^2 / directional accuracy, paired t-tests, writes `eval.json` per model and `comparison_seed{seed}.json`
  - `src/plots.py` — 4-panel figure (actual vs pred, loss curves, R^2 bars, error boxplots) → `artifacts/plots/comparison_seed{seed}.png`
- [x] **VADER scoring + aggregation ported into `src/`** (done 2026-11-09)
  - `src/score_vader.py` — Models A/B/C in one pass, methodology mirrors dissertation exactly (35-term crypto lexicon + intensity modifiers with sign-preserving boost, clamp to [-1, 1])
  - `src/aggregate_vader.py` — joins per-tweet scores with daily OHLCV, outputs the 37-column training CSV schema the LSTM pipeline consumes
  - `scripts/fetch_prices.py` — one-off historical OHLCV fetcher for arbitrary date windows
  - `src/config.py` — Model B2 wired (same features as B, trained on 2022 CSV via TRAINING_CSV_OVERRIDES)
  - Smoke-tested end-to-end on a 5-tweet sample; outputs match dissertation's 37-column schema

### Reproduced dissertation baseline (seed=42)
This is the benchmark the 2022 extension (Model B2) will be compared against.

| Metric | Model A | Model B | Model C |
|---|---|---|---|
| RMSE ($) | 1780.81 | 1827.50 | 1831.64 |
| MAE ($) | 1293.43 | 1300.39 | 1316.21 |
| MAPE (%) | 2.15 | 2.15 | 2.19 |
| R^2 | 0.7817 | 0.7701 | 0.7691 |
| Dir. Accuracy (%) | 34.62 | 38.46 | 38.46 |

Paired t-tests on |errors|: A vs B p=0.88, A vs C p=0.42, B vs C p=0.73 — all NS. Confirms dissertation finding that the three VADER variants are statistically indistinguishable on this dataset. **Directional accuracy below 50%** is L4 reproduced.

### Phase 2 — dissertation limitation closures
Mapping each named limitation to the deliverable that closes it.
- [x] **L2 (single window Sept–Nov 2021)** → Closed 2026-11-11. Model B2 retrained on **H1 2022 (Jan 1 – Jun 30, 181 days, 7.3M tweets)**. Train/test 70/30 → 127/54 (test set 2x the dissertation's). Results on 54-day test set: **R² 0.018, MAE $3,670, MAPE 15.7%, Directional accuracy 60.4%**. Interpretation: pipeline generalises weakly on magnitude under regime change (R² drops from 0.77 → 0.018) but retains directional signal above random. Published on site under "Experiment A results". Full reproducible record: Colab notebook committed as `Bitcoin_LSTM_2022_Dataset.ipynb`; training CSV at `reference/data/NEW_lstm_training_data_2022.csv`.
- [x] **L3 (Twitter-only source)** → **Reddit added as second live sentiment source** (r/Bitcoin top 25 daily, VADER-scored, same methodology as the dissertation's Twitter pipeline). On-chain metrics moved to README future-work.
- [ ] **L5 (unweighted aggregation)** → weight Reddit posts by upvotes (small change to `daily_reddit_sentiment.py`); Twitter followers if re-added

### Moved to future work (README)
Documented limitations addressed in backlog rather than active sprint. The homepage shows only shipped or in-progress items so visitors don't read planned items as unshipped promises; these still live in README so the honest trail is preserved.
- **L1** — transformer sentiment ablation (FinBERT / CryptoBERT). Deliberately kept out of Phase 2 so time-period and source variables can be isolated.
- **L3 extended** — on-chain metrics ingest (blockchain.com public APIs for txn count, active addresses, exchange flow). Reddit already closes L3 for sentiment-source diversity; on-chain adds a non-sentiment signal family.
- **L4** — rolling directional accuracy on the live page. Requires daily model inference in the cron; compound work with L1 and on-chain, so queued alongside.

### Phase 3 — live inference & dashboard
- [x] **Daily ingest cron** — `scripts/daily_btc_ingest.py` + `scripts/daily_reddit_sentiment.py` + `.github/workflows/daily-ingest.yml` (pending commit at time of writing). BTC backfilled from 2024-01-10 (spot ETF approval day); r/Bitcoin sentiment starts from first cron run.
- [x] **Live "Today's signals" block on homepage** — imports `data/daily/latest_btc.json` and `data/daily/latest_reddit.json` at build time; Vercel redeploys on each cron commit, keeping the site fresh daily without runtime API calls.
- [x] **Research design block on homepage** — two-experiment framing (A: time regime, B: live Reddit) with explicit confound acknowledgement on the Reddit source.
- [x] **README / landing-page limitation → deliverable narrative** — L1–L5 rows on the site, research-design section in README with experiment design and confound note.
- [ ] Daily inference → predictions log (feeds L4 rolling accuracy)
- [ ] Live dashboard reads predictions + rolling accuracy

## Decisions log

- **2026-11-11** — **Experiment A executed on H1 2022 (not May-June).** Initial Terra/Luna + 3AC window (May-June, 61 days) gave R² = -2.43 — the 70/30 split put the test set inside the June 3AC crash, a distribution the training set never saw. Extended to full H1 2022 (Jan-Jun, 181 days, 7.3M tweets) which directly addresses the dissertation's own named limitation on sample size (Section 7.3) and its Future Work recommendation of a "6-12 month window on the 2022 bear market" (Section 7.4). Re-run gave R² = 0.018 (positive), MAE $3,670, Directional 60.4%. Positive-but-low R² is an honest generalisation finding, not a tuning failure. Numbers published to site.
- **2026-11-09** — **Homepage L1–L5 list trimmed to shipped + in-progress only.** L1 (FinBERT) and L4 (rolling directional accuracy) removed from the public limitations list and moved into the README's new "Future work" section. L3 marked Done at the sentiment-source level (Reddit as second source); on-chain integration also moved to README future-work. Rationale: public face should read "look what I built / am building" rather than "look at this long planned list"; internal docs (PROGRESS, README) still carry the honest trail of what's queued and why. All three items remain trackable; they're just not advertised on the site as unshipped promises.
- **2026-11-09** — **Phase 2 reframed as two parallel experiments**, not a single "VADER-replacement" narrative. Experiment A (time regime generalisation) controls for sentiment method and varies time window using the 2022 Twitter dataset I already have on disk. Experiment B (live 2026 Reddit) is framed as a separate research question rather than a direct continuation, because the Twitter→Reddit source switch is a confound that L2 alone cleanly controls for. Writes up on the homepage and in the README.
- **2026-11-09** — **FinBERT deferred as future work**. VADER + crypto lexicon was the dissertation's novel methodological contribution; Phase 2 keeps it fixed as the sentiment method across both experiments so results are cleanly attributable to time-period and source variables. Transformer ablation (FinBERT or CryptoBERT) revisited post-interviews.
- **2026-11-09** — **Reddit RSS endpoint** chosen over JSON API after Reddit's anonymous `/top.json` endpoints began returning 403 in testing. RSS feed (`/top/.rss?t=day`) still serves anonymous clients and includes both title and selftext; may need to upgrade to OAuth if GitHub Actions shared IPs get rate-limited in CI.
- **2026-11-09** — **BTC daily backfill starts at 2024-01-10** (spot ETF approval day) rather than from first cron run. Gives the chart meaningful history on day one and aligns the framing with the dissertation's Future Work F4 (post-ETF era).
- **2026-11-09** — **Reddit sentiment data starts from first cron run**, not backfilled. Reddit's historical top-of-day endpoints are unreliable for anonymous access. Acknowledged in homepage copy.
- **2026-08-17** — Headline 2026 sentiment model: **FinBERT** (directly closes L1 as named in dissertation future-work), with **CryptoBERT** benchmarked in ablation. *Superseded 2026-11-09 — see decision above.*
- **2026-08-17** — Python 3.12.14 chosen over 3.13/3.14 for TF 2.17 compatibility.
- **2026-08-17** — `.python-version` committed (not gitignored) for reproducibility.

## Blocked / open questions

- Will the Reddit RSS endpoint hold against GitHub Actions shared IPs? First few cron runs will tell. OAuth upgrade path pre-planned.
- Daily model inference (needed for L4 rolling accuracy) still pending — requires either shipping the trained Keras model into the daily workflow or standing up a lightweight inference endpoint.
