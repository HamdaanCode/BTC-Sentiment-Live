# Project Progress

Rolling status doc. Updated as work is completed. Newest at top of each section.

Last updated: 2026-08-17 (3c complete — Phase 2 reproducible pipeline done)

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

### Reproduced dissertation baseline (seed=42)
This is the benchmark FinBERT has to beat.

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
- [ ] **L1 (VADER rule-based)** → swap VADER for FinBERT as the headline 2026 model; benchmark CryptoBERT alongside in ablation
- [ ] **L2 (single window Sept–Nov 2021)** → extend training set to include 2022 bear + 2024 post-ETF recovery
- [ ] **L3 (Twitter-only source)** → add Reddit ingest (`r/Bitcoin`, `r/CryptoCurrency`)
- [ ] **L4 (weak directional accuracy)** → log directional accuracy daily on the live board alongside MAE
- [ ] **L5 (unweighted aggregation)** → weight posts by author reach (Reddit upvotes; Twitter followers if re-added)

### Phase 3 — live inference & dashboard
- [ ] Daily ingest cron
- [ ] Daily inference → predictions log
- [ ] Live dashboard reads predictions + rolling accuracy
- [ ] README / landing-page limitation → deliverable narrative

## Decisions log

- **2026-08-17** — Headline 2026 sentiment model: **FinBERT** (directly closes L1 as named in dissertation future-work), with **CryptoBERT** benchmarked in ablation. Rationale: FinBERT gives the cleanest narrative arc; CryptoBERT is the stronger candidate for crypto-slang OOD and may promote to headline if it wins.
- **2026-08-17** — Python 3.12.14 chosen over 3.13/3.14 for TF 2.17 compatibility.
- **2026-08-17** — `.python-version` committed (not gitignored) for reproducibility.

## Blocked / open questions

- None right now.
