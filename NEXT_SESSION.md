# Next session — plan

Dated 2026-11-09. Delete this file (or overwrite it with the next batch) once the steps below are done.

The goal this session is to **close Experiment A (L2): train Model B2 on the 2022 bear-market window, publish the results on the site.** Optional stretch: ship L5 (upvote weighting).

---

## 0. Housekeeping first (5 min)

### 0a. Check Vercel redeployed cleanly from last push
Open [https://btc-sentiment-live.vercel.app](https://btc-sentiment-live.vercel.app) and scroll to "Taking it live in 2026". You should see:
- "Research design" block with Experiment A and B cards
- "Today's signals" block with four metric tiles (BTC close, r/Bitcoin, Posts scored, BTC rows)
- L1 badge flipped to Planned, L2 showing as active in Pipeline Status

If anything looks stale, the Vercel deploy didn't trigger — check Vercel dashboard for the `BTC-Sentiment-Live` project.

### 0b. Give the daily cron permission to push back
One-off GitHub setting. Without it, the cron succeeds but can't commit the updated CSVs back to the repo.

- Repo on GitHub → **Settings** → **Actions** → **General**
- Under **Workflow permissions**: select **"Read and write permissions"** → Save

### 0c. Verify the daily cron works at least once
- GitHub repo → **Actions** tab → **Daily ingest** workflow → **Run workflow** → Run on `main`
- Should finish in ~90 seconds and land a bot-authored commit on main (`chore(data): daily ingest 2026-11-XX`)
- If the Reddit step 403s (shared GitHub Actions IPs sometimes get rate-limited by Reddit), the BTC step still commits; we'll switch Reddit to OAuth next session if it keeps failing

---

## 1. Run Experiment A (L2) — 2022 bear-market retrain (~90 min)

### 1a. Filter 2022 raw tweets to the Terra/Luna + 3AC collapse window
Mirrors the dissertation's "heightened trading activity" criterion from Sept-Nov 2021. This is the right apples-to-apples window for a bear-regime comparison.

```bash
cd ~/Documents/btc-sentiment
source .venv/bin/activate
mkdir -p artifacts/raw

python -c "
import pandas as pd
src = '/Users/hamdaanismail/Documents/Extracted Tweets/archive (1) 2/bitcoin-tweets-2022.csv'
df = pd.read_csv(src, on_bad_lines='skip', engine='python')
df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
# May 1 → July 1 2022: Terra/Luna (May 7-15) + 3AC collapse (Jun 15-ish)
window = df[(df['datetime'] >= '2022-05-01') & (df['datetime'] < '2022-07-01')]
window.to_csv('artifacts/raw/2022_window_raw.csv', index=False)
print(f'filtered {len(df):,} -> {len(window):,} tweets')
"
```

### 1b. Preprocess → score → prices → aggregate

```bash
# Preprocess (10-20 min)
python -m src.preprocess \
    --input  artifacts/raw/2022_window_raw.csv \
    --output artifacts/raw/2022_cleaned.csv

# Score with Models A, B, C (20-40 min — the heaviest step)
python -m src.score_vader \
    --input  artifacts/raw/2022_cleaned.csv \
    --output artifacts/raw/2022_scored.csv

# Pull 2022 BTC OHLCV for the same window (5 sec)
python scripts/fetch_prices.py \
    --start 2022-05-01 --end 2022-07-01 \
    --output artifacts/raw/btc_prices_2022.csv

# Join per-tweet scores + daily prices → LSTM training CSV (2 min)
python -m src.aggregate_vader \
    --scored-tweets artifacts/raw/2022_scored.csv \
    --prices        artifacts/raw/btc_prices_2022.csv \
    --output        reference/data/NEW_lstm_training_data_2022.csv
```

### 1c. Train Model B2 (~30-60 min)

```bash
python -m src.train --model B2 --seed 42
```

Artifacts land at `artifacts/B2_seed42/`: `model.keras`, `scaler.pkl`, `metrics.json`, `history.json`.

### 1d. Evaluate (1 min)

```bash
python -m src.evaluate --model B2 --seed 42
```

Writes `artifacts/B2_seed42/eval.json`. Numbers to pull out and publish:
- **R²**
- **MAE ($)**
- **MAPE (%)**
- **Directional accuracy (%)**

Don't pre-dread a lower R² than the 2021 baseline — a degradation under regime change is the expected finding and makes the whole experiment meaningful.

---

## 2. Publish the results (20 min)

### 2a. Update `app/page.tsx`

Right after the "Today's signals" block, add a new "Experiment A results (2022 bear window)" block mirroring the Reproduced Baseline layout:

```tsx
<div className="mb-4">
  <h3 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
    Experiment A results (2022 bear window)
  </h3>
  <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
    Model B2: VADER + crypto lexicon, retrained on the Terra/Luna + 3AC
    collapse window (May–June 2022). Same architecture as Phase 1 Model B.
  </p>
</div>
<div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
  <Metric label="R²"     value="0.XXX"      caption="Model B2 · bear regime" />
  <Metric label="MAE"    value="$X,XXX"     caption="Model B2 · test set" />
  <Metric label="MAPE"   value="X.XX%"      caption="Model B2" />
  <Metric label="Dir. accuracy" value="XX.X%" caption="L4 recap" />
</div>
```

### 2b. Update the L2 row in the limitations list

Change from:
```
<LimitationRow
  id="L2"
  status="planned"
  title="Only a 91-day window (Sept–Nov 2021)"
  deliverable="Extend the training set to include the 2022 bear market..."
/>
```

To (replace XXX with your real numbers):
```
<LimitationRow
  id="L2"
  status="done"
  title="Only a 91-day window (Sept–Nov 2021)"
  deliverable="Closed. Model B2 retrained on May–June 2022 bear window (Terra/Luna + 3AC). R² X.XXX, MAE $X,XXX, directional accuracy XX.X%. [one-sentence takeaway on whether pipeline generalises]."
/>
```

### 2c. Update Pipeline Status row

Flip `"L2 · 2022 bear market retrain (VADER + crypto lexicon)"` from `status="active"` to `status="done"`.

### 2d. Update `PROGRESS.md`

- Change "Last updated" line to the new date
- Mark `[x]` on L2
- Add a dated entry to the Decisions log summarising what Experiment A found

---

## 3. (Optional stretch) Ship L5 — upvote weighting (~90 min)

If time permits, close a second L item in the same session.

### 3a. Modify `scripts/daily_reddit_sentiment.py`

- In `fetch_posts()`, pull `upvotes` from the RSS feed (Reddit's Atom feed exposes `media:statistics` or similar; may need to fall back to the HTML content parse)
- In `score_posts()`, compute an additional `weighted_compound` using `log(1 + upvotes)` as the per-post weight
- Add `weighted_compound` column to `COLUMNS` and the output row

### 3b. Update `app/page.tsx` "Today's signals" block to show both metrics

Add a 5th tile or split the r/Bitcoin tile into two — raw compound and upvote-weighted. Visual difference between them is the whole story of L5.

### 3c. Flip L5

- Site: `status="done"` on L5 row
- PROGRESS.md: `[x]` + decision log entry

---

## 4. Commit and push

```bash
git add .
git status  # eyeball — expect app/page.tsx, PROGRESS.md, NEXT_SESSION.md, maybe reference/data/, maybe scripts/daily_reddit_sentiment.py
git commit -m "Experiment A: Model B2 2022 bear-regime results; L2 closed"
git push
```

Vercel redeploys. Site now shows real experiment results.

---

## Success criteria for this session

Before you stop:
- [ ] Vercel site shows real 2022 R²/MAE numbers under "Experiment A results"
- [ ] L2 flipped to "Done" (green dot) on the site
- [ ] PROGRESS.md updated with date + decision log entry
- [ ] (Stretch) L5 shipped and flipped to Done

After this session you can honestly say in interviews: *"Two of the dissertation's five named limitations closed with published results on the live site; one deliberately deferred with methodological reasoning; two remaining as future work with a concrete plan."*

That's a materially stronger story than today's state.
