import {
  PriceChart,
  SentimentChart,
  TweetVolumeChart,
} from "./DissertationCharts";
import { dissertationData } from "./dissertationData";
import latestBtc from "../data/daily/latest_btc.json";
import latestReddit from "../data/daily/latest_reddit.json";

const totalTweets = dissertationData.reduce((s, d) => s + d.tweetCount, 0);
const dateRange = `${dissertationData[0].date} → ${
  dissertationData[dissertationData.length - 1].date
}`;

const formatUsd = (n: number) =>
  `$${n.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;

const sentimentLabel = (compound: number) =>
  compound >= 0.05 ? "Positive" : compound <= -0.05 ? "Negative" : "Neutral";

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-4xl px-6 py-16 sm:py-24">
      <header className="mb-16">
        <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
          Machine learning for Bitcoin price prediction
        </h1>
        <p className="mt-4 max-w-2xl text-lg leading-relaxed text-zinc-600 dark:text-zinc-400">
          An ongoing personal project. Three LSTM models forecasting daily
          Bitcoin close across the Sept–Nov 2021 all-time high, using
          tweet-derived sentiment as an input feature. Baseline VADER compared
          against a custom 35-term crypto lexicon and an intensity-weighted
          variant. Now being extended into a live pipeline for 2026 markets.
        </p>
        <div className="mt-8 flex flex-wrap gap-3 text-sm">
          <a
            href="https://github.com/HamdaanCode/BTC-Sentiment-Live"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full border border-zinc-300 px-4 py-2 font-medium transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-900"
          >
            Repo ↗
          </a>
          <a
            href="#phase-2"
            className="rounded-full border border-zinc-300 px-4 py-2 font-medium transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-900"
          >
            Phase 2 progress ↓
          </a>
        </div>
      </header>

      <section className="mb-20">
        <div className="mb-8">
          <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            The research, in numbers
          </h2>
          <p className="mt-3 max-w-2xl text-zinc-600 dark:text-zinc-400">
            Three LSTM variants trained on daily Bitcoin data plus tweet-derived
            sentiment. Sentiment came in three flavours: a baseline VADER
            (Model A), a custom 35-term crypto lexicon (Model B), and an
            intensity-weighted variant (Model C). Model B won.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Metric label="R²" value="0.787" caption="best model" />
          <Metric label="MAE" value="$1,300" caption="test set" />
          <Metric label="MAPE" value="2.16%" caption="Model B" />
          <Metric
            label="Dataset"
            value={`${(totalTweets / 1_000_000).toFixed(2)}M`}
            caption="tweets"
          />
        </div>

        <div className="mt-12 space-y-10">
          <ChartCard
            title="Bitcoin closing price"
            subtitle={`Daily close · ${dateRange}`}
          >
            <PriceChart data={dissertationData} />
          </ChartCard>

          <ChartCard
            title="Daily sentiment — three models"
            subtitle="Mean per-tweet sentiment aggregated per day. Higher = more bullish."
          >
            <SentimentChart data={dissertationData} />
            <Legend
              items={[
                { color: "#94a3b8", label: "A · baseline VADER" },
                { color: "#10b981", label: "B · crypto lexicon (winner)" },
                { color: "#a78bfa", label: "C · intensity-weighted" },
              ]}
            />
          </ChartCard>

          <ChartCard
            title="Tweet volume per day"
            subtitle="Volume of BTC-tagged tweets in the training window."
          >
            <TweetVolumeChart data={dissertationData} />
          </ChartCard>
        </div>
      </section>

      <section id="phase-2" className="mb-20 scroll-mt-16">
        <div className="mb-8">
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-emerald-300 bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
            Phase 2 · in progress
          </div>
          <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            Taking it live in 2026
          </h2>
          <p className="mt-3 max-w-2xl text-zinc-600 dark:text-zinc-400">
            Phase 1 asked{" "}
            <em>does sentiment help predict Bitcoin price?</em> Phase 2 asks the
            harder questions: <em>does the pipeline generalise across market
            regimes</em>, and <em>does sentiment from a different community
            still carry signal in a post-ETF market?</em>
          </p>
        </div>

        <div className="mb-4">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
            Research design
          </h3>
          <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
            VADER with my crypto lexicon stays fixed as the sentiment method
            across Phase 2 so time-period and source variables stay isolated.
            A transformer ablation (FinBERT or CryptoBERT) is deferred.
          </p>
        </div>
        <ul className="mb-10 space-y-3">
          <ExperimentRow
            id="A"
            title="Time regime generalisation"
            varies="Time period (2021 Sept–Nov vs 2022 bear market)"
            controls="Sentiment method (VADER + crypto lexicon), source (Twitter)"
            question="Does my pipeline still work when the market mood flips?"
          />
          <ExperimentRow
            id="B"
            title="Live 2026 Reddit pipeline"
            varies="Source (Reddit instead of Twitter), time period (post-ETF)"
            controls="Sentiment method (VADER + crypto lexicon)"
            question="Does sentiment from a different community carry signal in a post-ETF market?"
          />
        </ul>

        <div className="mb-4">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
            Today&apos;s signals
          </h3>
          <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
            Daily BTC close and r/Bitcoin sentiment pulled fresh every day via a
            GitHub Actions cron. BTC series begins on the spot ETF approval day
            (10 Jan 2024); the Reddit series starts from the first cron run.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Metric
            label="BTC close"
            value={formatUsd(latestBtc.close)}
            caption={latestBtc.date}
          />
          <Metric
            label="r/Bitcoin"
            value={sentimentLabel(latestReddit.avg_compound)}
            caption={`compound ${latestReddit.avg_compound.toFixed(3)}`}
          />
          <Metric
            label="Posts scored"
            value={String(latestReddit.posts_scored)}
            caption={`${latestReddit.posts_with_selftext} with selftext`}
          />
          <Metric
            label="BTC rows"
            value={String(latestBtc.rows_total)}
            caption={`since ${latestBtc.backfill_start}`}
          />
        </div>

        <div className="mb-4">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
            Reproduced baseline (seed=42)
          </h3>
          <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
            The 2021 pipeline now runs end-to-end from a clean venv.
            These are the numbers the 2022 extension has to match (or not).
          </p>
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Metric label="R²" value="0.782" caption="Model A · reproduced" />
          <Metric label="MAE" value="$1,293" caption="Model A · test set" />
          <Metric label="MAPE" value="2.15%" caption="Model A" />
          <Metric
            label="Dir. accuracy"
            value="34.6%"
            caption="below 50% → L4 confirmed"
          />
        </div>

        <div className="mt-12 mb-4">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
            Known limitations → Phase 2 deliverables
          </h3>
          <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
            Every limitation surfaced in Phase 1 mapped to what closes it.
          </p>
        </div>
        <ul className="space-y-3">
          <LimitationRow
            id="L1"
            status="planned"
            title="VADER is rule-based, misses context"
            deliverable="Transformer ablation (FinBERT or CryptoBERT) deferred as future work. VADER + crypto lexicon remains the fixed sentiment method across Phase 2 so time-period and source variables can be isolated cleanly."
          />
          <LimitationRow
            id="L2"
            status="planned"
            title="Only a 91-day window (Sept–Nov 2021)"
            deliverable="Extend the training set to include the 2022 bear market and the 2024 post-ETF recovery, so the model sees multiple regimes."
          />
          <LimitationRow
            id="L3"
            status="active"
            title="Twitter-only sentiment source"
            deliverable="Reddit ingest live: daily pull of top 25 r/Bitcoin posts, VADER-scored title + selftext, aggregated into a daily compound score. On-chain metrics slated after."
          />
          <LimitationRow
            id="L4"
            status="planned"
            title="Weak directional accuracy (~40%)"
            deliverable="Log rolling directional accuracy alongside MAE on this page, updated daily as the live pipeline logs predictions."
          />
          <LimitationRow
            id="L5"
            status="planned"
            title="All posts weighted equally in daily aggregation"
            deliverable="Weight posts by author reach — Reddit upvotes, and follower count if Twitter is re-added."
          />
        </ul>

        <div className="mt-12 mb-4">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
            Pipeline status
          </h3>
        </div>
        <ul className="space-y-2 text-sm">
          <StatusRow status="done" label="Environment: pyenv + Python 3.12 + reproducible .venv" />
          <StatusRow status="done" label="Training pipeline reproduces the 2021 baseline end-to-end" />
          <StatusRow status="done" label="Daily ingest cron (BTC price + r/Bitcoin sentiment)" />
          <StatusRow status="active" label="L2 · 2022 bear market retrain (VADER + crypto lexicon)" />
          <StatusRow status="planned" label="Live inference + rolling accuracy on this page" />
        </ul>
      </section>
    </main>
  );
}

type Status = "done" | "active" | "planned";

const STATUS_DOT: Record<Status, string> = {
  done: "bg-emerald-500",
  active: "bg-amber-400",
  planned: "bg-zinc-400 dark:bg-zinc-600",
};

const STATUS_LABEL: Record<Status, string> = {
  done: "Done",
  active: "In progress",
  planned: "Planned",
};

function ExperimentRow({
  id,
  title,
  varies,
  controls,
  question,
}: {
  id: string;
  title: string;
  varies: string;
  controls: string;
  question: string;
}) {
  return (
    <li className="rounded-xl border border-zinc-200 bg-zinc-50/60 p-4 dark:border-zinc-800 dark:bg-zinc-900/60">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="font-mono text-xs font-semibold text-zinc-500 dark:text-zinc-400">
          Experiment {id}
        </span>
        <span className="text-sm font-medium">{title}</span>
      </div>
      <dl className="mt-2 space-y-1 text-sm text-zinc-600 dark:text-zinc-400">
        <div className="flex gap-2">
          <dt className="shrink-0 font-mono text-xs uppercase tracking-wider text-zinc-500">
            varies
          </dt>
          <dd>{varies}</dd>
        </div>
        <div className="flex gap-2">
          <dt className="shrink-0 font-mono text-xs uppercase tracking-wider text-zinc-500">
            controls
          </dt>
          <dd>{controls}</dd>
        </div>
        <div className="flex gap-2">
          <dt className="shrink-0 font-mono text-xs uppercase tracking-wider text-zinc-500">
            asks
          </dt>
          <dd className="italic">{question}</dd>
        </div>
      </dl>
    </li>
  );
}

function LimitationRow({
  id,
  status,
  title,
  deliverable,
}: {
  id: string;
  status: Status;
  title: string;
  deliverable: string;
}) {
  return (
    <li className="rounded-xl border border-zinc-200 bg-zinc-50/60 p-4 dark:border-zinc-800 dark:bg-zinc-900/60">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="font-mono text-xs font-semibold text-zinc-500 dark:text-zinc-400">
          {id}
        </span>
        <span className="text-sm font-medium">{title}</span>
        <span className="ml-auto inline-flex items-center gap-1.5 text-xs text-zinc-500 dark:text-zinc-400">
          <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[status]}`} />
          {STATUS_LABEL[status]}
        </span>
      </div>
      <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
        {deliverable}
      </p>
    </li>
  );
}

function StatusRow({ status, label }: { status: Status; label: string }) {
  return (
    <li className="flex items-center gap-2.5">
      <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[status]}`} />
      <span
        className={
          status === "done"
            ? "text-zinc-700 line-through decoration-zinc-400 dark:text-zinc-400"
            : status === "active"
            ? "font-medium"
            : "text-zinc-500 dark:text-zinc-400"
        }
      >
        {label}
      </span>
    </li>
  );
}

function Metric({
  label,
  value,
  caption,
}: {
  label: string;
  value: string;
  caption: string;
}) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-zinc-50/60 p-4 dark:border-zinc-800 dark:bg-zinc-900/60">
      <div className="text-xs font-medium uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
        {label}
      </div>
      <div className="mt-1 font-mono text-2xl font-semibold tabular-nums">
        {value}
      </div>
      <div className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">
        {caption}
      </div>
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900/40">
      <div className="mb-4">
        <h3 className="text-sm font-semibold">{title}</h3>
        <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">
          {subtitle}
        </p>
      </div>
      {children}
    </div>
  );
}

function Legend({
  items,
}: {
  items: { color: string; label: string }[];
}) {
  return (
    <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-zinc-600 dark:text-zinc-400">
      {items.map((it) => (
        <div key={it.label} className="flex items-center gap-1.5">
          <span
            className="h-2 w-2 rounded-full"
            style={{ background: it.color }}
          />
          {it.label}
        </div>
      ))}
    </div>
  );
}
