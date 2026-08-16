import {
  PriceChart,
  SentimentChart,
  TweetVolumeChart,
} from "./DissertationCharts";
import { dissertationData } from "./dissertationData";

const totalTweets = dissertationData.reduce((s, d) => s + d.tweetCount, 0);
const dateRange = `${dissertationData[0].date} → ${
  dissertationData[dissertationData.length - 1].date
}`;

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-4xl px-6 py-16 sm:py-24">
      <header className="mb-16">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-zinc-300 bg-zinc-100 px-3 py-1 text-xs font-medium text-zinc-700 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300">
          BSc Dissertation · Heriot-Watt · First Class (84%)
        </div>
        <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
          Machine learning for Bitcoin price prediction
        </h1>
        <p className="mt-4 max-w-2xl text-lg leading-relaxed text-zinc-600 dark:text-zinc-400">
          Three LSTM models forecasting daily Bitcoin close across the Sept–Nov
          2021 all-time high, using tweet-derived sentiment as an input feature.
          Compared a baseline VADER against a custom 35-term crypto lexicon and
          an intensity-weighted variant.
        </p>
        <div className="mt-8 flex flex-wrap gap-3 text-sm">
          <a
            href="https://github.com/HamdaanCode"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full border border-zinc-300 px-4 py-2 font-medium transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-900"
          >
            GitHub ↗
          </a>
        </div>
      </header>

      <section className="mb-20">
        <div className="mb-8">
          <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            The dissertation, in numbers
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
    </main>
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
