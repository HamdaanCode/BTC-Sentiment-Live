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
    <main className="mx-auto w-full max-w-3xl px-5 py-12 sm:py-20">
      {/* HEADER */}
      <header className="mb-14">
        <div className="mb-3 text-xs text-[var(--muted)]">
          ~/btc-sentiment-live
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-[var(--accent)] sm:text-4xl">
          bitcoin-sentiment-live
        </h1>
        <p className="mt-5 leading-relaxed text-[var(--foreground)]">
          A Bitcoin price prediction project. Three LSTMs trained on daily
          price action plus social-media sentiment, answering a question that
          keeps getting more relevant as the market gets more reflexive:{" "}
          <span className="text-[var(--accent)]">
            does what people say about Bitcoin actually help predict where its
            price is going next?
          </span>
        </p>
        <p className="mt-3 leading-relaxed text-[var(--muted)]">
          Started as my dissertation at Heriot-Watt. Now running as a live
          extension — daily data ingested through a GitHub Actions cron, new
          experiments run against the original pipeline, everything published
          here with the actual numbers.
        </p>
        <div className="mt-7 flex flex-wrap gap-3 text-xs">
          <a
            href="https://github.com/HamdaanCode/BTC-Sentiment-Live"
            target="_blank"
            rel="noopener noreferrer"
            className="border border-[var(--border)] px-3 py-1.5 text-[var(--foreground)] transition-colors hover:border-[var(--accent)] hover:text-[var(--accent)]"
          >
            [ github ↗ ]
          </a>
          <a
            href="#live"
            className="border border-[var(--border)] px-3 py-1.5 text-[var(--foreground)] transition-colors hover:border-[var(--accent)] hover:text-[var(--accent)]"
          >
            [ today&apos;s signals ↓ ]
          </a>
        </div>
      </header>

      {/* LIVE SIGNALS — sticky-post styling */}
      <section id="live" className="forum-post mb-14 p-5 scroll-mt-16">
        <div className="mb-4 flex items-baseline justify-between">
          <div className="text-xs uppercase tracking-wider text-[var(--accent)]">
            [ live ] today&apos;s signals
          </div>
          <div className="text-xs text-[var(--muted)]">daily cron</div>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Metric
            label="btc close"
            value={formatUsd(latestBtc.close)}
            caption={latestBtc.date}
          />
          <Metric
            label="r/bitcoin"
            value={sentimentLabel(latestReddit.weighted_compound)}
            caption={`w ${latestReddit.weighted_compound.toFixed(3)} · r ${latestReddit.avg_compound.toFixed(3)}`}
          />
          <Metric
            label="posts scored"
            value={String(latestReddit.posts_scored)}
            caption={`${latestReddit.posts_with_selftext} with selftext`}
          />
          <Metric
            label="btc rows"
            value={String(latestBtc.rows_total)}
            caption={`since ${latestBtc.backfill_start}`}
          />
        </div>
      </section>

      {/* ORIGINAL POST — MARCH 2026 */}
      <section className="mb-14">
        <ForumHeader
          marker="original post"
          author="hamdaan.btc"
          date="March 2026"
          title="First results — can tweets predict Bitcoin price?"
        />
        <div className="mt-5 space-y-4 leading-relaxed text-[var(--foreground)]">
          <p>
            Three LSTMs trained on 91 days of daily Bitcoin price data plus
            tweet-derived sentiment, covering the Sept–Nov 2021 all-time-high
            run-up.{" "}
            <span className="text-[var(--accent)]">1.38 million tweets</span>{" "}
            scored with VADER plus a custom 35-term crypto lexicon.
          </p>
          <p>
            Three sentiment variants were compared head-to-head: a plain VADER
            baseline, a crypto-adjusted lexicon (adding terms like{" "}
            <span className="font-mono text-[var(--accent)]">hodl</span>,{" "}
            <span className="font-mono text-[var(--accent)]">moon</span>,{" "}
            <span className="font-mono text-[var(--accent)]">rekt</span> on
            VADER&apos;s scoring scale), and a crypto-plus-intensity variant
            that boosted CAPS, exclamations, and intensifiers.{" "}
            <span className="text-[var(--accent)]">
              Model B (crypto lexicon only) won.
            </span>
          </p>
        </div>
        <div className="mt-6 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Metric label="R²" value="0.787" caption="Model B" />
          <Metric label="MAE" value="$1,300" caption="test set" />
          <Metric label="MAPE" value="2.16%" caption="Model B" />
          <Metric
            label="dataset"
            value={`${(totalTweets / 1_000_000).toFixed(2)}M`}
            caption="tweets"
          />
        </div>
        <div className="mt-8 space-y-5">
          <ChartCard title="Bitcoin closing price" subtitle={dateRange}>
            <PriceChart data={dissertationData} />
          </ChartCard>
          <ChartCard
            title="Daily sentiment — three models"
            subtitle="higher = more bullish on the day"
          >
            <SentimentChart data={dissertationData} />
            <Legend
              items={[
                { color: "#8b949e", label: "A · baseline VADER" },
                { color: "#f7931a", label: "B · crypto lexicon (winner)" },
                { color: "#d29922", label: "C · intensity-weighted" },
              ]}
            />
          </ChartCard>
          <ChartCard
            title="Tweet volume per day"
            subtitle="BTC-tagged tweets in the training window"
          >
            <TweetVolumeChart data={dissertationData} />
          </ChartCard>
        </div>
      </section>

      {/* REPLY #1 — NOV 2026, 2022 BEAR EXTENSION */}
      <section className="mb-14">
        <ForumHeader
          marker="reply #1"
          author="hamdaan.btc"
          date="November 2026"
          title="Does it still work in a bear market?"
        />
        <div className="mt-5 space-y-4 leading-relaxed text-[var(--foreground)]">
          <p>
            Same methodology, new data. Took the{" "}
            <span className="text-[var(--accent)]">H1 2022</span> Bitcoin
            tweet dataset — six months of posts, 7.3 million tweets through
            the Terra/Luna collapse in May and the 3AC collapse in June — and
            retrained Model B on it.
          </p>
          <p>
            The honest answer: partly.{" "}
            <span className="text-[var(--accent)]">
              R² collapsed from 0.787 to 0.018
            </span>
            . The model struggles to predict Bitcoin price magnitudes across a
            full regime change. But{" "}
            <span className="text-[var(--accent)]">
              directional accuracy went up from 38% to 60%
            </span>
            , meaningfully above random. The pipeline keeps directional signal
            when markets are trending, even when the magnitudes fall apart.
          </p>
        </div>
        <div className="mt-6 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Metric label="R²" value="0.018" caption="bear regime" />
          <Metric label="MAE" value="$3,670" caption="test set" />
          <Metric label="MAPE" value="15.7%" caption="Model B2" />
          <Metric
            label="dir. accuracy"
            value="60.4%"
            caption="above random"
          />
        </div>
      </section>

      {/* REPLY #2 — NOV 2026, REDDIT RANK WEIGHTING */}
      <section className="mb-14">
        <ForumHeader
          marker="reply #2"
          author="hamdaan.btc"
          date="November 2026"
          title="Not all Reddit posts are equal."
        />
        <div className="mt-5 space-y-4 leading-relaxed text-[var(--foreground)]">
          <p>
            The daily sentiment pipeline was averaging every post equally. A
            top-of-the-day post and a bottom-of-the-day post counted the same,
            which smooths out the signal that engagement actually drives.
          </p>
          <p>
            Reddit&apos;s public endpoint doesn&apos;t expose raw vote counts,
            but it does expose rank — and Reddit&apos;s ranking is
            score-based. So the ingest now weights each post by its rank in
            the top-of-day listing.{" "}
            <span className="text-[var(--accent)]">
              Top posts count about 4.7x more
            </span>{" "}
            than bottom-ranked posts.
          </p>
          <p>
            Today the raw sentiment reads{" "}
            <span className="font-mono text-[var(--accent)]">
              {latestReddit.avg_compound >= 0 ? "+" : ""}
              {latestReddit.avg_compound.toFixed(3)}
            </span>
            ; the weighted sentiment reads{" "}
            <span className="font-mono text-[var(--accent)]">
              {latestReddit.weighted_compound >= 0 ? "+" : ""}
              {latestReddit.weighted_compound.toFixed(3)}
            </span>
            . When they diverge like this, top-ranked posts are more muted
            than the lower-ranked ones. That&apos;s the gap that previously
            got averaged away.
          </p>
        </div>
      </section>

      {/* COMING SOON */}
      <section className="mt-20 border-t border-dashed border-[var(--border)] pt-10">
        <div className="space-y-3 text-center text-sm text-[var(--muted)]">
          <div>&gt; drafting next reply...</div>
          <div className="text-[var(--accent)]">
            &gt; more coming soon
            <span className="animate-pulse">_</span>
          </div>
        </div>
      </section>
    </main>
  );
}

function ForumHeader({
  marker,
  author,
  date,
  title,
}: {
  marker: string;
  author: string;
  date: string;
  title: string;
}) {
  return (
    <>
      <div className="mb-3 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-[var(--muted)]">
        <span className="text-[var(--accent)]">[{marker}]</span>
        <span>·</span>
        <span>
          posted by{" "}
          <span className="text-[var(--foreground)]">{author}</span>
        </span>
        <span>·</span>
        <span>{date}</span>
      </div>
      <h2 className="text-2xl font-bold leading-tight tracking-tight text-[var(--foreground)] sm:text-3xl">
        {title}
      </h2>
    </>
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
    <div className="border border-[var(--border)] bg-[var(--panel)] p-3">
      <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">
        {label}
      </div>
      <div className="mt-1 text-xl font-bold tabular-nums text-[var(--accent)] sm:text-2xl">
        {value}
      </div>
      <div className="mt-0.5 text-[10px] text-[var(--muted)]">{caption}</div>
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
    <div className="border border-[var(--border)] bg-[var(--panel)] p-4">
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-[var(--foreground)]">
          {title}
        </h3>
        <p className="mt-0.5 text-xs text-[var(--muted)]">{subtitle}</p>
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
    <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-[var(--muted)]">
      {items.map((it) => (
        <div key={it.label} className="flex items-center gap-1.5">
          <span
            className="h-2 w-2"
            style={{ background: it.color }}
          />
          {it.label}
        </div>
      ))}
    </div>
  );
}
