"""Daily Reddit sentiment ingest for r/Bitcoin.

Pulls the top posts of the day from r/Bitcoin via the public Atom/RSS feed
(no auth required, still open after Reddit's 2024 API hardening) and scores
title + stripped selftext with VADER. Appends one aggregated row per day to
data/daily/reddit_sentiment.csv.

Idempotent: if today's row already exists, it is overwritten with the
latest numbers rather than duplicated.
"""
from __future__ import annotations

import html
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import feedparser
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "daily" / "reddit_sentiment.csv"
JSON_PATH = REPO_ROOT / "data" / "daily" / "latest_reddit.json"

SUBREDDIT = "Bitcoin"
POST_LIMIT = 25
FEED_URL = f"https://www.reddit.com/r/{SUBREDDIT}/top/.rss?t=day&limit={POST_LIMIT}"
USER_AGENT = "btc-sentiment-live/1.0 (github.com/HamdaanCode/BTC-Sentiment-Live)"

COLUMNS = [
    "date",
    "subreddit",
    "posts_scored",
    "posts_with_selftext",
    "avg_compound",
    "avg_pos",
    "avg_neu",
    "avg_neg",
    "pos_count",
    "neu_count",
    "neg_count",
    "top_post_title",
]

# Reddit's RSS content is HTML inside a wrapper table. The real selftext
# (when present) sits inside the <!-- SC_OFF --> … <!-- SC_ON --> markers,
# and everything else is link markup, "submitted by" lines and permalinks.
SELFTEXT_RE = re.compile(r"<!--\s*SC_OFF\s*-->(.*?)<!--\s*SC_ON\s*-->", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def extract_selftext(content_html: str) -> str:
    """Pull just the user-written body out of Reddit's RSS content block."""
    if not content_html:
        return ""
    match = SELFTEXT_RE.search(content_html)
    if not match:
        return ""
    stripped = TAG_RE.sub(" ", match.group(1))
    decoded = html.unescape(stripped)
    return WS_RE.sub(" ", decoded).strip()


def fetch_posts() -> list[dict]:
    """Return [{title, selftext}, ...] for today's top posts."""
    resp = requests.get(FEED_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    parsed = feedparser.parse(resp.content)
    posts: list[dict] = []
    for entry in parsed.entries:
        title = html.unescape((entry.get("title") or "").strip())
        content_blocks = entry.get("content") or []
        content_html = content_blocks[0]["value"] if content_blocks else entry.get("summary", "")
        posts.append({"title": title, "selftext": extract_selftext(content_html)})
    return posts


def score_posts(posts: list[dict], analyzer: SentimentIntensityAnalyzer) -> dict:
    compound_scores: list[float] = []
    pos_scores: list[float] = []
    neu_scores: list[float] = []
    neg_scores: list[float] = []
    pos_count = neu_count = neg_count = 0
    selftext_count = 0
    top_title = posts[0]["title"] if posts else ""

    for post in posts:
        title = post["title"]
        selftext = post["selftext"]
        text = f"{title}\n\n{selftext}" if selftext else title
        if not text:
            continue
        if selftext:
            selftext_count += 1
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        compound_scores.append(compound)
        pos_scores.append(scores["pos"])
        neu_scores.append(scores["neu"])
        neg_scores.append(scores["neg"])
        # VADER author-recommended thresholds for discrete classification.
        if compound >= 0.05:
            pos_count += 1
        elif compound <= -0.05:
            neg_count += 1
        else:
            neu_count += 1

    def avg(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "posts_scored": len(compound_scores),
        "posts_with_selftext": selftext_count,
        "avg_compound": avg(compound_scores),
        "avg_pos": avg(pos_scores),
        "avg_neu": avg(neu_scores),
        "avg_neg": avg(neg_scores),
        "pos_count": pos_count,
        "neu_count": neu_count,
        "neg_count": neg_count,
        "top_post_title": top_title[:200],
    }


def load_existing() -> pd.DataFrame:
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        existing = pd.read_csv(CSV_PATH)
        if not existing.empty and "date" in existing.columns:
            return existing
    return pd.DataFrame(columns=COLUMNS)


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    posts = fetch_posts()
    if not posts:
        print(f"[reddit] no posts returned for r/{SUBREDDIT}, aborting")
        return 1

    analyzer = SentimentIntensityAnalyzer()
    aggregates = score_posts(posts, analyzer)

    today = date.today().isoformat()
    row = {"date": today, "subreddit": SUBREDDIT, **aggregates}

    existing = load_existing()
    existing = existing[existing["date"] != today]
    new_row = pd.DataFrame([row])
    combined = new_row if existing.empty else pd.concat([existing, new_row], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(CSV_PATH, index=False)

    latest = combined.iloc[-1]
    recent = combined.tail(30).to_dict(orient="records")
    JSON_PATH.write_text(json.dumps({
        "date": latest["date"],
        "subreddit": latest["subreddit"],
        "posts_scored": int(latest["posts_scored"]),
        "posts_with_selftext": int(latest["posts_with_selftext"]),
        "avg_compound": float(latest["avg_compound"]),
        "pos_count": int(latest["pos_count"]),
        "neu_count": int(latest["neu_count"]),
        "neg_count": int(latest["neg_count"]),
        "top_post_title": str(latest["top_post_title"]),
        "rows_total": int(len(combined)),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recent": recent,
    }, indent=2))

    print(
        f"[reddit] r/{SUBREDDIT} {today}: "
        f"{aggregates['posts_scored']} posts "
        f"({aggregates['posts_with_selftext']} with selftext), "
        f"avg_compound={aggregates['avg_compound']:.4f}, "
        f"pos/neu/neg = {aggregates['pos_count']}/{aggregates['neu_count']}/{aggregates['neg_count']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
