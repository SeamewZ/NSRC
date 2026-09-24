from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from celebcrypto.data.schemas import TWEET_COLUMNS


@dataclass
class ApifyTweetScraper:
    """Thin wrapper around Apify Tweet Scraper V2 outputs.

    The public release keeps this module dependency-light. If `apify-client` is
    installed and an API token is supplied, `run_actor` can call the hosted actor;
    otherwise, downstream code can read exported Apify JSON/CSV files with
    `normalize_export`.
    """

    token: str | None = None
    actor_id: str = "apidojo/tweet-scraper"
    default_input: dict[str, Any] = field(default_factory=dict)

    def run_actor(self, actor_input: dict[str, Any]) -> pd.DataFrame:
        if not self.token:
            raise RuntimeError("Apify token is required for live scraping.")
        try:
            from apify_client import ApifyClient
        except ImportError as exc:
            raise RuntimeError("Install `apify-client` to run live Apify scraping.") from exc
        client = ApifyClient(self.token)
        run = client.actor(self.actor_id).call(run_input={**self.default_input, **actor_input})
        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        return normalize_export(items)


def normalize_export(items: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    raw = items if isinstance(items, pd.DataFrame) else pd.DataFrame(items)
    records = []
    for _, row in raw.iterrows():
        author = row.get("author") if isinstance(row.get("author"), dict) else {}
        metrics = row.get("public_metrics") if isinstance(row.get("public_metrics"), dict) else {}
        text = row.get("text") or row.get("fullText") or row.get("content") or ""
        records.append(
            {
                "tweet_id": row.get("id") or row.get("tweet_id"),
                "timestamp": pd.to_datetime(
                    row.get("created_at") or row.get("createdAt") or row.get("timestamp"),
                    utc=True,
                    errors="coerce",
                ),
                "author_id": author.get("id") or row.get("author_id"),
                "author_name": author.get("name") or author.get("username") or row.get("author_name"),
                "text": text,
                "like_count": _num(row.get("likeCount"), metrics.get("like_count")),
                "retweet_count": _num(row.get("retweetCount"), metrics.get("retweet_count")),
                "reply_count": _num(row.get("replyCount"), metrics.get("reply_count")),
                "quote_count": _num(row.get("quoteCount"), metrics.get("quote_count")),
                "follower_count": _num(author.get("followers"), row.get("follower_count")),
                "verified": bool(author.get("isVerified") or author.get("verified") or False),
                "asset": infer_asset(text),
            }
        )
    return pd.DataFrame(records, columns=TWEET_COLUMNS)


def infer_asset(text: str) -> str | None:
    text_upper = str(text).upper()
    for asset in ["BTC", "ETH", "SOL", "DOGE", "TRUMP"]:
        if f"${asset}" in text_upper or asset in text_upper.split():
            return asset
    return None


def _num(*values: Any) -> float:
    for value in values:
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0

