from __future__ import annotations

from dataclasses import dataclass


MARKET_COLUMNS = ["timestamp", "asset", "open", "high", "low", "close", "volume"]
TWEET_COLUMNS = [
    "tweet_id",
    "timestamp",
    "author_id",
    "author_name",
    "text",
    "like_count",
    "retweet_count",
    "reply_count",
    "quote_count",
    "follower_count",
    "verified",
    "asset",
]
CELEBRITY_COLUMNS = [
    "celebrity_id",
    "canonical_name",
    "aliases",
    "category",
    "follower_count",
    "verified",
    "reputation_score",
]
EVENT_COLUMNS = [
    "event_id",
    "timestamp",
    "usable_time",
    "asset",
    "celebrity_id",
    "canonical_name",
    "category",
    "direction",
    "text_cluster",
    "engagement",
    "priority_weight",
    "text",
]


@dataclass(frozen=True)
class EventKey:
    category: str
    direction: str
    text_cluster: str | int

    def text_key(self) -> tuple[str, str, str]:
        return (str(self.category), str(self.direction), str(self.text_cluster))

    def specific_key(self) -> tuple[str, str]:
        return (str(self.category), str(self.direction))

    def global_key(self) -> str:
        return str(self.direction)


def validate_columns(columns: list[str], required: list[str], name: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

