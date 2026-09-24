from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from celebcrypto.data.schemas import EVENT_COLUMNS, MARKET_COLUMNS, validate_columns
from celebcrypto.utils.time import ceil_to_hour, floor_to_hour, split_by_time, to_utc


def aggregate_5m_to_hourly(market: pd.DataFrame) -> pd.DataFrame:
    validate_columns(list(market.columns), MARKET_COLUMNS, "market")
    df = market.copy()
    df["timestamp"] = to_utc(df["timestamp"])
    df = df.dropna(subset=["timestamp", "asset"]).sort_values(["asset", "timestamp"])
    df["hour"] = floor_to_hour(df["timestamp"])
    grouped = df.groupby(["asset", "hour"], as_index=False)
    hourly = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return hourly.rename(columns={"hour": "timestamp"}).sort_values(["asset", "timestamp"])


def add_market_state_labels(
    market: pd.DataFrame,
    threshold: float = 0.001,
    price_col: str = "close",
) -> pd.DataFrame:
    df = market.copy().sort_values(["asset", "timestamp"])
    df["return_rate"] = df.groupby("asset")[price_col].pct_change().fillna(0.0)
    df["market_state"] = np.select(
        [df["return_rate"] > threshold, df["return_rate"] < -threshold],
        ["Bullish", "Bearish"],
        default="Consolidation",
    )
    return df


def dynamic_engagement_filter(tweets: pd.DataFrame, quantile: float = 0.90) -> pd.DataFrame:
    df = tweets.copy()
    df["timestamp"] = to_utc(df["timestamp"])
    df["engagement"] = (
        df.get("like_count", 0).fillna(0)
        + df.get("retweet_count", 0).fillna(0)
        + df.get("reply_count", 0).fillna(0)
        + df.get("quote_count", 0).fillna(0)
    )
    df["date"] = df["timestamp"].dt.date
    cutoff = df.groupby("date")["engagement"].transform(lambda s: s.quantile(quantile))
    return df[df["engagement"] >= cutoff].drop(columns=["date"])


def canonicalize_events(
    tweets: pd.DataFrame,
    celebrities: pd.DataFrame | None = None,
    default_category: str = "celebrity_tweet",
) -> pd.DataFrame:
    df = tweets.copy()
    df["timestamp"] = to_utc(df["timestamp"])
    if "engagement" not in df.columns:
        df["engagement"] = (
            df.get("like_count", 0).fillna(0)
            + df.get("retweet_count", 0).fillna(0)
            + df.get("reply_count", 0).fillna(0)
            + df.get("quote_count", 0).fillna(0)
        )
    df["category"] = df.get("category", default_category)
    df["direction"] = df.get("direction", df["text"].map(rule_direction))
    df["text_cluster"] = df.get("text_cluster", df["text"].map(stable_text_bucket))
    df["usable_time"] = ceil_to_hour(df["timestamp"])
    df["priority_weight"] = np.log1p(df["engagement"].astype(float).clip(lower=0))
    df["event_id"] = [
        stable_event_id(row.timestamp, row.author_id, row.text)
        for row in df[["timestamp", "author_id", "text"]].itertuples(index=False)
    ]
    df["celebrity_id"] = df.get("celebrity_id", df.get("author_id"))
    df["canonical_name"] = df.get("canonical_name", df.get("author_name"))
    if celebrities is not None and not celebrities.empty and "author_id" in df.columns:
        celeb = celebrities.copy()
        if "author_id" in celeb.columns:
            df = df.merge(celeb, on="author_id", how="left", suffixes=("", "_celeb"))
            df["celebrity_id"] = df["celebrity_id_celeb"].fillna(df["celebrity_id"])
            df["canonical_name"] = df["canonical_name_celeb"].fillna(df["canonical_name"])
    return df.reindex(columns=EVENT_COLUMNS)


def stable_event_id(timestamp: object, author_id: object, text: object) -> str:
    payload = f"{timestamp}|{author_id}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()[:16]


def stable_text_bucket(text: str, buckets: int = 128) -> str:
    digest = hashlib.md5(str(text).lower().encode("utf-8", errors="ignore")).hexdigest()
    return f"z{int(digest, 16) % buckets:03d}"


def rule_direction(text: str) -> str:
    text = str(text).lower()
    bullish = ["buy", "bull", "moon", "pump", "long", "accumulate", "btfd", "btd"]
    bearish = ["sell", "bear", "dump", "short", "crash", "fud", "scam"]
    b_pos = sum(word in text for word in bullish)
    b_neg = sum(word in text for word in bearish)
    if b_pos > b_neg:
        return "Bullish"
    if b_neg > b_pos:
        return "Bearish"
    return "Consolidation"


def attach_active_events(
    market: pd.DataFrame,
    events: pd.DataFrame,
    active_event_hours: int = 24,
) -> pd.DataFrame:
    m = market.copy()
    e = events.copy()
    m["timestamp"] = to_utc(m["timestamp"])
    e["usable_time"] = to_utc(e["usable_time"])
    rows = []
    for asset, asset_market in m.groupby("asset", sort=False):
        asset_events = e[(e["asset"].isna()) | (e["asset"].astype(str).str.upper() == asset.upper())]
        asset_events = asset_events.sort_values("usable_time")
        for row in asset_market.itertuples(index=False):
            cutoff = row.timestamp
            start = cutoff - pd.Timedelta(hours=active_event_hours)
            active = asset_events[
                (asset_events["usable_time"] <= cutoff) & (asset_events["usable_time"] > start)
            ]
            if active.empty:
                rows.append({**row._asdict(), **empty_event_payload()})
            else:
                selected = active.sort_values("priority_weight", ascending=False).iloc[0]
                rows.append({**row._asdict(), **event_payload(selected)})
    return pd.DataFrame(rows)


def empty_event_payload() -> dict[str, object]:
    return {
        "event_id": "none",
        "event_category": "none",
        "event_direction": "Consolidation",
        "event_text_cluster": "none",
        "event_weight": 0.0,
    }


def event_payload(row: pd.Series) -> dict[str, object]:
    return {
        "event_id": row.get("event_id", "none"),
        "event_category": row.get("category", "none"),
        "event_direction": row.get("direction", "Consolidation"),
        "event_text_cluster": row.get("text_cluster", "none"),
        "event_weight": float(row.get("priority_weight", 1.0) or 1.0),
    }


def add_chronological_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    out = df.copy()
    out["split"] = split_by_time(out, timestamp_col, train_end=train_end, val_end=val_end)
    return out


def write_schema_reference(path: str | Path) -> None:
    Path(path).write_text(
        "Market columns: " + ", ".join(MARKET_COLUMNS) + "\n"
        "Event columns: " + ", ".join(EVENT_COLUMNS) + "\n",
        encoding="utf-8",
    )

