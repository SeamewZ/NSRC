from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from celebcrypto.annotation.rules import influencer_tag
from celebcrypto.utils.io import ensure_dir, write_json, write_table


ASSET_PATTERNS = {
    "BTC": [r"\$BTC\b", r"\bBTC\b", r"\bBitcoin\b"],
    "ETH": [r"\$ETH\b", r"\bETH\b", r"\bEthereum\b"],
    "SOL": [r"\$SOL\b", r"\bSOL\b", r"\bSolana\b"],
    "DOGE": [r"\$DOGE\b", r"\bDOGE\b", r"\bDogecoin\b"],
    "TRUMP": [r"\$TRUMP\b", r"\bTRUMP\b", r"\bOfficial Trump\b"],
}

AUTHOR_DEFAULT_ASSET = {
    "michael saylor": "BTC",
    "vitalik buterin": "ETH",
    "elon musk": "DOGE",
    "donald trump": "TRUMP",
    "eric trump": "TRUMP",
}


@dataclass(frozen=True)
class BundlePaths:
    root: Path

    @property
    def ohlcv(self) -> Path:
        return self.root / "ohlcv"

    @property
    def events(self) -> Path:
        return self.root / "events"

    @property
    def tweet_core(self) -> Path:
        return self.root / "tweet_core"

    @property
    def tweet_related(self) -> Path:
        return self.root / "tweet_related"

    @property
    def kline_photo(self) -> Path:
        return self.root / "kline_photo"

    @property
    def model_prediction(self) -> Path:
        return self.root / "model_prediction"


def load_bundled_market(bundle_root: str | Path) -> pd.DataFrame:
    paths = BundlePaths(Path(bundle_root))
    frames = []
    for file in sorted(paths.ohlcv.glob("*.csv")):
        asset = asset_from_name(file.name)
        symbol = symbol_from_name(file.name)
        df = pd.read_csv(file)
        df = df.rename(columns={"open_time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["asset"] = asset
        df["symbol"] = symbol
        df["source_file"] = str(file.relative_to(paths.root))
        frames.append(df[["timestamp", "asset", "symbol", "open", "high", "low", "close", "volume", "source_file"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_bundled_events(bundle_root: str | Path) -> pd.DataFrame:
    paths = BundlePaths(Path(bundle_root))
    rows = []
    for file in sorted(paths.events.glob("*.json")):
        data = json.loads(file.read_text(encoding="utf-8"))
        analysis = data.get("llm_analysis") or {}
        entities = analysis.get("entities") or {}
        content = data.get("content") or ""
        title = data.get("title") or ""
        asset_mentions = sorted(set(analysis.get("cryptocurrencies") or []) | set(infer_assets(f"{title}\n{content}")))
        rows.append(
            {
                "event_id": stable_id(file.stem),
                "timestamp": pd.to_datetime(data.get("time"), utc=True, errors="coerce"),
                "title": title,
                "summary": data.get("summary") or "",
                "author": data.get("author") or "",
                "content": content,
                "url": data.get("url") or "",
                "impact_sentiment": analysis.get("impact_sentiment") or "Consolidation",
                "reasoning": analysis.get("reasoning") or "",
                "cryptocurrencies": "|".join(asset_mentions),
                "persons": "|".join(entities.get("persons") or []),
                "companies": "|".join(entities.get("companies") or []),
                "organizations": "|".join(entities.get("organizations") or []),
                "source_file": str(file.relative_to(paths.root)),
            }
        )
    return pd.DataFrame(rows)


def load_bundled_tweets(bundle_root: str | Path) -> pd.DataFrame:
    paths = BundlePaths(Path(bundle_root))
    rows: list[dict[str, Any]] = []
    for file in sorted(paths.tweet_core.glob("*_clean.txt")):
        rows.extend(parse_tweet_file(file, paths.root, source_group="core", asset_hint=None))
    for file in sorted(paths.tweet_related.glob("*/*_clean.txt")):
        rows.extend(
            parse_tweet_file(
                file,
                paths.root,
                source_group="related",
                asset_hint=asset_from_name(file.parent.name),
            )
        )
    return pd.DataFrame(rows)


def parse_tweet_file(
    file: Path,
    bundle_root: Path,
    source_group: str,
    asset_hint: str | None,
) -> list[dict[str, Any]]:
    text = file.read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(r"\[TWEET START\](.*?)\[TWEET END\]", text, flags=re.S)
    rows = []
    file_canonical_name = canonical_name_from_tweet_file(file, source_group)
    for idx, block in enumerate(blocks):
        tweet_text = extract_between(block, 'Text: "', '"\n---\n[METADATA]') or ""
        tweet_text = tweet_text.replace('""', '"').strip()
        metadata = parse_bullet_section(block, "[METADATA]", "[ANALYZEDATA]")
        analysis = parse_bullet_section(block, "[ANALYZEDATA]", None)
        author_username = metadata.get("author_username") or slugify(file_canonical_name)
        canonical_name = file_canonical_name if source_group == "core" else author_username
        asset_mentions = infer_assets(tweet_text)
        asset = asset_hint or primary_asset(asset_mentions, canonical_name)
        created_at = pd.to_datetime(metadata.get("created_at"), utc=True, errors="coerce")
        follower_count = to_number(metadata.get("author_followers"))
        row = {
            "tweet_id": stable_id(f"{file}:{idx}:{created_at}:{tweet_text[:80]}"),
            "timestamp": created_at,
            "author_id": slugify(author_username),
            "author_name": canonical_name,
            "author_username": author_username,
            "text": tweet_text,
            "like_count": to_number(metadata.get("likes")),
            "retweet_count": to_number(metadata.get("retweets")),
            "reply_count": to_number(metadata.get("replies")),
            "quote_count": to_number(metadata.get("quotes")),
            "views": to_number(metadata.get("views")),
            "follower_count": follower_count,
            "verified": follower_count >= 1_000_000,
            "asset": asset,
            "asset_mentions": "|".join(asset_mentions),
            "source_group": source_group,
            "source_file": str(file.relative_to(bundle_root)),
            "label": normalize_direction(analysis.get("label")),
            "key_word_used": analysis.get("key_word_used") or "",
            "reasoning": analysis.get("reasoning") or "",
            "influencer_tag": influencer_tag(follower_count, follower_count >= 1_000_000),
        }
        rows.append(row)
    return rows


def load_bundled_celebrities(tweets: pd.DataFrame) -> pd.DataFrame:
    if tweets.empty:
        return pd.DataFrame()
    grouped = tweets.groupby(["author_id", "author_name"], dropna=False)
    rows = []
    for (author_id, author_name), df in grouped:
        follower_count = float(df["follower_count"].max())
        rows.append(
            {
                "celebrity_id": author_id,
                "author_id": author_id,
                "canonical_name": author_name,
                "aliases": "|".join(sorted(set(df["author_username"].astype(str)))),
                "category": "core" if (df["source_group"] == "core").any() else "related",
                "follower_count": follower_count,
                "verified": follower_count >= 1_000_000,
                "reputation_score": 1.0 if follower_count >= 1_000_000 else 0.5,
                "tweet_count": int(len(df)),
            }
        )
    return pd.DataFrame(rows).sort_values(["category", "follower_count"], ascending=[True, False])


def load_kline_image_manifest(bundle_root: str | Path) -> pd.DataFrame:
    paths = BundlePaths(Path(bundle_root))
    rows = []
    pattern = re.compile(r"(?P<prefix>.+?)_(?P<symbol>[A-Z]+USDT)_(?P<start>\d{8}_\d{6})_(?P<end>\d{8}_\d{6})\.png$")
    for file in sorted(paths.kline_photo.glob("*/*.png")):
        m = pattern.search(file.name)
        start = end = symbol = None
        if m:
            symbol = m.group("symbol")
            start = pd.to_datetime(m.group("start"), format="%Y%m%d_%H%M%S", utc=True)
            end = pd.to_datetime(m.group("end"), format="%Y%m%d_%H%M%S", utc=True)
        rows.append(
            {
                "image_path": str(file.relative_to(paths.root)),
                "asset": asset_from_name(file.name),
                "symbol": symbol or symbol_from_name(file.name),
                "window_start": start,
                "window_end": end,
                "source_dir": file.parent.name,
            }
        )
    return pd.DataFrame(rows)


def load_prediction_manifest(bundle_root: str | Path) -> pd.DataFrame:
    paths = BundlePaths(Path(bundle_root))
    rows = []
    for file in sorted(paths.model_prediction.glob("*.csv")):
        header = pd.read_csv(file, nrows=0).columns.tolist()
        rows.append(
            {
                "prediction_file": str(file.relative_to(paths.root)),
                "rows": count_csv_rows(file),
                "columns": len(header),
                "schema": "policy_impact" if "policy_id" in header else "forecast_prediction",
                "model_hint": model_hint_from_prediction_file(file.name),
            }
        )
    return pd.DataFrame(rows)


def export_standardized_bundle(bundle_root: str | Path, output_dir: str | Path = "data/raw") -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    market = load_bundled_market(bundle_root)
    events = load_bundled_events(bundle_root)
    tweets = load_bundled_tweets(bundle_root)
    celebrities = load_bundled_celebrities(tweets)
    images = load_kline_image_manifest(bundle_root)
    prediction_manifest = load_prediction_manifest(bundle_root)

    write_table(market, output_dir / "market_5m.csv")
    write_table(tweets, output_dir / "tweets.csv")
    write_table(celebrities, output_dir / "celebrities.csv")
    write_table(events, output_dir / "news_events.csv")
    write_table(images, output_dir / "kline_images.csv")
    write_table(prediction_manifest, output_dir / "model_prediction_manifest.csv")
    manifest = summarize_bundle(bundle_root, market, tweets, events, images, prediction_manifest)
    write_json(manifest, output_dir / "bundle_manifest.json")
    return manifest


def summarize_bundle(
    bundle_root: str | Path,
    market: pd.DataFrame | None = None,
    tweets: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    images: pd.DataFrame | None = None,
    prediction_manifest: pd.DataFrame | None = None,
) -> dict[str, Any]:
    root = Path(bundle_root)
    market = market if market is not None else load_bundled_market(root)
    tweets = tweets if tweets is not None else load_bundled_tweets(root)
    events = events if events is not None else load_bundled_events(root)
    images = images if images is not None else load_kline_image_manifest(root)
    prediction_manifest = prediction_manifest if prediction_manifest is not None else load_prediction_manifest(root)
    return {
        "bundle_root": str(root),
        "market_rows": int(len(market)),
        "market_assets": sorted(market["asset"].dropna().unique().tolist()) if not market.empty else [],
        "tweet_rows": int(len(tweets)),
        "tweet_core_files": len(list((root / "tweet_core").glob("*_clean.txt"))),
        "tweet_related_files": len(list((root / "tweet_related").glob("*/*_clean.txt"))),
        "event_json_files": len(list((root / "events").glob("*.json"))),
        "event_rows": int(len(events)),
        "kline_images": int(len(images)),
        "prediction_files": int(len(prediction_manifest)),
        "market_time_min": str(market["timestamp"].min()) if not market.empty else None,
        "market_time_max": str(market["timestamp"].max()) if not market.empty else None,
    }


def parse_bullet_section(block: str, start_marker: str, end_marker: str | None) -> dict[str, str]:
    if start_marker not in block:
        return {}
    section = block.split(start_marker, 1)[1]
    if end_marker and end_marker in section:
        section = section.split(end_marker, 1)[0]
    out: dict[str, str] = {}
    current_key: str | None = None
    for raw_line in section.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.strip() == "---":
            continue
        if line.lstrip().startswith("- ") and ":" in line:
            key, value = line.lstrip()[2:].split(":", 1)
            current_key = snake_key(key)
            out[current_key] = value.strip()
        elif current_key:
            out[current_key] = f"{out[current_key]}\n{line.strip()}".strip()
    return out


def extract_between(text: str, start: str, end: str) -> str | None:
    if start not in text:
        return None
    tail = text.split(start, 1)[1]
    if end not in tail:
        return tail
    return tail.split(end, 1)[0]


def infer_assets(text: str) -> list[str]:
    found = []
    for asset, patterns in ASSET_PATTERNS.items():
        if any(re.search(pattern, str(text), flags=re.I) for pattern in patterns):
            found.append(asset)
    return found


def primary_asset(asset_mentions: list[str], canonical_name: str | None = None) -> str | None:
    if asset_mentions:
        return asset_mentions[0]
    if canonical_name:
        return AUTHOR_DEFAULT_ASSET.get(canonical_name.lower())
    return None


def asset_from_name(name: str) -> str | None:
    upper = name.upper()
    for asset in ["BTC", "ETH", "SOL", "DOGE", "TRUMP"]:
        if f"({asset})" in upper or f"_{asset}USDT" in upper or upper.startswith(asset):
            return asset
    return None


def symbol_from_name(name: str) -> str | None:
    m = re.search(r"([A-Z]+USDT)", name.upper())
    return m.group(1) if m else None


def canonical_name_from_tweet_file(file: Path, source_group: str) -> str:
    if source_group == "related":
        return file.stem.replace("_clean", "")
    stem = file.stem.replace("_tweets_clean", "")
    stem = stem.replace("_binance", "")
    return stem.replace("_", " ")


def normalize_direction(value: Any) -> str:
    value = str(value or "Consolidation").strip()
    mapping = {"bullish": "Bullish", "bearish": "Bearish", "neutral": "Consolidation", "consolidation": "Consolidation"}
    return mapping.get(value.lower(), value)


def to_number(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def snake_key(key: str) -> str:
    key = key.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def slugify(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def stable_id(payload: Any) -> str:
    return hashlib.sha1(str(payload).encode("utf-8", errors="ignore")).hexdigest()[:16]


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return max(0, sum(1 for _ in f) - 1)


def model_hint_from_prediction_file(name: str) -> str:
    lower = name.lower()
    for model in ["patchtst", "timellm", "dlinear", "itransformer", "btc"]:
        if model in lower:
            return model
    return Path(name).stem
