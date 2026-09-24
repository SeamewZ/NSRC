from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RuleAnnotation:
    relevance: str
    direction: str
    rationale: str


ASSET_PATTERN = re.compile(r"(\$?(BTC|ETH|SOL|DOGE|TRUMP)\b)", re.IGNORECASE)


def relevance_filter(text: str) -> bool:
    return bool(ASSET_PATTERN.search(str(text)))


def annotate_direction(text: str) -> RuleAnnotation:
    text_l = str(text).lower()
    if not relevance_filter(text_l):
        return RuleAnnotation("irrelevant", "Consolidation", "No tracked asset mention.")
    bullish_words = {
        "buy",
        "btd",
        "btfd",
        "bull",
        "moon",
        "pump",
        "support",
        "accumulate",
        "long",
        "breakout",
    }
    bearish_words = {
        "sell",
        "dump",
        "bear",
        "short",
        "crash",
        "rug",
        "scam",
        "fud",
        "liquidate",
    }
    pos = sum(word in text_l for word in bullish_words)
    neg = sum(word in text_l for word in bearish_words)
    if pos > neg:
        return RuleAnnotation("relevant", "Bullish", "Bullish lexical cues dominate.")
    if neg > pos:
        return RuleAnnotation("relevant", "Bearish", "Bearish lexical cues dominate.")
    return RuleAnnotation("relevant", "Consolidation", "No clear directional cue.")


def influencer_tag(
    follower_count: float,
    verified: bool,
    category: str | None = None,
    core_threshold: float = 1_000_000,
) -> str:
    if bool(verified) and float(follower_count or 0) >= core_threshold:
        return "Core"
    if str(category or "").lower() in {"ceo", "politician", "founder", "legislator"}:
        return "Core"
    return "Related"

