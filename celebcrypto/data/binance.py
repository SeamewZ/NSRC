from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import requests

from celebcrypto.data.schemas import MARKET_COLUMNS


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


@dataclass
class BinanceClient:
    """Minimal Binance public REST client for 5-minute OHLCV acquisition."""

    quote_asset: str = "USDT"
    interval: str = "5m"
    limit: int = 1000

    def _symbol(self, asset: str) -> str:
        asset = asset.upper().replace("$", "")
        return f"{asset}{self.quote_asset}"

    def fetch_klines(
        self,
        asset: str,
        start: str | datetime,
        end: str | datetime,
    ) -> pd.DataFrame:
        start_ms = _to_ms(start)
        end_ms = _to_ms(end)
        symbol = self._symbol(asset)
        rows: list[list] = []
        cursor = start_ms
        while cursor < end_ms:
            params = {
                "symbol": symbol,
                "interval": self.interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": self.limit,
            }
            resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][0]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
        return _klines_to_frame(rows, asset)

    def fetch_many(
        self,
        assets: Iterable[str],
        start: str | datetime,
        end: str | datetime,
    ) -> pd.DataFrame:
        frames = [self.fetch_klines(asset, start, end) for asset in assets]
        if not frames:
            return pd.DataFrame(columns=MARKET_COLUMNS)
        return pd.concat(frames, ignore_index=True).sort_values(["asset", "timestamp"])


def _to_ms(value: str | datetime) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return int(ts.timestamp() * 1000)


def _klines_to_frame(rows: list[list], asset: str) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append(
            {
                "timestamp": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                "asset": asset.upper(),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )
    return pd.DataFrame(records, columns=MARKET_COLUMNS)

