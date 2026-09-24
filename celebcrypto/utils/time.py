from __future__ import annotations

import pandas as pd


def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def ceil_to_hour(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.ceil("h")


def floor_to_hour(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.floor("h")


def split_by_time(
    df: pd.DataFrame,
    timestamp_col: str,
    train_end: str,
    val_end: str,
) -> pd.Series:
    ts = pd.to_datetime(df[timestamp_col], utc=True)
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    val_end_ts = pd.Timestamp(val_end, tz="UTC")
    split = pd.Series("test", index=df.index, dtype="object")
    split[ts <= train_end_ts] = "train"
    split[(ts > train_end_ts) & (ts <= val_end_ts)] = "val"
    return split

