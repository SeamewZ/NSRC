from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WindowedDataset:
    x: np.ndarray
    y: np.ndarray
    base_index: pd.DataFrame
    mean: np.ndarray
    std: np.ndarray
    feature_columns: list[str]
    target_column: str


def make_windows(
    hourly: pd.DataFrame,
    lookback: int,
    horizon: int,
    feature_columns: list[str],
    target_column: str = "close",
    normalization: str = "zscore",
) -> WindowedDataset:
    frames_x, frames_y, index_rows = [], [], []
    stats_mean, stats_std = [], []
    for asset, df_asset in hourly.sort_values(["asset", "timestamp"]).groupby("asset", sort=False):
        values = df_asset[feature_columns].astype(float).to_numpy()
        target = df_asset[target_column].astype(float).to_numpy()
        if "split" in df_asset.columns and (df_asset["split"].astype(str) == "train").any():
            train_values = df_asset.loc[df_asset["split"].astype(str) == "train", feature_columns]
            mean, std = _stats(train_values.astype(float).to_numpy(), normalization)
        else:
            mean, std = _stats(values, normalization)
        values_norm = _normalize(values, mean, std, normalization)
        target_mean = mean[feature_columns.index(target_column)]
        target_std = std[feature_columns.index(target_column)]
        target_norm = _normalize_target(target, target_mean, target_std, normalization)
        for end in range(lookback - 1, len(df_asset) - horizon):
            start = end - lookback + 1
            y_start = end + 1
            y_end = end + horizon + 1
            frames_x.append(values_norm[start : end + 1])
            frames_y.append(target_norm[y_start:y_end])
            origin = df_asset.iloc[end]
            index_rows.append(
                {
                    "asset": asset,
                    "timestamp": origin["timestamp"],
                    "event_id": origin.get("event_id", "none"),
                    "event_category": origin.get("event_category", "none"),
                    "event_direction": origin.get("event_direction", "Consolidation"),
                    "event_text_cluster": origin.get("event_text_cluster", "none"),
                    "event_weight": origin.get("event_weight", 0.0),
                    "split": origin.get("split", "unknown"),
                }
            )
        stats_mean.append(mean)
        stats_std.append(std)
    x = np.stack(frames_x).astype(np.float32) if frames_x else np.empty((0, lookback, len(feature_columns)))
    y = np.stack(frames_y).astype(np.float32) if frames_y else np.empty((0, horizon))
    return WindowedDataset(
        x=x,
        y=y,
        base_index=pd.DataFrame(index_rows),
        mean=np.mean(np.stack(stats_mean), axis=0) if stats_mean else np.zeros(len(feature_columns)),
        std=np.mean(np.stack(stats_std), axis=0) if stats_std else np.ones(len(feature_columns)),
        feature_columns=feature_columns,
        target_column=target_column,
    )


def _stats(values: np.ndarray, normalization: str) -> tuple[np.ndarray, np.ndarray]:
    if normalization == "minmax":
        vmin = np.nanmin(values, axis=0)
        vmax = np.nanmax(values, axis=0)
        return vmin, np.maximum(vmax - vmin, 1e-8)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    return mean, np.maximum(std, 1e-8)


def _normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray, normalization: str) -> np.ndarray:
    return (values - mean) / std if normalization == "zscore" else (values - mean) / std


def _normalize_target(
    target: np.ndarray,
    mean: float,
    std: float,
    normalization: str,
) -> np.ndarray:
    return (target - mean) / std if normalization == "zscore" else (target - mean) / std


def split_indices(index: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        split: index.index[index["split"].astype(str).str.lower() == split].to_numpy()
        for split in ["train", "val", "test"]
    }
