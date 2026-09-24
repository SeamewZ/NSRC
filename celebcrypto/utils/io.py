from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, **kwargs)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True, **kwargs)
    if suffix == ".json":
        return pd.read_json(path, **kwargs)
    return pd.read_csv(path, **kwargs)


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False, **kwargs: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=index, **kwargs)
    elif suffix in {".jsonl", ".ndjson"}:
        df.to_json(path, orient="records", lines=True, force_ascii=False, **kwargs)
    elif suffix == ".json":
        df.to_json(path, orient="records", force_ascii=False, indent=2, **kwargs)
    else:
        df.to_csv(path, index=index, **kwargs)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}

