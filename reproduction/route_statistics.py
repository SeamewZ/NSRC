"""Statistics for the frozen NSRC route evaluation.

This script never selects a route from test data. It reads the route frozen by
the validation run in ``nsrc_safe_route_20260922`` and computes paired metrics
on the same test origins.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("runs/nsrc_safe_route_20260922")
FEATURE_ROOT = Path("runs/full_evaluation_20260921/data")
OUT = ROOT / "route_statistics"
DATASETS = ["celebrity", "news", "macro_crypto", "macro_equities"]
SEEDS = [2026, 2027, 2028]
METRICS = ["mse", "mae", "mape"]
BOOTSTRAPS = 49999


def losses(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    err = pred - y
    return np.stack(
        [err * err, np.abs(err), 100.0 * np.abs(err) / np.maximum(np.abs(y), 1e-8)],
        axis=-1,
    )


def bootstrap_weights(n_days: int, block_days: int, seed: int = 20260923) -> np.ndarray:
    rng = np.random.default_rng(seed + 17 * n_days + block_days)
    starts = rng.integers(0, n_days, size=(BOOTSTRAPS, int(np.ceil(n_days / block_days))))
    idx = ((starts[:, :, None] + np.arange(block_days)) % n_days).reshape(BOOTSTRAPS, -1)[:, :n_days]
    weights = np.zeros((BOOTSTRAPS, n_days), dtype=np.float64)
    np.add.at(weights, (np.arange(BOOTSTRAPS)[:, None], idx), 1.0)
    return weights


def paired_block_bootstrap(delta: np.ndarray, dates: pd.Series, block_days: int):
    day_values = pd.to_datetime(dates, utc=True).dt.floor("D")
    day = (day_values - day_values.min()).dt.days.to_numpy()
    n_days = int(day.max()) + 1
    counts = np.bincount(day, minlength=n_days).astype(float)
    totals = np.stack(
        [np.bincount(day, weights=delta[:, j], minlength=n_days) for j in range(delta.shape[1])],
        axis=1,
    )
    observed = delta.mean(axis=0)
    weights = bootstrap_weights(n_days, block_days)
    denominator = weights @ counts
    valid = denominator > 0
    samples = (weights[valid] @ totals) / denominator[valid, None]
    ci = np.quantile(samples, [0.025, 0.975], axis=0)
    p = (1.0 + (np.abs(samples - observed) >= np.abs(observed)).sum(axis=0)) / (len(samples) + 1.0)
    return observed, ci, p, int((counts > 0).sum()), int(len(samples))


def holm(values: pd.Series) -> pd.Series:
    out = np.full(len(values), np.nan, dtype=float)
    valid = values.notna().to_numpy()
    ids = np.flatnonzero(valid)
    if not len(ids):
        return pd.Series(out, index=values.index)
    order = ids[np.argsort(values.iloc[ids].to_numpy())]
    adjusted = np.maximum.accumulate(
        np.minimum(1.0, values.iloc[order].to_numpy() * (len(order) - np.arange(len(order))))
    )
    out[order] = adjusted
    return pd.Series(out, index=values.index)


def load_dataset(ds: str):
    packs = {}
    idx = None
    y = None
    for seed in SEEDS:
        folder = ROOT / ds / f"fixed_DLinear_s{seed}"
        arr = np.load(folder / "predictions.npz")
        current = pd.read_csv(folder / "test_index.csv")
        if idx is None:
            idx, y = current, arr["y"]
        else:
            if not idx.equals(current) or not np.array_equal(y, arr["y"]):
                raise ValueError(f"test index or target differs across seeds for {ds}")
        for name in ["DLinear", "ARM90", "NSRC", "NSRC_Route"]:
            packs[(name, seed)] = arr[name]
    return idx, y, packs


def average_pair(packs, left: str, right: str, y: np.ndarray):
    """Return one paired loss difference per origin after equal seed averaging."""
    left_loss = np.stack([losses(y, packs[(left, s)]).mean(axis=1) for s in SEEDS]).mean(axis=0)
    right_loss = np.stack([losses(y, packs[(right, s)]).mean(axis=1) for s in SEEDS]).mean(axis=0)
    return left_loss - right_loss


def make_significance():
    rows = []
    for ds in DATASETS:
        idx, y, packs = load_dataset(ds)
        comparisons = [("route_vs_dlinear", "DLinear", "NSRC_Route")]
        # The semantic mechanism is evaluated separately from the deployed route.
        comparisons.append(("semantic_vs_arm90", "ARM90", "NSRC"))
        dates = idx["timestamp"]
        for family, reference, candidate in comparisons:
            delta = average_pair(packs, reference, candidate, y)
            for block in [3, 7, 14]:
                obs, ci, p, active, valid = paired_block_bootstrap(delta, dates, block)
                for j, metric in enumerate(METRICS):
                    rows.append(
                        {
                            "dataset": ds,
                            "family": family,
                            "candidate": candidate,
                            "reference": reference,
                            "metric": metric,
                            "block_days": block,
                            "n_origins": len(idx),
                            "active_days": active,
                            "delta_reference_minus_candidate": obs[j],
                            "ci_low": ci[0, j],
                            "ci_high": ci[1, j],
                            "p_two_sided": p[j],
                            "valid_resamples": valid,
                            "route_frozen_on_validation": True,
                        }
                    )
    result = pd.DataFrame(rows)
    result["p_holm"] = np.nan
    for family in result["family"].unique():
        mask = (result["family"] == family) & (result["block_days"] == 7)
        result.loc[mask, "p_holm"] = holm(result.loc[mask, "p_two_sided"]).to_numpy()
    result.to_csv(OUT / "significance.csv", index=False)
    return result


def make_stability():
    horizon_rows, period_rows, volatility_rows, seed_rows = [], [], [], []
    for ds in DATASETS:
        idx, y, packs = load_dataset(ds)
        dates = pd.to_datetime(idx["timestamp"], utc=True)
        route_delta = average_pair(packs, "DLinear", "NSRC_Route", y)
        semantic_delta = average_pair(packs, "ARM90", "NSRC", y)
        for family, delta in [("route_vs_dlinear", route_delta), ("semantic_vs_arm90", semantic_delta)]:
            for h in range(24):
                d = delta[:, :,] if False else None
                # Recover per-horizon paired losses instead of using the pooled loss.
                if family == "route_vs_dlinear":
                    ref, cand = "DLinear", "NSRC_Route"
                else:
                    ref, cand = "ARM90", "NSRC"
                ref_loss = np.stack([losses(y, packs[(ref, s)])[:, h, :] for s in SEEDS]).mean(axis=0)
                cand_loss = np.stack([losses(y, packs[(cand, s)])[:, h, :] for s in SEEDS]).mean(axis=0)
                dh = ref_loss - cand_loss
                for j, metric in enumerate(METRICS):
                    horizon_rows.append(
                        {
                            "dataset": ds,
                            "family": family,
                            "horizon": h + 1,
                            "metric": metric,
                            "delta": dh[:, j].mean(),
                            "wins": int((dh[:, j] > 0).sum()),
                            "ties": int((np.abs(dh[:, j]) <= 1e-12).sum()),
                            "n_origins": len(idx),
                            "improved_fraction": float((dh[:, j] > 0).mean()),
                        }
                    )
            midpoint = dates.min() + (dates.max() - dates.min()) / 2
            periods = np.where(dates <= midpoint, "early", "late")
            for period in ["early", "late"]:
                keep = periods == period
                for j, metric in enumerate(METRICS):
                    period_rows.append(
                        {
                            "dataset": ds,
                            "family": family,
                            "period": period,
                            "metric": metric,
                            "delta": delta[keep, j].mean(),
                            "wins": int((delta[keep, j] > 0).sum()),
                            "n_origins": int(keep.sum()),
                            "improved_fraction": float((delta[keep, j] > 0).mean()),
                        }
                    )
        # Fixed volatility thresholds from non-test observations only.
        feature_dir = FEATURE_ROOT / ds
        base_idx = pd.read_csv(feature_dir / "index.csv")
        sigma = np.load(feature_dir / "features.npz")["sigma"]
        if len(sigma) != len(base_idx):
            raise ValueError(
                f"feature/index length mismatch for {ds}: {len(sigma)} vs {len(base_idx)}"
            )
        base_keys = base_idx["asset"].astype(str) + "|" + base_idx["timestamp"].astype(str)
        route_keys = idx["asset"].astype(str) + "|" + idx["timestamp"].astype(str)
        lookup = dict(zip(base_keys, sigma))
        test_sigma = np.asarray([lookup[k] for k in route_keys])
        train_sigma = sigma[~base_idx["test"].to_numpy()]
        cuts = np.quantile(train_sigma, [1 / 3, 2 / 3])
        buckets = np.digitize(test_sigma, cuts)
        for family, delta in [("route_vs_dlinear", route_delta), ("semantic_vs_arm90", semantic_delta)]:
            for b, label in enumerate(["low", "medium", "high"]):
                keep = buckets == b
                if not keep.any():
                    continue
                for j, metric in enumerate(METRICS):
                    volatility_rows.append(
                        {
                            "dataset": ds,
                            "family": family,
                            "volatility": label,
                            "metric": metric,
                            "delta": delta[keep, j].mean(),
                            "wins": int((delta[keep, j] > 0).sum()),
                            "n_origins": int(keep.sum()),
                            "improved_fraction": float((delta[keep, j] > 0).mean()),
                            "lower_cut": cuts[0],
                            "upper_cut": cuts[1],
                        }
                    )
        for method in ["DLinear", "ARM90", "NSRC", "NSRC_Route"]:
            for metric_index, metric in enumerate(METRICS):
                per_seed = []
                for seed in SEEDS:
                    per_seed.append(losses(y, packs[(method, seed)]).mean(axis=(0, 1))[metric_index])
                seed_rows.append(
                    {
                        "dataset": ds,
                        "method": method,
                        "metric": metric,
                        "mean": np.mean(per_seed),
                        "std": np.std(per_seed, ddof=1),
                        "min": np.min(per_seed),
                        "max": np.max(per_seed),
                        "seed_2026": per_seed[0],
                        "seed_2027": per_seed[1],
                        "seed_2028": per_seed[2],
                    }
                )
    horizon = pd.DataFrame(horizon_rows)
    period = pd.DataFrame(period_rows)
    volatility = pd.DataFrame(volatility_rows)
    seeds = pd.DataFrame(seed_rows)
    horizon.to_csv(OUT / "horizon_stability.csv", index=False)
    period.to_csv(OUT / "period_stability.csv", index=False)
    volatility.to_csv(OUT / "volatility_stability.csv", index=False)
    seeds.to_csv(OUT / "seed_stability.csv", index=False)
    return horizon, period, volatility, seeds


def make_figures(horizon, period, volatility):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharey=False, constrained_layout=True)
    for ax, ds in zip(axes.flat, DATASETS):
        z = horizon[(horizon.dataset == ds) & (horizon.family == "route_vs_dlinear") & (horizon.metric == "mae")]
        ax.axhline(0, color="black", lw=0.7)
        ax.plot(z.horizon, z.delta * 100, color="#228833", marker="o", ms=2, lw=1.1)
        ax.set_title(ds.replace("_", " ").title())
        ax.set_xlabel("Forecast step")
        ax.set_ylabel("DLinear MAE - route MAE (×100)")
        ax.grid(alpha=0.18)
    fig.savefig(OUT / "horizon_stability.pdf")
    fig.savefig(OUT / "horizon_stability.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True, constrained_layout=True)
    for ax, ds in zip(axes, DATASETS[:2]):
        z = period[(period.dataset == ds) & (period.family == "route_vs_dlinear") & (period.metric == "mae")]
        ax.bar(z.period, z.delta * 100, color="#228833")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(ds.title())
        ax.set_ylabel("DLinear MAE - route MAE (×100)")
    fig.savefig(OUT / "period_stability.pdf")
    fig.savefig(OUT / "period_stability.png", dpi=220)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    significance = make_significance()
    stability = make_stability()
    make_figures(*stability[:3])
    (OUT / "README.md").write_text(
        "Frozen-route paired significance and stability. Routes were selected on validation only. "
        "Significance uses calendar-day circular block bootstrap (49,999 resamples), with Holm "
        "correction within each comparison family at the 7-day block size. Volatility thresholds "
        "are computed from non-test observations.\n"
    )
    print(significance.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
