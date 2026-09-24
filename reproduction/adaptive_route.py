"""Leakage-safe validation-selected candidate router for NSRC.

At each event origin, four candidate paths are available: Persistence, the
frozen DLinear path, ARM90, and semantic NSRC. The router can select a path
from this set or from fixed, predeclared blends of Persistence with ARM90 or
NSRC. All route hyperparameters and any fixed candidate are selected on a
pre-test validation interval and frozen before test evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reproduction.causal_gate import (
    BASE,
    H,
    OUTROOT,
    _effect,
    _past,
    _times,
    causal_gain,
    load,
)

SAFE_ROUTE_CANDIDATES = [
    {"name": "s10_w20_m0", "min_events": 10, "window_events": 20, "margin": 0.00, "anchor": "Persistence"},
    {"name": "s10_w30_m0", "min_events": 10, "window_events": 30, "margin": 0.00, "anchor": "Persistence"},
    {"name": "s10_w60_m0", "min_events": 10, "window_events": 60, "margin": 0.00, "anchor": "Persistence"},
    {"name": "s10_w20_m1", "min_events": 10, "window_events": 20, "margin": 0.01, "anchor": "Persistence"},
    {"name": "s10_w30_m1", "min_events": 10, "window_events": 30, "margin": 0.01, "anchor": "Persistence"},
    {"name": "s10_w60_m1", "min_events": 10, "window_events": 60, "margin": 0.01, "anchor": "Persistence"},
    {"name": "s10_w20_m5", "min_events": 10, "window_events": 20, "margin": 0.05, "anchor": "Persistence"},
    {"name": "s10_w30_m5", "min_events": 10, "window_events": 30, "margin": 0.05, "anchor": "Persistence"},
    {"name": "s10_w60_m5", "min_events": 10, "window_events": 60, "margin": 0.05, "anchor": "Persistence"},
    {"name": "s10_w30_m0_h", "min_events": 10, "window_events": 30, "margin": 0.00, "anchor": "Persistence", "horizonwise": True},
    {"name": "s10_w60_m0_h", "min_events": 10, "window_events": 60, "margin": 0.00, "anchor": "Persistence", "horizonwise": True},
    {"name": "s10_w60_m5_h", "min_events": 10, "window_events": 60, "margin": 0.05, "anchor": "Persistence", "horizonwise": True},
]


BLEND_ALPHAS = (0.10, 0.25, 0.50, 0.75)
FIXED_CANDIDATES = ("Persistence", "DLinear", "ARM90", "NSRC") + tuple(
    name for alpha in BLEND_ALPHAS
    for name in (
        f"P+ARM{int(round(alpha * 100)):02d}",
        f"P+NSRC{int(round(alpha * 100)):02d}",
    )
)

def candidate_paths(persistence, base, arm, semantic):
    """Return fixed, predeclared causal candidates for one event origin."""
    out = {
        "Persistence": persistence,
        "DLinear": base,
        "ARM90": arm,
        "NSRC": semantic,
    }
    for alpha in BLEND_ALPHAS:
        tag = f"{int(round(alpha * 100)):02d}"
        out[f"P+ARM{tag}"] = (1.0 - alpha) * persistence + alpha * arm
        out[f"P+NSRC{tag}"] = (1.0 - alpha) * persistence + alpha * semantic
    return out


def route_stream(ds, backbone, seed, route, until=None, collect_start=None, y_override=None):
    idx, y, base, events, sigma, manifest = load(ds, backbone, seed)
    if y_override is not None:
        y = np.asarray(y_override, dtype=float)
    ts, end = _times(idx)
    order = np.flatnonzero(idx.memory_eligible.values & (idx.events.values > 0))
    order = order[np.argsort(ts[order], kind="mergesort")]
    if until is not None:
        order = order[ts[order] < until]
    assets = idx.asset.values
    scale = sigma[:, None] * np.sqrt(np.arange(1, H + 1))
    delta = y - base
    values = np.clip(delta / scale, -5, 5)

    persistence = np.ones_like(base)
    arm = np.array(base, copy=True)
    semantic = np.array(base, copy=True)
    arm_effect = np.zeros_like(base)
    semantic_effect = np.zeros_like(base)
    route_pred = np.array(base, copy=True)
    processed = []
    history = []
    traces = []

    for q in order:
        past = _past(idx, processed, end, ts[q], 90.0)
        ae = _effect(q, past, "agnostic", idx, events, assets, values, sigma, ts)
        se = _effect(q, past, "semantic", idx, events, assets, values, sigma, ts)
        arm_effect[q] = ae
        semantic_effect[q] = se
        arm_known = past[np.any(np.abs(arm_effect[past]) > 1e-14, axis=1)]
        sem_known = past[np.any(np.abs(semantic_effect[past]) > 1e-14, axis=1)]
        ga = causal_gain(delta[arm_known], arm_effect[arm_known], assets[arm_known], 10.0)
        gs = causal_gain(delta[sem_known], semantic_effect[sem_known], assets[sem_known], 10.0)
        arm[q] = base[q] + ga * ae
        semantic[q] = base[q] + gs * se
        candidates = candidate_paths(persistence[q], base[q], arm[q], semantic[q])

        old = [r for r in history if r["end"] < ts[q]]
        old = old[-int(route["window_events"]):]
        losses = {}
        anchor = route.get("anchor", "Persistence")
        fixed_candidate = route.get("fixed_candidate")
        winner = fixed_candidate or anchor
        if winner not in candidates:
            raise ValueError(f"Unknown route candidate: {winner}")
        route_pred[q] = candidates[winner]
        if fixed_candidate is not None:
            pass
        elif len(old) >= int(route["min_events"]):
            if route.get("horizonwise", False):
                selected = []
                for h in range(H):
                    h_losses = {name: float(np.mean([r["loss_vectors"][name][h] for r in old])) for name in candidates}
                    wh = min(h_losses, key=lambda name: (h_losses[name], name))
                    if wh != anchor and h_losses[wh] > h_losses[anchor] * (1.0 - float(route["margin"])):
                        wh = anchor
                    route_pred[q, h] = candidates[wh][h]
                    selected.append(wh)
                winner = "horizonwise:" + ",".join(selected)
            else:
                for name in candidates:
                    losses[name] = float(np.mean([r["losses"][name] for r in old]))
                winner = min(losses, key=lambda name: (losses[name], name))
                if winner != anchor and losses[winner] > losses[anchor] * (1.0 - float(route["margin"])):
                    winner = anchor
        if fixed_candidate is None and not route.get("horizonwise", False):
            route_pred[q] = candidates[winner]

        point_loss_vectors = {name: np.abs(y[q] - pred) for name, pred in candidates.items()}
        point_losses = {name: float(np.mean(vector)) for name, vector in point_loss_vectors.items()}
        history.append({
            "q": int(q), "ts": float(ts[q]), "end": float(end[q]),
            "losses": point_losses,
            "loss_vectors": {name: vector.tolist() for name, vector in point_loss_vectors.items()},
        })
        traces.append({
            "row": int(q),
            "origin": idx.iloc[q].timestamp,
            "asset": assets[q],
            "memory_n": int(len(past)),
            "arm_gain": float(ga),
            "semantic_gain": float(gs),
            "route_history_n": int(len(old)),
            "route": winner,
            "route_arm_mae": float(losses.get("ARM90", np.nan)),
            "route_winner_mae": float(losses.get(winner, np.nan)),
        })
        processed.append(int(q))

    if collect_start is None:
        collect = order
    else:
        collect = order[ts[order] >= collect_start]
    collect = collect[idx.iloc[collect].events.values > 0]
    # Validation origins must have complete target windows before the frozen cut.
    if until is not None:
        collect = collect[end[collect] < until]
    return {
        "idx": idx,
        "y": y,
        "base": base,
        "persistence": persistence,
        "dlinear": base,
        "arm": arm,
        "semantic": semantic,
        "route": route_pred,
        "ts": ts,
        "end": end,
        "rows": collect,
        "traces": traces,
    }


def candidate_arrays(stream):
    return candidate_paths(stream["persistence"], stream["base"], stream["arm"], stream["semantic"])

def score_stream(stream):
    rows = stream["rows"]
    y = stream["y"]
    output = {}
    candidates = [(name, arr) for name, arr in candidate_arrays(stream).items()]
    candidates.append(("NSRC-Route", stream["route"]))
    for name, pred in candidates:
        e = y[rows] - pred[rows]
        output[name] = {
            "mse": float(np.mean(e * e)),
            "mae": float(np.mean(np.abs(e))),
            "mape": float(np.mean(np.abs(e) / np.maximum(np.abs(y[rows]), 1e-9)) * 100.0),
            "n": int(len(rows)),
        }
    return output


def select(ds, backbone, seed):
    manifest = json.loads((BASE / "data" / ds / "manifest.json").read_text())
    test_start = pd.Timestamp(manifest["test_start"], tz="UTC").value / 86400e9
    val_start = test_start - 30.0
    rows = []
    for route in SAFE_ROUTE_CANDIDATES:
        stream = route_stream(ds, backbone, seed, route, until=test_start, collect_start=val_start)
        scores = score_stream(stream)
        rows.append(dict(
            dataset=ds,
            backbone=backbone,
            seed=seed,
            **route,
            **{f"{name}_{metric}": value[metric] for name, value in scores.items()
               for metric in ["mse", "mae", "mape", "n"]},
        ))
    frame = pd.DataFrame(rows)
    best = frame.sort_values(["NSRC-Route_mae", "NSRC-Route_mse", "name"]).iloc[0].to_dict()
    best = {key: (value.item() if hasattr(value, "item") else value) for key, value in best.items()}
    return frame, best


def select_fixed(ds, backbone, seed):
    """Score the predeclared fixed candidates on leakage-safe validation."""
    manifest = json.loads((BASE / "data" / ds / "manifest.json").read_text())
    test_start = pd.Timestamp(manifest["test_start"], tz="UTC").value / 86400e9
    val_start = test_start - 30.0
    reference = SAFE_ROUTE_CANDIDATES[2]
    stream = route_stream(ds, backbone, seed, reference, until=test_start, collect_start=val_start)
    scores = score_stream(stream)
    rows = [dict(dataset=ds, backbone=backbone, seed=seed, method=name,
                 validation_mse=scores[name]["mse"], validation_mae=scores[name]["mae"],
                 validation_mape=scores[name]["mape"], n=scores[name]["n"])
             for name in FIXED_CANDIDATES]
    return pd.DataFrame(rows), stream


def evaluate(ds, backbone, seed, route):
    stream = route_stream(ds, backbone, seed, route)
    test = stream["idx"].test.values
    rows = stream["rows"]
    rows = rows[test[rows]]
    y = stream["y"]
    result = []
    for name, pred in list(candidate_arrays(stream).items()) + [("NSRC-Route", stream["route"])]:

        e = y[rows] - pred[rows]
        result.append(dict(
            dataset=ds,
            backbone=backbone,
            seed=seed,
            method=name,
            n=int(len(rows)),
            mse=float(np.mean(e * e)),
            mae=float(np.mean(np.abs(e))),
            mape=float(np.mean(np.abs(e) / np.maximum(np.abs(y[rows]), 1e-9)) * 100.0),
        ))
    return pd.DataFrame(result), stream, rows


def verify(ds, backbone, seed, route, stream):
    idx = stream["idx"]
    quiet = idx.events.values == 0
    no_event_identity = bool(np.array_equal(stream["arm"][quiet], stream["base"][quiet]) and np.array_equal(stream["semantic"][quiet], stream["base"][quiet]) and np.array_equal(stream["route"][quiet], stream["base"][quiet]))
    manifest = json.loads((BASE / "data" / ds / "manifest.json").read_text())
    cut = pd.Timestamp(manifest["test_start"], tz="UTC").value / 86400e9 + 14.0
    changed = stream["y"].copy()
    changed[stream["end"] >= cut] *= 1.5
    intervention = route_stream(ds, backbone, seed, route, y_override=changed)
    before = stream["ts"] < cut
    before_candidates = candidate_arrays(stream)
    after_candidates = candidate_arrays(intervention)
    future_target_invariant = bool(
        all(np.array_equal(before_candidates[name][before], after_candidates[name][before])
            for name in before_candidates)
        and np.array_equal(stream["route"][before], intervention["route"][before])
    )
    mature_memory_only = all(float(trace["route_history_n"]) >= 0 for trace in stream["traces"])
    checks = {
        "dataset": ds,
        "backbone": backbone,
        "seed": seed,
        "route": route,
        "selection_validation_only": True,
        "test_start": manifest["test_start"],
        "future_target_intervention": future_target_invariant,
        "no_event_identity": no_event_identity,
        "mature_memory_only": mature_memory_only,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if not all(checks[key] for key in [
        "future_target_intervention",
        "no_event_identity",
        "mature_memory_only",
    ]):
        raise AssertionError(checks)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", default="DLinear")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--route-name", default=None)
    parser.add_argument("--fixed-candidate", default=None, choices=FIXED_CANDIDATES)
    args = parser.parse_args()
    outroot = Path("runs/nsrc_safe_route_20260922")
    outroot.mkdir(parents=True, exist_ok=True)
    if args.select:
        frame, best = select(args.dataset, args.backbone, args.seed)
        out = outroot / args.dataset
        out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out / "validation_candidates.csv", index=False)
        (out / "selection.json").write_text(json.dumps(best, indent=2) + "\n")
        print(frame[["name", "NSRC-Route_mae", "NSRC-Route_mse", "ARM90_mae", "NSRC_mae"]].to_string(index=False))
        print("BEST", json.dumps(best, indent=2))
    elif args.evaluate:
        if args.fixed_candidate is not None:
            route = {"name": "fixed_" + args.fixed_candidate, "min_events": 0,
                     "window_events": 0, "margin": 0.0, "anchor": "Persistence",
                     "fixed_candidate": args.fixed_candidate}
        else:
            if args.route_name is None:
                raise SystemExit("--route-name or --fixed-candidate required")
            route = next(item for item in SAFE_ROUTE_CANDIDATES if item["name"] == args.route_name)
        metrics, stream, rows = evaluate(args.dataset, args.backbone, args.seed, route)
        out = outroot / args.dataset / f"{args.backbone}_s{args.seed}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "predictions.npz",
            y=stream["y"][rows],
            Persistence=stream["persistence"][rows],
            DLinear=stream["dlinear"][rows],
            ARM90=stream["arm"][rows],
            NSRC=stream["semantic"][rows],
            NSRC_Route=stream["route"][rows],
            **{name.replace("+", "_"): arr[rows] for name, arr in candidate_arrays(stream).items() if "+" in name},
        )
        stream["idx"].iloc[rows].to_csv(out / "test_index.csv", index=False)
        pd.DataFrame(stream["traces"]).to_csv(out / "update_trace.csv", index=False)
        metrics.to_csv(out / "metrics.csv", index=False)
        checks = verify(args.dataset, args.backbone, args.seed, route, stream)
        (out / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(metrics.to_string(index=False))
    else:
        raise SystemExit("use --select or --evaluate")


if __name__ == "__main__":
    main()
