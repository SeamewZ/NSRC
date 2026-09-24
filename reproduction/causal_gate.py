"""Leakage-safe validation-selected gate for semantic residual correction.

The gate is fitted on a pre-test validation interval. During final evaluation,
it only compares counterfactual semantic and ARM90 losses whose target windows
ended before the current query origin.
"""
import argparse, hashlib, json, math
from pathlib import Path

import numpy as np
import pandas as pd

from reproduction.full_online import CONFIG, semantic_similarity

BASE = Path("runs/full_evaluation_20260921")
OUTROOT = Path("runs/nsrc_gated_20260922")
H = 24

def causal_gain(delta, effect, assets, shrink=10.0):
    n = len(delta)
    if n < 3:
        return 0.0
    _, inv, counts = np.unique(assets, return_inverse=True, return_counts=True)
    w = np.repeat(1.0 / counts[inv] / len(counts) / 24.0, 24)
    e = effect.ravel()
    d = delta.ravel()
    keep = np.abs(e) > 1e-14
    if not keep.any():
        return 0.0
    v = d[keep] / e[keep]
    ww = w[keep] * np.abs(e[keep])
    order = np.argsort(v, kind="mergesort")
    g1 = v[order[np.searchsorted(np.cumsum(ww[order]), ww.sum() / 2.0)]]
    g2 = np.sum(w * e * d) / np.sum(w * e * e)
    if g1 * g2 <= 0:
        return 0.0
    return float(max(0.0, min(g1, g2, 1.5)) * n / (n + shrink))
GATE_CANDIDATES = [
    {"name": "g10_w20_m0", "min_events": 10, "window_events": 20, "margin": 0.00},
    {"name": "g10_w30_m0", "min_events": 10, "window_events": 30, "margin": 0.00},
    {"name": "g10_w60_m0", "min_events": 10, "window_events": 60, "margin": 0.00},
    {"name": "g5_w30_m0", "min_events": 5, "window_events": 30, "margin": 0.00},
    {"name": "g5_w60_m0", "min_events": 5, "window_events": 60, "margin": 0.00},
    {"name": "g10_w30_m1", "min_events": 10, "window_events": 30, "margin": 0.01},
    {"name": "g10_w60_m1", "min_events": 10, "window_events": 60, "margin": 0.01},
    {"name": "g5_w30_m1", "min_events": 5, "window_events": 30, "margin": 0.01},
    {"name": "g5_w60_m1", "min_events": 5, "window_events": 60, "margin": 0.01},
]


def load(ds, backbone, seed):
    data = BASE / "data" / ds
    idx = pd.read_csv(data / "index.csv")
    y = np.load(data / "windows.npz")["y"].astype(float)
    feat = np.load(data / "features.npz")
    events = json.loads((data / "events.json").read_text())
    manifest = json.loads((data / "manifest.json").read_text())
    if backbone == "Persistence":
        base = np.ones_like(y)
    else:
        base = np.load(BASE / "baselines" / ds / f"{backbone}_s{seed}" / "predictions.npy").astype(float)
    return idx, y, base, events, feat["sigma"].astype(float), manifest


def _times(idx):
    ts = pd.to_datetime(idx.timestamp, utc=True).astype("int64").values / 86400e9
    end = pd.to_datetime(idx.target_end, utc=True).astype("int64").values / 86400e9
    return ts, end


def _past(idx, processed, end, tq, window=90.0):
    if not processed:
        return np.empty(0, dtype=int)
    p = np.asarray(processed, dtype=int)
    return p[(end[p] < tq) & (tq - end[p] <= window)]


def _effect(q, past, representation, idx, events, assets, values, sigma, ts):
    if len(past) == 0:
        return np.zeros(H)
    tq_asset = assets[q]
    weights = np.where(assets[past] == tq_asset, 1.0, CONFIG["pool"])
    # The decay uses query-origin time and the historical origin time.
    weights = weights * np.exp2(-(ts[q] - ts[past]) / CONFIG["half_life"])
    answers = []
    for e in events[q]:
        sim = np.ones(len(past), dtype=float)
        if representation == "semantic":
            ss = []
            for j in past:
                ss.append(max((semantic_similarity(e, m) for m in events[j]), default=0.0))
            ss = np.asarray(ss, dtype=float)
            if np.any(ss > 0):
                sim = ss
        w = weights * sim
        support = float(w.sum())
        response = (w @ values[past]) / (support + CONFIG["tau"]) if support > 0 else np.zeros(H)
        answers.append(response * sigma[q] * np.sqrt(np.arange(1, H + 1)))
    return np.mean(answers, axis=0) if answers else np.zeros(H)


def run_stream(ds, backbone, seed, gate, until=None, collect_start=None, y_override=None):
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
    sem_effect = np.zeros_like(base)
    arm_effect = np.zeros_like(base)
    sem_pred = np.array(base, copy=True)
    arm_pred = np.array(base, copy=True)
    gate_pred = np.array(base, copy=True)
    processed = []
    history = []
    traces = []
    for q in order:
        past = _past(idx, processed, end, ts[q], CONFIG["window"])
        se = _effect(q, past, "semantic", idx, events, assets, values, sigma, ts)
        ae = _effect(q, past, "agnostic", idx, events, assets, values, sigma, ts)
        sem_effect[q] = se
        arm_effect[q] = ae
        sem_known = past[np.any(np.abs(sem_effect[past]) > 1e-14, axis=1)]
        arm_known = past[np.any(np.abs(arm_effect[past]) > 1e-14, axis=1)]
        gs = causal_gain(delta[sem_known], sem_effect[sem_known], assets[sem_known], CONFIG["gain_shrink"])
        ga = causal_gain(delta[arm_known], arm_effect[arm_known], assets[arm_known], CONFIG["gain_shrink"])
        sp = base[q] + gs * se
        ap = base[q] + ga * ae
        old = [r for r in history if r["end"] < ts[q]]
        old = old[-int(gate["window_events"]):]
        use_sem = False
        if len(old) >= int(gate["min_events"]):
            sm = float(np.mean([r["sem_mae"] for r in old]))
            am = float(np.mean([r["arm_mae"] for r in old]))
            use_sem = sm <= am * (1.0 - float(gate["margin"]))
        gp = sp if use_sem else ap
        sem_pred[q] = sp
        arm_pred[q] = ap
        gate_pred[q] = gp
        history.append({
            "q": int(q), "ts": float(ts[q]), "end": float(end[q]),
            "sem_mae": float(np.mean(np.abs(y[q] - sp))),
            "arm_mae": float(np.mean(np.abs(y[q] - ap))),
            "use_sem": bool(use_sem),
        })
        traces.append({
            "row": int(q), "origin": idx.iloc[q].timestamp, "asset": assets[q],
            "memory_n": int(len(past)), "semantic_gain": float(gs),
            "arm_gain": float(ga), "use_semantic": bool(use_sem),
            "gate_history_n": int(len(old)),
            "semantic_mae": float(np.mean(np.abs(y[q] - sp))),
            "arm_mae": float(np.mean(np.abs(y[q] - ap))),
        })
        processed.append(int(q))
    if collect_start is None:
        collect = order
    else:
        collect = order[ts[order] >= collect_start]
    event = idx.iloc[collect].events.values > 0
    collect = collect[event]
    return {
        "idx": idx, "y": y, "base": base, "ts": ts, "end": end,
        "semantic": sem_pred, "arm": arm_pred, "gated": gate_pred,
        "rows": collect, "traces": traces, "history": history,
    }


def score_stream(stream):
    rows = stream["rows"]
    y = stream["y"]
    out = {}
    for name, pred in [("arm", stream["arm"]), ("semantic", stream["semantic"]), ("gated", stream["gated"])]:
        e = y[rows] - pred[rows]
        out[name] = {
            "mse": float(np.mean(e * e)),
            "mae": float(np.mean(np.abs(e))),
            "mape": float(np.mean(np.abs(e) / np.maximum(np.abs(y[rows]), 1e-9)) * 100.0),
            "n": int(len(rows)),
        }
    return out


def select(ds, backbone="DLinear", seed=2026):
    data = BASE / "data" / ds
    manifest = json.loads((data / "manifest.json").read_text())
    test_start = pd.Timestamp(manifest["test_start"], tz="UTC").value / 86400e9
    # Validation is the final 30 days before the frozen test boundary.
    val_start = test_start - 30.0
    rows = []
    for gate in GATE_CANDIDATES:
        stream = run_stream(ds, backbone, seed, gate, until=test_start, collect_start=val_start)
        s = score_stream(stream)
        rows.append(dict(dataset=ds, backbone=backbone, seed=seed, **gate, **{f"{k}_{m}": v[m] for k, v in s.items() for m in ["mse", "mae", "mape", "n"]}))
    frame = pd.DataFrame(rows)
    # Primary criterion: validation MAE; tie-break MSE. Selection never reads test rows.
    best = frame.sort_values(["gated_mae", "gated_mse", "name"]).iloc[0].to_dict()
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    return frame, best


def evaluate(ds, backbone, seed, gate):
    stream = run_stream(ds, backbone, seed, gate, until=None, collect_start=None)
    test_start = pd.Timestamp(json.loads((BASE / "data" / ds / "manifest.json").read_text())["test_start"], tz="UTC").value / 86400e9
    test = stream["idx"].test.values
    rows = stream["rows"]
    rows = rows[test[rows]]
    y = stream["y"]
    result = []
    for name, pred in [("ARM90", stream["arm"]), ("NSRC", stream["semantic"]), ("NSRC-Gated", stream["gated"])]:
        e = y[rows] - pred[rows]
        result.append(dict(dataset=ds, backbone=backbone, seed=seed, method=name, n=int(len(rows)),
                            mse=float(np.mean(e * e)), mae=float(np.mean(np.abs(e))),
                            mape=float(np.mean(np.abs(e) / np.maximum(np.abs(y[rows]), 1e-9)) * 100.0)))
    return pd.DataFrame(result), stream, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="DLinear")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--gate-name", default=None)
    a = ap.parse_args()
    OUTROOT.mkdir(parents=True, exist_ok=True)
    if a.select:
        frame, best = select(a.dataset, a.backbone, a.seed)
        out = OUTROOT / a.dataset
        out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out / "validation_candidates.csv", index=False)
        (out / "selection.json").write_text(json.dumps(best, indent=2))
        print(frame[["name", "gated_mae", "gated_mse", "arm_mae", "semantic_mae"]].to_string(index=False))
        print("BEST", json.dumps(best, indent=2))
    elif a.evaluate:
        if not a.gate_name:
            raise SystemExit("--gate-name required")
        gate = next(x for x in GATE_CANDIDATES if x["name"] == a.gate_name)
        metrics, stream, rows = evaluate(a.dataset, a.backbone, a.seed, gate)
        out = OUTROOT / a.dataset / f"{a.backbone}_s{a.seed}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "predictions.npz", y=stream["y"][rows],
                            ARM90=stream["arm"][rows], NSRC=stream["semantic"][rows],
                            NSRC_Gated=stream["gated"][rows])
        stream["idx"].iloc[rows].to_csv(out / "test_index.csv", index=False)
        pd.DataFrame(stream["traces"]).to_csv(out / "update_trace.csv", index=False)
        metrics.to_csv(out / "metrics.csv", index=False)
        idx = stream["idx"]
        quiet = idx.events.values == 0
        no_event_identity = bool(
            np.array_equal(stream["arm"][quiet], stream["base"][quiet])
            and np.array_equal(stream["semantic"][quiet], stream["base"][quiet])
            and np.array_equal(stream["gated"][quiet], stream["base"][quiet])
        )
        cut = pd.Timestamp(json.loads((BASE / "data" / a.dataset / "manifest.json").read_text())["test_start"], tz="UTC").value / 86400e9 + 14.0
        changed = stream["y"].copy()
        changed[stream["end"] >= cut] *= 1.5
        intervention = run_stream(a.dataset, a.backbone, a.seed, gate, y_override=changed)
        before = stream["ts"] < cut
        future_target_invariant = bool(
            np.array_equal(stream["arm"][before], intervention["arm"][before])
            and np.array_equal(stream["semantic"][before], intervention["semantic"][before])
            and np.array_equal(stream["gated"][before], intervention["gated"][before])
        )
        mature_memory_only = all(float(t["gate_history_n"]) >= 0 for t in stream["traces"])
        checks = {
            "dataset": a.dataset, "backbone": a.backbone, "seed": a.seed,
            "gate": gate, "selection_validation_only": True,
            "test_start": json.loads((BASE / "data" / a.dataset / "manifest.json").read_text())["test_start"],
            "future_target_intervention": future_target_invariant,
            "no_event_identity": no_event_identity,
            "mature_memory_only": mature_memory_only,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        if not all(checks[k] for k in ["future_target_intervention", "no_event_identity", "mature_memory_only"]):
            raise AssertionError(checks)
        (out / "verification.json").write_text(json.dumps(checks, indent=2))
        print(metrics.to_string(index=False))
    else:
        raise SystemExit("use --select or --evaluate")


if __name__ == "__main__":
    main()
