"""Score MAK2+ against the extended PCRedux + chipPCR labelled datasets.

For each dataset under tests/fixtures/pcredux_extended/<name>/:

  1. Load curves.csv + labels.csv
  2. Fit every curve via fit_well (cached per-dataset on disk)
  3. Apply DEFAULT_GATES
  4. Print per-dataset confusion matrix + per-class P/R/F1
  5. Print an aggregate roll-up across all datasets

Usage:
  python tuning/score_extended.py                  # score all datasets
  python tuning/score_extended.py --datasets kbqPCR  # subset
  python tuning/score_extended.py --fresh          # rebuild fit caches
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MAK2_RANDOM_SEED", "42")

from config import DEFAULT_GATES  # noqa: E402
from fit_well import fit_well  # noqa: E402
from run_batch import run_quality_gates  # noqa: E402

EXTENDED_ROOT = ROOT / "tests" / "fixtures" / "pcredux_extended"


def discover_datasets() -> list[str]:
    return sorted(
        p.name for p in EXTENDED_ROOT.iterdir()
        if p.is_dir() and (p / "curves.csv").exists() and (p / "labels.csv").exists()
    )


def fit_all_curves(curves: pd.DataFrame, labels: pd.DataFrame) -> dict[str, dict | None]:
    """Fit every curve via fit_well. Returns curve_id → result dict (or None)."""
    cache: dict[str, dict | None] = {}
    by_curve = {cid: g for cid, g in curves.groupby("curve_id")}
    t0 = time.perf_counter()
    n = len(labels)
    for i, cid in enumerate(labels["curve_id"]):
        g = by_curve[cid].sort_values("cycle")
        c = g["cycle"].to_numpy(float)
        f = g["fluorescence"].to_numpy(float)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = fit_well(c, f, verbose=False)
        cache[cid] = None if r.get("error") else r
        if (i + 1) % 100 == 0 or i + 1 == n:
            elapsed = time.perf_counter() - t0
            print(f"  fit {i+1}/{n}  ({elapsed:.0f}s, {elapsed/(i+1)*1000:.0f} ms/curve)",
                  flush=True)
    return cache


def score_dataset(curves: pd.DataFrame, labels: pd.DataFrame,
                  cache: dict[str, dict | None]) -> pd.DataFrame:
    """Apply gates to cached fits; return labels with a 'pred' column."""
    cycles = (
        curves[curves["curve_id"] == labels["curve_id"].iloc[0]]
        .sort_values("cycle")["cycle"].to_numpy(float)
    )
    preds = []
    for cid in labels["curve_id"]:
        fit = cache.get(cid)
        if fit is None:
            preds.append("n")
            continue
        result = dict(fit)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_quality_gates([result], cycles, gates=DEFAULT_GATES)
        err = result.get("error")
        if err and "No amplification" in str(err):
            preds.append("n")
        elif result.get("Success") == "✓" and not err:
            preds.append("y")
        else:
            preds.append("n")
    out = labels.copy()
    out["pred"] = preds
    return out


def metrics_block(df: pd.DataFrame, name: str) -> dict:
    """Print confusion + per-class metrics; return aggregate row for roll-up."""
    print(f"\n=== {name} ({len(df)} curves) ===")
    print(pd.crosstab(df["label"], df["pred"], margins=True))

    per_class = {}
    print("\nPer-class:")
    for cls in ["y", "n", "a"]:
        tp = int(((df["label"] == cls) & (df["pred"] == cls)).sum())
        fp = int(((df["label"] != cls) & (df["pred"] == cls)).sum())
        fn = int(((df["label"] == cls) & (df["pred"] != cls)).sum())
        support = tp + fn
        if support == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / support
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else float("nan"))
        per_class[cls] = dict(precision=precision, recall=recall, f1=f1,
                              tp=tp, fp=fp, fn=fn, support=support)
        print(f"  {cls}: P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  "
              f"(TP={tp}, FP={fp}, FN={fn}, n={support})")

    raw_acc = (df["label"] == df["pred"]).mean()
    # "Effective" y-recall ignoring class 'a' (predicting 'y' on an 'a'
    # curve is treated as neither right nor wrong for this metric since
    # MAK2 has no 'ambiguous' output).
    yn = df[df["label"].isin(["y", "n"])]
    yn_acc = (yn["label"] == yn["pred"]).mean()
    print(f"\nRaw accuracy:       {raw_acc:.4f}")
    print(f"y/n-only accuracy:  {yn_acc:.4f}")
    return {"name": name, "n": len(df), "raw_acc": raw_acc, "yn_acc": yn_acc,
            "y_recall": per_class.get("y", {}).get("recall"),
            "n_recall": per_class.get("n", {}).get("recall"),
            "y_F1":     per_class.get("y", {}).get("f1")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Subset of dataset names to score (default: all)")
    ap.add_argument("--fresh", action="store_true",
                    help="Rebuild fit caches from scratch")
    args = ap.parse_args()

    all_ds = discover_datasets()
    targets = args.datasets if args.datasets else all_ds
    missing = [d for d in targets if d not in all_ds]
    if missing:
        sys.exit(f"Datasets not found: {missing}\nAvailable: {all_ds}")

    rollup = []
    for name in targets:
        d = EXTENDED_ROOT / name
        curves = pd.read_csv(d / "curves.csv")
        labels = pd.read_csv(d / "labels.csv")
        cache_path = d / "_fit_cache.pkl"

        if cache_path.exists() and not args.fresh:
            print(f"[{name}] loading cached fits from {cache_path.name}", flush=True)
            with open(cache_path, "rb") as fh:
                cache = pickle.load(fh)
        else:
            print(f"[{name}] fitting {len(labels)} curves...", flush=True)
            cache = fit_all_curves(curves, labels)
            with open(cache_path, "wb") as fh:
                pickle.dump(cache, fh)

        df = score_dataset(curves, labels, cache)
        rollup.append(metrics_block(df, name))
        df.to_csv(d / "predictions.csv", index=False)

    # Aggregate roll-up
    if len(rollup) > 1:
        print("\n" + "=" * 60)
        print("ROLL-UP")
        print("=" * 60)
        rdf = pd.DataFrame(rollup)
        print(rdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
