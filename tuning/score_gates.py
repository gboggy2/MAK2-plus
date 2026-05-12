"""Score a ``QualityGateConfig`` against the PCRedux labelled dataset.

Runs MAK2's full per-well fit + gates pipeline over every PCRedux
curve, then computes confusion matrix + balanced accuracy + F1-weighted
score against the rater-consensus labels. Used as the objective
function for the tuning driver (``tune_gates.py``) and as a one-shot
diagnostic for the current default gates.

Designed to be CPU-friendly:

  - Fits are run with seeded RNG (``MAK2_RANDOM_SEED=42``) so the
    same config always gives the same score.
  - Per-curve fits are cached on disk (``_score_cache.pkl``) keyed by
    curve_id — the optimisation loop varies only the gate config, not
    the per-curve fit parameters, so we can fit each curve *once*
    across many tuning trials.

Usage:
  python tuning/score_gates.py                # score DEFAULT_GATES
  python tuning/score_gates.py --no-cache     # ignore the per-curve cache
  python tuning/score_gates.py --fresh        # rebuild the cache from scratch
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Make engine modules importable when running from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MAK2_RANDOM_SEED", "42")

from config import DEFAULT_GATES, QualityGateConfig  # noqa: E402
from fit_well import fit_well  # noqa: E402
from run_batch import run_quality_gates  # noqa: E402

LABEL_MAP = {"y": "PASS", "n": "FAIL", "a": "INDETERMINATE"}
INVERSE_MAP = {v: k for k, v in LABEL_MAP.items()}

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "pcredux"
LABELS_CSV  = FIXTURE_DIR / "pcredux_labels.csv"
CURVES_CSV  = FIXTURE_DIR / "pcredux_curves.csv"
CACHE_PATH  = Path(__file__).resolve().parent / "_score_cache.pkl"


def fit_curve(cycles: np.ndarray, fluor: np.ndarray) -> dict | None:
    """Fit a single curve via the production preprocessing pipeline.

    Delegates to ``fit_well.fit_well`` so PCRedux scoring uses the
    *same* smart-start window selection + background pre-estimation
    that ``run_batch.run_pass1`` applies in production. An earlier
    version of this function called ``MAK2Optimizer.fit`` directly
    with raw cycles + fluor, which skipped the left-trim and produced
    visibly worse fits on late amplifiers (long baseline dominated
    the SSR; F_bg_slope absorbed what should have been growth
    signal). See ``fit_well.py`` for the pipeline details.

    Returns None on hard failure (no fit produced). The gates already
    skip curves with non-None ``error`` and treat them as FAIL.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fit_well(cycles, fluor, verbose=False)
    if result.get("error") is not None:
        return None
    return result


def _build_cache(labels: pd.DataFrame, curves: pd.DataFrame) -> dict[str, dict | None]:
    """Run MAK2 fit on every PCRedux curve, return curve_id → result dict.

    Caches fit results to ``_score_cache.pkl`` so subsequent tuning
    trials don't re-fit (gate scoring is the part that varies, not the
    underlying fit).
    """
    curve_groups = {cid: g for cid, g in curves.groupby("curve_id")}
    cache: dict[str, dict | None] = {}

    t0 = time.perf_counter()
    n = len(labels)
    for i, row in enumerate(labels.itertuples(index=False)):
        g = curve_groups[row.curve_id]
        c = g["cycle"].to_numpy(float)
        f = g["fluorescence"].to_numpy(float)
        cache[row.curve_id] = fit_curve(c, f)
        if (i + 1) % 100 == 0 or i + 1 == n:
            elapsed = time.perf_counter() - t0
            print(
                f"  fit {i+1}/{n}  ({elapsed:.0f}s, {elapsed/(i+1)*1000:.0f} ms/curve)",
                flush=True,
            )
    return cache


def score_config(
    gates: QualityGateConfig,
    fit_cache: dict[str, dict | None],
    labels: pd.DataFrame,
    cycles: np.ndarray,
) -> dict:
    """Apply ``gates`` to the cached fits, return scoring metrics + predictions."""
    preds: list[str] = []
    for cid in labels["curve_id"]:
        fit = fit_cache.get(cid)
        if fit is None:
            preds.append("n")
            continue
        # run_quality_gates mutates the result dict — operate on a copy so
        # cached fits are reusable across many tuning trials.
        result = dict(fit)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_quality_gates([result], cycles, gates=gates)
        if result.get("error") and "No amplification" in str(result["error"]):
            preds.append("n")
        elif result.get("Success") == "✓" and not result.get("error"):
            preds.append("y")
        else:
            preds.append("n")

    df = labels.copy()
    df["pred"] = preds

    # Confusion matrix as a flat dict
    conf = pd.crosstab(df["label"], df["pred"]).to_dict()

    # Per-class precision/recall/F1
    per_class = {}
    recalls = []
    for cls in ["y", "n", "a"]:
        tp = int(((df["label"] == cls) & (df["pred"] == cls)).sum())
        fp = int(((df["label"] != cls) & (df["pred"] == cls)).sum())
        fn = int(((df["label"] == cls) & (df["pred"] != cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall    = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall else float("nan")
        )
        per_class[cls] = {
            "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "support": tp + fn,
        }
        if (tp + fn) > 0 and not np.isnan(recall):
            recalls.append(recall)

    balanced_accuracy = float(np.mean(recalls)) if recalls else float("nan")
    raw_accuracy = float((df["label"] == df["pred"]).mean())

    return {
        "config":            asdict(gates),
        "confusion":         conf,
        "per_class":         per_class,
        "balanced_accuracy": balanced_accuracy,
        "raw_accuracy":      raw_accuracy,
        "predictions":       df,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-cache", action="store_true",
                    help="Don't read or write the fit cache")
    ap.add_argument("--fresh", action="store_true",
                    help="Rebuild the fit cache from scratch")
    args = ap.parse_args()

    labels = pd.read_csv(LABELS_CSV)
    curves = pd.read_csv(CURVES_CSV)

    # Engine assumes cycles are uniform across wells in a "plate"; PCRedux
    # subdatasets all use 1..45, so pick one curve's cycle array.
    sample_cycles = (
        curves[curves["curve_id"] == labels["curve_id"].iloc[0]]["cycle"]
        .to_numpy(float)
    )

    use_cache = not args.no_cache and CACHE_PATH.exists() and not args.fresh
    if use_cache:
        print(f"loading cached fits from {CACHE_PATH}", flush=True)
        with open(CACHE_PATH, "rb") as fh:
            fit_cache = pickle.load(fh)
    else:
        print(f"fitting {len(labels)} curves (~5-10 min)", flush=True)
        fit_cache = _build_cache(labels, curves)
        if not args.no_cache:
            with open(CACHE_PATH, "wb") as fh:
                pickle.dump(fit_cache, fh)
            print(f"  cached to {CACHE_PATH}", flush=True)

    print(f"scoring DEFAULT_GATES...", flush=True)
    result = score_config(DEFAULT_GATES, fit_cache, labels, sample_cycles)

    print("\nCONFUSION MATRIX (rows = true, cols = predicted):")
    print(pd.crosstab(result["predictions"]["label"], result["predictions"]["pred"],
                      margins=True))

    print("\nPER-CLASS METRICS:")
    for cls, m in result["per_class"].items():
        print(f"  {cls}: precision={m['precision']:.3f}  "
              f"recall={m['recall']:.3f}  F1={m['f1']:.3f}  "
              f"(support n={m['support']})")

    print(f"\nBalanced accuracy: {result['balanced_accuracy']:.3f}")
    print(f"Raw accuracy:      {result['raw_accuracy']:.3f}")


if __name__ == "__main__":
    main()
