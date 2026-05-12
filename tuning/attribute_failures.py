"""Per-curve gate-failure attribution for PCRedux false-FAILs.

Re-applies ``run_quality_gates`` to the cached fits and classifies every
false-FAIL (PCRedux label = ``y``, MAK2 predicts ``n``) by which gate
rejected it. Output:

  - Histogram of rejection reasons over the false-FAIL set.
  - Per-curve CSV (``tuning/false_fail_attribution.csv``) with the
    rejection reason and the relevant fit metrics, so the user can sort
    by gate, by R², by fit-window width, etc., and decide which knobs
    to tune.

Designed to be fast (~seconds) since the fits come from ``_score_cache.pkl``.

Note: the no-amp pre-check inside ``run_pass1`` (run_batch.py:414-502)
is *not* on this path — ``score_gates.fit_curve`` calls the optimizer
directly. So every false-FAIL here is a quality-gate rejection, not an
upstream pre-check rejection. That's a useful answer on its own:
tuning the gates *can* recover these curves.

Usage:
  python tuning/attribute_failures.py
  python tuning/attribute_failures.py --label y    # which label to attribute
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MAK2_RANDOM_SEED", "42")

from config import DEFAULT_GATES  # noqa: E402
from tuning.score_gates import (  # noqa: E402
    CACHE_PATH, CURVES_CSV, LABELS_CSV,
)


# ── Reason → gate classification ────────────────────────────────────────
# run_quality_gates writes 'No amplification detected (<reason>)' on
# rejection. These regexes match the reason text emitted at the four
# gate sites in run_batch.py (lines 1310, 1320, 1372, 1421).
GATE_PATTERNS = [
    ("gate0_r2_floor",        re.compile(r"R² = [\d.]+ < [\d.]+")),
    ("gate2_fit_window",      re.compile(r"Fit window \d+ cycles <")),
    ("gate2b_mak2_vs_linear", re.compile(r"MAK2 not better than linear")),
    ("gate3_no_inflection",   re.compile(r"No inflection")),
]


def classify_reason(err: str | None) -> str:
    """Map a 'No amplification detected (<reason>)' string to a gate tag."""
    if not err:
        return "no_error"
    for tag, pat in GATE_PATTERNS:
        if pat.search(err):
            return tag
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", default="y",
                    help="PCRedux label whose false-FAILs to attribute (default: y)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "false_fail_attribution.csv"),
                    help="Output CSV path")
    args = ap.parse_args()

    if not CACHE_PATH.exists():
        sys.exit(
            f"Fit cache not found at {CACHE_PATH}.\n"
            f"Run `python tuning/score_gates.py` first (~15 min)."
        )

    print(f"loading fit cache from {CACHE_PATH}", flush=True)
    with open(CACHE_PATH, "rb") as fh:
        fit_cache = pickle.load(fh)

    labels = pd.read_csv(LABELS_CSV)
    curves = pd.read_csv(CURVES_CSV)
    sample_cycles = (
        curves[curves["curve_id"] == labels["curve_id"].iloc[0]]["cycle"]
        .to_numpy(float)
    )

    # Re-score under DEFAULT_GATES so we know which curves are predicted FAIL.
    # We need the per-curve error string, which score_config drops — so we
    # re-run the gates ourselves here, mirroring score_config's loop but
    # keeping the reason.
    from run_batch import run_quality_gates  # local import; engine path set above
    import io, contextlib

    rows = []
    for cid, true_label in zip(labels["curve_id"], labels["label"]):
        fit = fit_cache.get(cid)
        if fit is None:
            rows.append({
                "curve_id": cid, "label": true_label, "pred": "n",
                "reason": "fit_failed", "gate": "fit_failed",
                "R2": np.nan, "fit_start": np.nan, "fit_end": np.nan,
                "fit_width": np.nan, "D0": np.nan,
            })
            continue
        result = dict(fit)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_quality_gates([result], sample_cycles, gates=DEFAULT_GATES)
        err = result.get("error")
        if err and "No amplification" in str(err):
            pred = "n"
        elif result.get("Success") == "✓" and not err:
            pred = "y"
        else:
            pred = "n"
        fs = fit.get("fit_start_cycle")
        fe = fit.get("fit_end_cycle")
        rows.append({
            "curve_id":  cid,
            "label":     true_label,
            "pred":      pred,
            "reason":    str(err) if err else "",
            "gate":      classify_reason(err) if err else "passed",
            "R2":        fit.get("R2"),
            "fit_start": fs,
            "fit_end":   fe,
            "fit_width": (fe - fs) if (fs is not None and fe is not None) else np.nan,
            "D0":        fit.get("D0"),
        })

    df = pd.DataFrame(rows)
    target = df[(df["label"] == args.label) & (df["pred"] == "n")].copy()

    print(f"\nFalse-FAIL set: label='{args.label}', predicted='n' "
          f"→ {len(target)} curves\n")

    # Histogram
    hist = target["gate"].value_counts().sort_values(ascending=False)
    print("Rejection-reason histogram:")
    for gate, n in hist.items():
        print(f"  {gate:30s} {n:3d}")

    # Per-gate distribution of relevant metrics
    print("\nMetric distributions by gate (R², fit_width):")
    for gate, sub in target.groupby("gate"):
        r2 = sub["R2"].dropna()
        fw = sub["fit_width"].dropna()
        print(f"  {gate}:")
        if len(r2):
            print(f"    R²        n={len(r2):2d}  min={r2.min():.4f}  "
                  f"median={r2.median():.4f}  max={r2.max():.4f}")
        if len(fw):
            print(f"    fit_width n={len(fw):2d}  min={fw.min():.1f}  "
                  f"median={fw.median():.1f}  max={fw.max():.1f}")

    # Curve-ID listing per gate (handy for plotting)
    print("\nCurve IDs by gate:")
    for gate, sub in target.groupby("gate"):
        ids = ", ".join(sub["curve_id"].tolist())
        print(f"  {gate} ({len(sub)}): {ids}")

    target.to_csv(args.out, index=False)
    print(f"\nWrote per-curve attribution to {args.out}")


if __name__ == "__main__":
    main()
