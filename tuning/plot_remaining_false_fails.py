"""Plot the remaining false-FAIL PCRedux curves with their MAK2+ fits overlaid.

After the gate tuning in commit 846e824, ~9 PCRedux curves remain
false-FAILs (label='y', predicted='n'). This script:

  1. Re-applies the *current* DEFAULT_GATES to the cached fits to find
     the remaining false-FAIL set.
  2. For each curve, reconstructs the MAK2+ fitted curve from the
     cached parameters (D0, k, P0, F_bg_intercept, F_bg_slope).
  3. Plots raw fluorescence + fitted MAK2 curve + fit window, annotated
     with R², D0, the rejection reason, and PCRedux label.

Output: tuning/remaining_false_fails.png

Usage:
  python tuning/plot_remaining_false_fails.py
  python tuning/plot_remaining_false_fails.py --label y  # which label to plot
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MAK2_RANDOM_SEED", "42")

from config import DEFAULT_GATES  # noqa: E402
from mak2_model import MAK2Model  # noqa: E402
from run_batch import run_quality_gates  # noqa: E402
from tuning.score_gates import CACHE_PATH, CURVES_CSV, LABELS_CSV  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", default="y",
                    help="PCRedux label to filter on (default: y for false-FAILs; "
                         "use 'n' to plot false-PASSes)")
    ap.add_argument("--pred", default=None,
                    help="Prediction to filter on. Default: opposite of --label "
                         "(y→n for false-FAILs, n→y for false-PASSes)")
    ap.add_argument("--out", default=None,
                    help="Output PNG path (default: derived from label+pred)")
    args = ap.parse_args()
    if args.pred is None:
        args.pred = "n" if args.label == "y" else "y"
    if args.out is None:
        kind = "false_fails" if (args.label == "y" and args.pred == "n") \
               else "false_passes" if (args.label == "n" and args.pred == "y") \
               else f"{args.label}_pred_{args.pred}"
        args.out = str(Path(__file__).parent / f"remaining_{kind}.png")

    with open(CACHE_PATH, "rb") as fh:
        fit_cache = pickle.load(fh)
    labels = pd.read_csv(LABELS_CSV)
    curves = pd.read_csv(CURVES_CSV)
    sample_cycles = (
        curves[curves["curve_id"] == labels["curve_id"].iloc[0]]["cycle"]
        .to_numpy(float)
    )

    # Group curves once for O(1) per-curve lookup
    curve_groups = {cid: g for cid, g in curves.groupby("curve_id")}

    # Re-grade under current DEFAULT_GATES and collect curves with label==--label
    # whose prediction == --pred.
    selected = []
    for cid, true_label in zip(labels["curve_id"], labels["label"]):
        if true_label != args.label:
            continue
        fit = fit_cache.get(cid)
        if fit is None:
            pred = "n"
            reason = "fit_failed"
        else:
            result = dict(fit)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_quality_gates([result], sample_cycles, gates=DEFAULT_GATES)
            err = result.get("error")
            if err and "No amplification" in str(err):
                pred = "n"
                reason = str(err)
            elif result.get("Success") == "✓" and not err:
                pred = "y"
                reason = "passed all gates"
            else:
                pred = "n"
                reason = "unknown"
        if pred == args.pred:
            selected.append((cid, fit, reason))

    n = len(selected)
    print(f"Curves with label='{args.label}', pred='{args.pred}': {n}")
    if n == 0:
        print("Nothing to plot.")
        return

    # Grid layout
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    model = MAK2Model()

    for ax, (cid, fit, reason) in zip(axes, selected):
        g = curve_groups[cid]
        cyc = g["cycle"].to_numpy(float)
        flu = g["fluorescence"].to_numpy(float)
        ax.plot(cyc, flu, "o", ms=3, color="tab:blue", label="raw")

        if fit is not None and fit.get("D0") is not None:
            try:
                pred = model.simulate_to_cycle(
                    D0=fit["D0"], k=fit["k"], P0=fit["P0"],
                    cycles=cyc,
                    F_bg_intercept=fit["F_bg_intercept"],
                    F_bg_slope=fit["F_bg_slope"],
                )
                ax.plot(cyc, pred, "-", color="tab:red", lw=1.5, label="MAK2+ fit")
            except Exception as e:
                ax.text(0.5, 0.5, f"sim failed: {e}",
                        transform=ax.transAxes, ha="center")

            fs, fe = fit.get("fit_start_cycle"), fit.get("fit_end_cycle")
            if fs is not None and fe is not None:
                ax.axvspan(fs, fe, alpha=0.08, color="tab:green", label="fit window")

            r2 = fit.get("R2"); d0 = fit.get("D0")
            ax.set_title(
                f"{cid}\nR²={r2:.4f}  D0={d0:.2e}\n"
                f"reason: {reason.replace('No amplification detected ', '').strip('()')}",
                fontsize=9,
            )
        else:
            ax.set_title(f"{cid}\n(no fit cached)", fontsize=9)

        ax.set_xlabel("Cycle")
        ax.set_ylabel("Fluorescence")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
