"""Optimise the quality-gate parameters against the PCRedux labels.

Uses scipy's differential_evolution to search for a
``QualityGateConfig`` that maximises agreement with the
rater-consensus labels in ``tests/fixtures/pcredux/pcredux_labels.csv``,
subject to an asymmetric cost (false PASS hurts more than false FAIL).

Output:
    tuning/results/<timestamp>-best.json      — best config found
    tuning/results/<timestamp>-history.csv   — every trial's score
    tuning/results/<timestamp>-report.md     — summary for humans

Usage:
    python tuning/tune_gates.py [--maxiter N] [--popsize K]

The score_gates module is responsible for the per-curve fit + scoring
work; this script is just the outer optimisation loop.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MAK2_RANDOM_SEED", "42")

from config import DEFAULT_GATES, QualityGateConfig  # noqa: E402
from tuning.score_gates import (  # noqa: E402
    CACHE_PATH, LABELS_CSV, CURVES_CSV,
    _build_cache, score_config,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─── Search space ─────────────────────────────────────────────────────────────
#
# Each entry: (config field name, (lo, hi), is_integer). The differential-
# evolution optimiser searches the box; floats stay floats, integers round
# at evaluation time.
#
# Bounds are chosen to bracket the current defaults by ~2x in each
# direction without exploring physically nonsensical regions (e.g. R²
# floors below 0.5 just reject nothing).

SEARCH_SPACE: list[tuple[str, tuple[float, float], bool]] = [
    ("r2_floor_standard",                 (0.90,  0.999), False),
    ("r2_floor_late_amplifier",           (0.50,  0.99),  False),
    ("late_amplifier_tail_window",        (1,     10),    True ),
    ("min_fit_window_cycles",             (5,     20),    True ),
    ("min_r2_gap_mak2_vs_linear",         (0.0,   0.15),  False),
    ("gate_2b_late_bypass_r2",            (0.95,  1.0),   False),
    ("inflection_threshold_pct_of_range", (0.001, 0.05),  False),
    ("gate_3_high_r2_bypass_r2",          (0.99,  1.0),   False),
    ("gate_3_high_r2_bypass_min_window",  (5,     20),    True ),
    ("gate_3_late_bypass_r2",             (0.95,  1.0),   False),
]


def vec_to_config(x: np.ndarray) -> QualityGateConfig:
    """Map a length-N optimisation vector back to a QualityGateConfig.

    Integer-typed knobs are rounded; everything else stays float.
    Fields not in SEARCH_SPACE inherit the values from DEFAULT_GATES.
    """
    overrides = {}
    for value, (name, _bounds, is_int) in zip(x, SEARCH_SPACE):
        overrides[name] = int(round(value)) if is_int else float(value)
    return replace(DEFAULT_GATES, **overrides)


# ─── Objective function ───────────────────────────────────────────────────────


def make_objective(fit_cache, labels, cycles, *,
                   false_pass_weight: float = 2.0,
                   false_fail_weight: float = 1.0,
                   history: list | None = None):
    """Build the cost function passed to differential_evolution.

    Minimised: a weighted misclassification cost, with false PASS
    (predicting `y` when the label is `n`) penalised more than the
    reverse. Returns a negative-utility-style scalar so DE minimises
    toward the best config.

    The optional ``history`` list accumulates (config_dict, score) per
    call for post-tuning analysis.
    """
    trial_no = [0]

    def objective(x: np.ndarray) -> float:
        gates = vec_to_config(x)
        result = score_config(gates, fit_cache, labels, cycles)
        per_class = result["per_class"]
        # FP for `y` = predicting y when true label is n or a
        # FP for `n` = predicting n when true label is y or a
        false_pass = per_class["y"]["fp"]
        false_fail = per_class["n"]["fp"]
        cost = (
            false_pass_weight * false_pass
            + false_fail_weight * false_fail
        )
        trial_no[0] += 1
        if trial_no[0] % 10 == 0 or trial_no[0] == 1:
            print(
                f"  trial {trial_no[0]}: cost={cost:.0f}  "
                f"BA={result['balanced_accuracy']:.3f}  "
                f"FP_y={false_pass}  FP_n={false_fail}",
                flush=True,
            )
        if history is not None:
            history.append((asdict(gates), cost, result["balanced_accuracy"]))
        return float(cost)

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=30,
                    help="DE generations (default 30; ~30*15 = 450 trials)")
    ap.add_argument("--popsize", type=int, default=15,
                    help="DE population size (default 15)")
    ap.add_argument("--false-pass-weight", type=float, default=2.0,
                    help="cost multiplier for false-positive predictions")
    ap.add_argument("--false-fail-weight", type=float, default=1.0,
                    help="cost multiplier for false-negative predictions")
    args = ap.parse_args()

    labels = pd.read_csv(LABELS_CSV)
    curves = pd.read_csv(CURVES_CSV)
    sample_cycles = (
        curves[curves["curve_id"] == labels["curve_id"].iloc[0]]["cycle"]
        .to_numpy(float)
    )

    # Reuse the fit cache score_gates.py builds — gates vary, fits don't.
    if CACHE_PATH.exists():
        print(f"loading fit cache from {CACHE_PATH}", flush=True)
        with open(CACHE_PATH, "rb") as fh:
            fit_cache = pickle.load(fh)
    else:
        print(f"building fit cache (one-time, ~5-10 min)", flush=True)
        fit_cache = _build_cache(labels, curves)
        with open(CACHE_PATH, "wb") as fh:
            pickle.dump(fit_cache, fh)

    print(
        f"\nstarting DE optimisation: maxiter={args.maxiter}, "
        f"popsize={args.popsize}, "
        f"false_pass_weight={args.false_pass_weight}, "
        f"false_fail_weight={args.false_fail_weight}",
        flush=True,
    )
    print(f"  search space: {len(SEARCH_SPACE)} knobs", flush=True)

    history: list = []
    objective = make_objective(
        fit_cache, labels, sample_cycles,
        false_pass_weight=args.false_pass_weight,
        false_fail_weight=args.false_fail_weight,
        history=history,
    )

    bounds = [(lo, hi) for _, (lo, hi), _ in SEARCH_SPACE]
    t0 = time.perf_counter()
    result = differential_evolution(
        objective,
        bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=42,
        tol=0.01,
        polish=False,
        updating="deferred",
        workers=1,
    )
    elapsed = time.perf_counter() - t0

    best_config = vec_to_config(result.x)
    best_score = score_config(best_config, fit_cache, labels, sample_cycles)

    print(f"\nDE finished in {elapsed/60:.1f} min, {result.nit} generations, "
          f"{result.nfev} trials", flush=True)
    print(f"best cost: {result.fun:.0f}", flush=True)
    print(f"best balanced accuracy: {best_score['balanced_accuracy']:.3f}", flush=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Write best config
    best_path = RESULTS_DIR / f"{timestamp}-best.json"
    best_path.write_text(json.dumps(asdict(best_config), indent=2))
    print(f"\nbest config saved to {best_path}", flush=True)

    # Write trial history
    history_path = RESULTS_DIR / f"{timestamp}-history.csv"
    with open(history_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        if history:
            writer.writerow(["trial"] + list(history[0][0].keys()) + ["cost", "balanced_accuracy"])
            for i, (cfg, cost, ba) in enumerate(history, 1):
                writer.writerow([i] + list(cfg.values()) + [cost, ba])
    print(f"trial history saved to {history_path}", flush=True)


if __name__ == "__main__":
    main()
