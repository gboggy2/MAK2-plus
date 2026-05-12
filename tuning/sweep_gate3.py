"""Hand-tuning sweep for Gate-3 bypass thresholds + Gate-0 R² floor.

Re-scores PCRedux under a handful of candidate ``QualityGateConfig``
variants, starting from ``DEFAULT_GATES`` and changing one knob at a
time, then a combined config. Prints a side-by-side confusion + per-class
summary so we can pick a winner without launching DE.
"""
from __future__ import annotations

import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MAK2_RANDOM_SEED", "42")

from config import DEFAULT_GATES  # noqa: E402
from tuning.score_gates import CACHE_PATH, CURVES_CSV, LABELS_CSV, score_config  # noqa: E402


CONFIGS = {
    "baseline (DEFAULT_GATES)": DEFAULT_GATES,
    "G3 high-R² bypass 0.999→0.995": replace(DEFAULT_GATES, gate_3_high_r2_bypass_r2=0.995),
    "G3 late bypass 0.995→0.99":    replace(DEFAULT_GATES, gate_3_late_bypass_r2=0.99),
    "G3 both bypasses loosened":    replace(
        DEFAULT_GATES,
        gate_3_high_r2_bypass_r2=0.995,
        gate_3_late_bypass_r2=0.99,
    ),
    "G3 inflection thresh 0.01→0.005": replace(
        DEFAULT_GATES, inflection_threshold_pct_of_range=0.005,
    ),
    "G0 r2_floor 0.99→0.97":        replace(DEFAULT_GATES, r2_floor_standard=0.97),
    "G3 both + G0 0.97":            replace(
        DEFAULT_GATES,
        gate_3_high_r2_bypass_r2=0.995,
        gate_3_late_bypass_r2=0.99,
        r2_floor_standard=0.97,
    ),
    "G3 high-R² 0.995 + G0 0.97 (safest combo)": replace(
        DEFAULT_GATES,
        gate_3_high_r2_bypass_r2=0.995,
        r2_floor_standard=0.97,
    ),
    "G3 high-R² 0.997 only (smaller step)": replace(
        DEFAULT_GATES, gate_3_high_r2_bypass_r2=0.997,
    ),
    "G3 both + G0 0.97 + inflection 0.005": replace(
        DEFAULT_GATES,
        gate_3_high_r2_bypass_r2=0.995,
        gate_3_late_bypass_r2=0.99,
        r2_floor_standard=0.97,
        inflection_threshold_pct_of_range=0.005,
    ),
}


def main():
    with open(CACHE_PATH, "rb") as fh:
        fit_cache = pickle.load(fh)
    labels = pd.read_csv(LABELS_CSV)
    curves = pd.read_csv(CURVES_CSV)
    sample_cycles = (
        curves[curves["curve_id"] == labels["curve_id"].iloc[0]]["cycle"]
        .to_numpy(float)
    )

    rows = []
    for name, gates in CONFIGS.items():
        r = score_config(gates, fit_cache, labels, sample_cycles)
        preds = r["predictions"]
        y_tp = int(((preds["label"] == "y") & (preds["pred"] == "y")).sum())
        y_fn = int(((preds["label"] == "y") & (preds["pred"] == "n")).sum())
        n_tn = int(((preds["label"] == "n") & (preds["pred"] == "n")).sum())
        n_fp = int(((preds["label"] == "n") & (preds["pred"] == "y")).sum())
        rows.append({
            "config":        name,
            "y_TP":          y_tp,
            "y_FN (false FAIL)": y_fn,
            "n_TN":          n_tn,
            "n_FP (false PASS)": n_fp,
            "y_recall":      r["per_class"]["y"]["recall"],
            "y_precision":   r["per_class"]["y"]["precision"],
            "y_F1":          r["per_class"]["y"]["f1"],
            "n_recall":      r["per_class"]["n"]["recall"],
            "raw_acc":       r["raw_accuracy"],
            "bal_acc(y/n)":  (r["per_class"]["y"]["recall"] + r["per_class"]["n"]["recall"]) / 2,
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 50)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
