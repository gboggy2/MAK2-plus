"""Sweep Gate 0 R² floor configurations to address the remaining errors.

Goal: recover maro1.299 (false FAIL, R²=0.9835, not late) and reject
maro1.356 (false PASS, R²=0.9083, late) without affecting the rest.
"""
from __future__ import annotations
import os, pickle, sys
from dataclasses import replace
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MAK2_RANDOM_SEED", "42")

from config import DEFAULT_GATES
from tuning.score_gates import CACHE_PATH, CURVES_CSV, LABELS_CSV, score_config

with open(CACHE_PATH, "rb") as f: cache = pickle.load(f)
labels = pd.read_csv(LABELS_CSV)
curves = pd.read_csv(CURVES_CSV)
sample_cycles = curves[curves["curve_id"]==labels["curve_id"].iloc[0]]["cycle"].to_numpy(float)

CONFIGS = {
    "baseline (0.99 std / 0.85 late)": DEFAULT_GATES,
    "tighten late: 0.85 → 0.92":        replace(DEFAULT_GATES, r2_floor_late_amplifier=0.92),
    "tighten late: 0.85 → 0.95":        replace(DEFAULT_GATES, r2_floor_late_amplifier=0.95),
    "lower std: 0.99 → 0.98":           replace(DEFAULT_GATES, r2_floor_standard=0.98),
    "both: std=0.98, late=0.92":        replace(DEFAULT_GATES, r2_floor_standard=0.98, r2_floor_late_amplifier=0.92),
    "both: std=0.98, late=0.95":        replace(DEFAULT_GATES, r2_floor_standard=0.98, r2_floor_late_amplifier=0.95),
}

rows = []
for name, gates in CONFIGS.items():
    r = score_config(gates, cache, labels, sample_cycles)
    preds = r["predictions"]
    rows.append({
        "config": name,
        "y_TP": int(((preds["label"]=="y") & (preds["pred"]=="y")).sum()),
        "y_FN": int(((preds["label"]=="y") & (preds["pred"]=="n")).sum()),
        "n_TN": int(((preds["label"]=="n") & (preds["pred"]=="n")).sum()),
        "n_FP": int(((preds["label"]=="n") & (preds["pred"]=="y")).sum()),
        "y_recall": r["per_class"]["y"]["recall"],
        "y_F1": r["per_class"]["y"]["f1"],
        "raw_acc": r["raw_accuracy"],
    })
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
