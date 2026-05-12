"""Print full fit parameters for the remaining false-FAIL / false-PASS curves."""
from __future__ import annotations
import os, pickle, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MAK2_RANDOM_SEED", "42")

import pandas as pd
from tuning.score_gates import CACHE_PATH, LABELS_CSV

with open(CACHE_PATH, "rb") as fh:
    cache = pickle.load(fh)
labels = pd.read_csv(LABELS_CSV)

CURVES = ["maro1.299"] + ["maro1.135","maro1.356","maro1.369","maro2.177","maro2.212","maro3.225"]
print(f"{'curve':<12} {'label':<5} {'R²':>7} {'D0':>10} {'k':>6} {'bg_slope':>10} {'bg_int':>8} {'fit_start':>9} {'fit_end':>7}")
for cid in CURVES:
    fit = cache.get(cid)
    lab = labels.loc[labels["curve_id"]==cid, "label"].iloc[0]
    print(f"{cid:<12} {lab:<5} {fit['R2']:>7.4f} {fit['D0']:>10.2e} {fit['k']:>6.3f} "
          f"{fit['F_bg_slope']:>10.4f} {fit['F_bg_intercept']:>8.3f} "
          f"{fit['fit_start_cycle']:>9.1f} {fit['fit_end_cycle']:>7.1f}")
