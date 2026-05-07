"""Regenerate the Phase 0.5 test fixtures from boggy_input.csv.

Run from the repo root:

    MAK2_RANDOM_SEED=42 python tests/fixtures/build_fixtures.py

Produces:

  - ``tests/fixtures/single_well_F1_1.json``: the F1.1 well's cycle and
    fluorescence arrays, used by tests/unit/test_mak2_model.py for
    deterministic forward-simulation reference values.
  - ``tests/fixtures/boggy_reference.json``: the seeded MAK2 fit
    output (D0, k, P0, F_bg, R², RMSE, Ct, Status) for all 12 wells.
    Used by tests/regression/test_plate_regression.py for exact /
    tolerance-mode equality assertions.

The fixtures must be regenerated whenever the engine's behaviour
intentionally changes (e.g. tier logic, bounds derivation). Do not
regenerate to "fix" a broken test — the test exists to catch
unintended behaviour shifts. The diff between old and new fixture
JSON is the permanent record of what changed scientifically.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the engine importable when running this script from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mak2_model import MAK2Model  # noqa: E402
from optimizer import MAK2Optimizer  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent
INPUT_CSV = FIXTURES_DIR / "boggy_input.csv"


def _to_jsonable(o):
    """Convert numpy types to JSON-serialisable primitives.

    Floats use ``repr`` to preserve every bit so the regression
    test's exact-equality mode is meaningful.
    """
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_to_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        if np.isnan(v):
            return "NaN"
        if np.isinf(v):
            return "Inf" if v > 0 else "-Inf"
        return repr(v)
    if isinstance(o, (np.integer, int, bool, np.bool_)):
        return int(o) if not isinstance(o, (bool, np.bool_)) else bool(o)
    return None if o is None else str(o)


def build_single_well():
    """Write the F1.1 well's cycle + fluorescence arrays to JSON."""
    df = pd.read_csv(INPUT_CSV)
    cycles = df["Cycles"].astype(float).tolist()
    fluor = df["F1.1"].astype(float).tolist()
    out = {
        "well": "F1.1",
        "channel": "FAM",  # nominal — Boggy.csv doesn't carry channel info
        "cycles": cycles,
        "fluorescence": fluor,
    }
    out_path = FIXTURES_DIR / "single_well_F1_1.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path.name}: {len(cycles)} cycles")


def build_reference():
    """Run a seeded MAK2 fit on every Boggy well and dump the results."""
    if os.environ.get("MAK2_RANDOM_SEED") != "42":
        raise RuntimeError(
            "Set MAK2_RANDOM_SEED=42 before regenerating the reference fixture. "
            "Without seeding, the reference would be non-deterministic."
        )

    df = pd.read_csv(INPUT_CSV)
    cycles = df["Cycles"].to_numpy(dtype=float)

    results = {}
    for col in df.columns:
        if col == "Cycles":
            continue
        fluor = df[col].to_numpy(dtype=float)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                opt = MAK2Optimizer(MAK2Model())
                params = opt.fit(cycles, fluor, verbose=False)
                metrics = opt.calculate_fit_metrics()
                ct = opt.calculate_ct(method="threshold")
            results[col] = {
                "status": "PASS",
                "D0": params["D0"],
                "k": params["k"],
                "P0": params["P0"],
                "F_bg_intercept": params["F_bg_intercept"],
                "F_bg_slope": params["F_bg_slope"],
                "R2": metrics["r_squared"],
                "RMSE": metrics["rmse"],
                "Ct": ct["ct"],
                "tier": params.get("tier", "T1-Full"),
            }
        except Exception as e:
            results[col] = {"status": "FAIL", "error": repr(e)[:120]}

    out_path = FIXTURES_DIR / "boggy_reference.json"
    out_path.write_text(json.dumps(_to_jsonable(results), indent=2, sort_keys=True))
    n_pass = sum(1 for r in results.values() if r["status"] == "PASS")
    print(f"wrote {out_path.name}: {n_pass}/{len(results)} wells PASS")


if __name__ == "__main__":
    build_single_well()
    build_reference()
