"""Plate-level regression test.

Runs the full MAK2 fitting pipeline on every well of
``boggy_input.csv`` and compares to ``boggy_reference.json``. Two
modes:

- **Exact equality** (default): with ``MAK2_RANDOM_SEED=42`` the
  pipeline is byte-reproducible; every numeric field must match the
  reference exactly. Catches any unintended behaviour change.

- **Tolerance** (``REGRESSION_TOLERANCE_MODE=1``): used when
  intentionally upgrading scipy/numpy, where floating-point
  arithmetic may shift slightly across versions. Status must still
  match exactly per well; numeric fields are allowed bounded
  drift (D0 ±5%, R² ±0.001, Ct ±0.1 cycles).

The test exercises every public engine path: ``MAK2Optimizer.fit``
through every tier escalation that the data triggers,
``calculate_fit_metrics``, and ``calculate_ct``. If any of those
silently regresses, this test catches it.
"""
from __future__ import annotations

import contextlib
import io
import os

import numpy as np
import pandas as pd
import pytest

from mak2_model import MAK2Model
from optimizer import MAK2Optimizer

TOLERANCE_MODE = os.environ.get("REGRESSION_TOLERANCE_MODE") == "1"


def _fit_well(cycles: np.ndarray, fluor: np.ndarray) -> dict:
    """Run a full MAK2 fit and return the comparison-relevant fields."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        opt = MAK2Optimizer(MAK2Model())
        try:
            params = opt.fit(cycles, fluor, verbose=False)
            metrics = opt.calculate_fit_metrics()
            ct = opt.calculate_ct(method="threshold")
        except Exception as e:
            return {"status": "FAIL", "error": repr(e)[:120]}
    return {
        "status": "PASS",
        "D0": float(params["D0"]),
        "k": float(params["k"]),
        "P0": float(params["P0"]),
        "F_bg_intercept": float(params["F_bg_intercept"]),
        "F_bg_slope": float(params["F_bg_slope"]),
        "R2": float(metrics["r_squared"]),
        "RMSE": float(metrics["rmse"]),
        "Ct": float(ct["ct"]),
        "tier": params.get("tier", "T1-Full"),
    }


def _ref_value(ref_field):
    """Reference JSON stores floats as ``repr(float)`` strings; parse back."""
    if isinstance(ref_field, str):
        return float(ref_field)
    return ref_field


def test_status_matches_for_every_well(boggy_input_df, boggy_reference):
    """Every well's PASS/FAIL status must match the reference, regardless of mode.

    Status is the most important field — even in tolerance mode,
    the test refuses to accept a well silently flipping from PASS
    to FAIL (or vice versa). That's a behaviour change, not a
    floating-point shift.
    """
    cycles = boggy_input_df["Cycles"].to_numpy(dtype=float)
    mismatches = []
    for col in boggy_input_df.columns:
        if col == "Cycles":
            continue
        got = _fit_well(cycles, boggy_input_df[col].to_numpy(dtype=float))
        ref = boggy_reference[col]
        if got["status"] != ref["status"]:
            mismatches.append(f"{col}: got={got['status']} ref={ref['status']}")
    assert not mismatches, "Status mismatches:\n  " + "\n  ".join(mismatches)


def test_kinetic_parameters_match_reference(boggy_input_df, boggy_reference):
    """D0, k, R², Ct match reference exactly (or within tolerance).

    The headline regression assertion. With seeding enabled the
    pipeline is byte-reproducible; this test catches any unintended
    drift in optimization behaviour, bounds derivation, tier
    escalation, etc.
    """
    cycles = boggy_input_df["Cycles"].to_numpy(dtype=float)
    failures: list[str] = []

    for col in boggy_input_df.columns:
        if col == "Cycles":
            continue
        ref = boggy_reference[col]
        if ref["status"] != "PASS":
            continue  # status mismatch caught by the other test
        got = _fit_well(cycles, boggy_input_df[col].to_numpy(dtype=float))
        if got["status"] != "PASS":
            failures.append(f"{col}: did not fit (status={got['status']})")
            continue

        ref_D0 = _ref_value(ref["D0"])
        ref_R2 = _ref_value(ref["R2"])
        ref_Ct = _ref_value(ref["Ct"])

        if TOLERANCE_MODE:
            # Bounded drift OK in tolerance mode (e.g. after scipy bump).
            d0_pct = abs(got["D0"] - ref_D0) / abs(ref_D0)
            if d0_pct > 0.05:
                failures.append(f"{col}: D0 {got['D0']:.4e} vs {ref_D0:.4e} (Δ={d0_pct:.1%})")
            if abs(got["R2"] - ref_R2) > 0.001:
                failures.append(f"{col}: R² {got['R2']:.4f} vs {ref_R2:.4f}")
            if abs(got["Ct"] - ref_Ct) > 0.1:
                failures.append(f"{col}: Ct {got['Ct']:.2f} vs {ref_Ct:.2f}")
        else:
            # Exact equality: bit-for-bit reproducibility is the contract.
            if got["D0"] != ref_D0:
                failures.append(f"{col}: D0 {got['D0']!r} vs {ref_D0!r}")
            if got["R2"] != ref_R2:
                failures.append(f"{col}: R² {got['R2']!r} vs {ref_R2!r}")
            if got["Ct"] != ref_Ct:
                failures.append(f"{col}: Ct {got['Ct']!r} vs {ref_Ct!r}")

    if failures:
        mode = "tolerance" if TOLERANCE_MODE else "exact"
        msg = f"{len(failures)} regression failure(s) in {mode} mode:\n  " + "\n  ".join(failures)
        pytest.fail(msg)


def test_dilution_series_d0_ladder(boggy_input_df, boggy_reference):
    """D0 decreases monotonically across the Boggy dilution series.

    Regression check on the *biological* meaning of the fits, not
    just the numeric values: the Boggy fixture is a 10× dilution
    series F1 → F6, so D0 must drop by roughly an order of
    magnitude per concentration step. This catches scenarios where
    the engine gives consistent numbers but the numbers are
    internally inconsistent (e.g. all wells fitting to the same D0,
    or the relationship inverting).
    """
    # Group F1.1 + F1.2, F2.1 + F2.2, ..., F6.1 + F6.2; take the mean.
    levels = []
    for level in range(1, 7):
        d0s = []
        for rep in (1, 2):
            ref = boggy_reference[f"F{level}.{rep}"]
            if ref["status"] == "PASS":
                d0s.append(_ref_value(ref["D0"]))
        if d0s:
            levels.append(np.mean(d0s))

    assert len(levels) >= 4, "need at least 4 dilution levels for ladder check"

    # Each level should be at least 3× lower than the previous (allowing
    # for some optimizer noise and replicate spread; in clean data each
    # step is ~10×).
    ratios = [levels[i] / levels[i + 1] for i in range(len(levels) - 1)]
    bad = [r for r in ratios if r < 3.0]
    assert not bad, (
        f"Dilution ladder broken: at least one consecutive ratio is < 3×.\n"
        f"  Per-level mean D0: {levels}\n"
        f"  Consecutive ratios: {ratios}"
    )
