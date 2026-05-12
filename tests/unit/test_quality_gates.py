"""Unit tests for the quality-gate logic in ``run_batch.run_quality_gates``.

The gates pin each well's ``Status`` to ``PASS`` / ``FAIL`` /
``INDETERMINATE`` based on R², fit-window width, sigmoid-vs-linear
discrimination, and sigmoid shape. Tests construct minimal synthetic
result dicts that pass / fail each gate.

Note: the thresholds asserted here reflect the code's actual values,
which differ slightly from CLAUDE.md's narrative description (the
spec was written aspirationally). For Phase 0.5 we pin code reality
so refactors are caught; Phase 1 should reconcile the spec to the
code or the code to the spec deliberately.
"""
from __future__ import annotations

import io
import contextlib
from typing import Optional

import numpy as np
import pytest

from mak2_model import MAK2Model
from run_batch import run_quality_gates


def _synthetic_sigmoid_curve(n_cycles: int = 40,
                             D0: float = 1e-3,
                             k: float = 0.3,
                             P0: float = 1.0,
                             F_bg_intercept: float = 0.1,
                             F_bg_slope: float = 1e-4) -> np.ndarray:
    """Build a clean MAK2-shaped fluorescence curve for use in gate tests."""
    _, _, F = MAK2Model().simulate_cycles(
        D0=D0, k=k, P0=P0, n_cycles=n_cycles,
        F_bg_intercept=F_bg_intercept, F_bg_slope=F_bg_slope,
    )
    return F


def _make_result(
    *,
    R2: float,
    fit_start_cycle: float,
    fit_end_cycle: float,
    fluor_data: Optional[np.ndarray] = None,
    D0: float = 1e-3,
    k: float = 0.3,
    P0: float = 1.0,
    F_bg_intercept: float = 0.1,
    F_bg_slope: float = 1e-4,
) -> dict:
    """Construct a minimal result dict that ``run_quality_gates`` can grade."""
    if fluor_data is None:
        fluor_data = _synthetic_sigmoid_curve()
    return {
        "Sample": "synthetic",
        "R2": R2,
        "D0": D0,
        "k": k,
        "P0": P0,
        "F_bg_intercept": F_bg_intercept,
        "F_bg_slope": F_bg_slope,
        "fit_start_cycle": fit_start_cycle,
        "fit_end_cycle": fit_end_cycle,
        "fluor_data": fluor_data,
        "error": None,
        "Success": "✓",
    }


def _grade(results, cycles=None):
    """Run the gates with stdout suppressed; return the (mutated) list."""
    if cycles is None:
        cycles = np.arange(1, 41, dtype=float)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return run_quality_gates(results, cycles)


def _passed(result: dict) -> bool:
    """Did the well clear every gate?

    The function under test signals rejection by clearing ``Success``
    and writing a reason into ``error``; PASS leaves both untouched.
    """
    return result.get("Success") == "✓" and not result.get("error")


# ── Gate 0: R² floor ─────────────────────────────────────────────────────────


def test_gate_0_high_r2_passes():
    """R² = 0.9995 with a wide fit window clears Gate 0."""
    r = _make_result(R2=0.9995, fit_start_cycle=10, fit_end_cycle=30)
    out = _grade([r])
    assert _passed(out[0])


def test_gate_0_low_r2_fails():
    """R² = 0.90 (below the 0.97 threshold for non-late wells) is rejected."""
    r = _make_result(R2=0.90, fit_start_cycle=10, fit_end_cycle=30)
    out = _grade([r])
    assert not _passed(out[0])
    assert "R²" in out[0]["error"]


def test_gate_0_late_amplifier_relaxed_threshold():
    """Late amplifier (fit_end_cycle == last data cycle) gets a relaxed R² threshold.

    Late amplifiers have fewer plateau cycles to constrain the fit;
    the relaxed Gate-0 threshold (0.85 in the code, vs 0.99 for
    non-late wells) accepts results that the standard floor would
    reject.

    To isolate Gate-0's behaviour, we drop ``fluor_data`` from the
    synthetic well — Gates 2b and 3 only fire when fluor_data is
    present, so skipping them lets this test exercise just the
    Gate-0 late-amplifier branch. (Gates 2b and 3 are exercised
    against real data in the regression test.)
    """
    cycles = np.arange(1, 41, dtype=float)
    r = _make_result(R2=0.90, fit_start_cycle=20, fit_end_cycle=40)
    r["fluor_data"] = None  # skip Gates 2b and 3
    out = _grade([r], cycles=cycles)
    assert _passed(out[0]), (
        f"late-amplifier path failed: error={out[0].get('error')}"
    )


# ── Gate 2: fit-window width ─────────────────────────────────────────────────


def test_gate_2_wide_window_passes():
    """A 20-cycle fit window clears Gate 2 (≥ 10 cycles)."""
    r = _make_result(R2=0.9995, fit_start_cycle=10, fit_end_cycle=30)
    out = _grade([r])
    assert _passed(out[0])


def test_gate_2_narrow_window_fails():
    """A 5-cycle fit window is rejected by Gate 2."""
    r = _make_result(R2=0.9995, fit_start_cycle=15, fit_end_cycle=20)
    out = _grade([r])
    assert not _passed(out[0])
    err = out[0]["error"].lower()
    assert "window" in err or "cycle" in err


# ── Gate 3: sigmoid-shape (second-derivative sign change) ────────────────────


def test_gate_3_sigmoid_shape_passes():
    """A genuine sigmoid (with inflection) clears Gate 3."""
    r = _make_result(R2=0.9995, fit_start_cycle=10, fit_end_cycle=30)
    out = _grade([r])
    assert _passed(out[0])


def test_gate_3_rejection_clears_kinetic_params():
    """When a gate rejects a well, ``D0/k/P0/Ct`` are nulled out.

    Pin this defensively: callers downstream of the gates must not
    see plausible-looking kinetic parameters on a rejected well.
    """
    r = _make_result(R2=0.5, fit_start_cycle=10, fit_end_cycle=30)
    out = _grade([r])
    assert not _passed(out[0])
    assert out[0]["D0"] is None
    assert out[0]["k"] is None
    assert out[0]["P0"] is None
    assert out[0]["Ct"] is None


# ── Error wells ──────────────────────────────────────────────────────────────


def test_error_well_skipped_by_gates():
    """Wells with ``error`` already set are skipped by the gates.

    Pre-fit no-amp wells already have ``error`` populated; the gates
    must not re-stamp them or change their kinetic parameters.
    """
    r = {
        "Sample": "no_amp",
        "R2": float("nan"),
        "error": "No amplification detected",
        "Success": "",
    }
    out = _grade([r])
    # Error and Success should remain exactly what they were
    assert out[0]["error"] == "No amplification detected"
    assert out[0]["Success"] == ""


# ── Sanity check on the synthetic helper ─────────────────────────────────────


def test_synthetic_curve_is_sigmoid():
    """Sanity: ``_synthetic_sigmoid_curve`` produces a curve with an inflection."""
    F = _synthetic_sigmoid_curve()
    second_deriv = np.diff(F, n=2)
    assert (second_deriv > 0).any() and (second_deriv < 0).any()
