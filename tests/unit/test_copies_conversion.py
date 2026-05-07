"""Unit tests for D0 → copies conversion (calibration arithmetic).

Tests the ``apply_calibration`` paths only — not the curve-fitting
in ``build_standard_curve``, which needs a fitted-results DataFrame
that's harder to manufacture in a unit test (covered by the
regression test instead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibration import apply_calibration


def test_apply_calibration_with_manual_cf():
    """``manual_cf`` path: ``Copies_D0 = manual_cf * D0`` for each row.

    The simplest calibration: a single per-instrument conversion
    factor multiplied against D0. Used when standards aren't
    available and the user has previously calibrated the instrument
    (D0_single).
    """
    df = pd.DataFrame({"Sample": ["A1", "A2"], "D0": [0.038, 0.0001]})
    manual_cf = 1 / 3.87e-8  # ≈ 2.58e7 — the spec's reference value
    out = apply_calibration(df, manual_cf=manual_cf)
    assert "Copies_D0" in out.columns
    np.testing.assert_allclose(
        out["Copies_D0"].values,
        [0.038 * manual_cf, 0.0001 * manual_cf],
    )


def test_apply_calibration_reference_arithmetic():
    """The CLAUDE.md spec value: ``D0=0.03848``, ``D0_single=3.87e-8`` → ~994067 copies.

    Pinned in CLAUDE.md as the canonical reference for this conversion.
    Catches any future change to the formula.
    """
    df = pd.DataFrame({"Sample": ["A1"], "D0": [0.03848]})
    manual_cf = 1 / 3.87e-8
    out = apply_calibration(df, manual_cf=manual_cf)
    expected = 0.03848 / 3.87e-8
    assert out["Copies_D0"].iloc[0] == pytest.approx(expected, rel=0.001)
    # Spot-check the spec value: ~994067
    assert 990000 < out["Copies_D0"].iloc[0] < 1000000


def test_apply_calibration_handles_missing_d0():
    """Wells with NaN or zero ``D0`` (failed fits) get ``NaN`` in ``Copies_D0``.

    The optimizer reports NaN/0 for failed wells; the calibration
    must not raise on them and must not fabricate copy numbers from
    invalid D0.
    """
    df = pd.DataFrame({
        "Sample": ["A1", "A2", "A3", "A4"],
        "D0": [0.038, np.nan, 0.0, -1e-5],  # valid, NaN, zero, negative
    })
    manual_cf = 1e7
    out = apply_calibration(df, manual_cf=manual_cf)
    assert not np.isnan(out["Copies_D0"].iloc[0])
    assert np.isnan(out["Copies_D0"].iloc[1])
    assert np.isnan(out["Copies_D0"].iloc[2])
    assert np.isnan(out["Copies_D0"].iloc[3])


def test_apply_calibration_with_power_law():
    """``calibration`` path: ``copies = 10^(slope * log10(D0) + intercept)``.

    The power-law form (from ``build_standard_curve``) can absorb
    the ~4% MAK2 D0-compression bias by having slope ≠ 1. Verify
    the arithmetic against a manually computed reference.
    """
    df = pd.DataFrame({"Sample": ["A1"], "D0": [1e-3]})
    cal = {"slope": 0.95, "intercept": 4.0}  # synthetic
    out = apply_calibration(df, calibration=cal)
    # 10^(0.95 * log10(1e-3) + 4.0) = 10^(0.95 * -3 + 4) = 10^1.15
    expected = 10 ** 1.15
    assert out["Copies_D0"].iloc[0] == pytest.approx(expected, rel=1e-9)


def test_apply_calibration_no_op_when_both_missing():
    """No ``calibration`` and no ``manual_cf`` → returned unchanged, no Copies column."""
    df = pd.DataFrame({"Sample": ["A1"], "D0": [0.038]})
    out = apply_calibration(df)
    assert "Copies_D0" not in out.columns


def test_apply_calibration_does_not_mutate_input():
    """The caller's DataFrame is not modified (returns a copy)."""
    df = pd.DataFrame({"Sample": ["A1"], "D0": [0.038]})
    out = apply_calibration(df, manual_cf=1e7)
    assert "Copies_D0" not in df.columns
    assert "Copies_D0" in out.columns
