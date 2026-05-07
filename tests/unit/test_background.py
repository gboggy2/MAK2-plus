"""Unit tests for the background-regression helpers.

``mak2_model.pre_estimate_background`` is a thin linear regression on
the pre-amplification baseline window — the workhorse of the
"background separation" architectural decision (see ARCHITECTURE.md
§8). These tests pin its arithmetic and the defensive fallbacks for
degenerate inputs.

``data_processing.estimate_baseline_end`` is the algorithmic
discovery of where the baseline ends — used as a fallback / sanity
check against instrument metadata.
"""
from __future__ import annotations

import numpy as np
import pytest

from data_processing import estimate_baseline_end
from mak2_model import pre_estimate_background


def test_pre_estimate_background_flat_baseline():
    """A flat baseline returns slope ≈ 0 and intercept ≈ the constant level.

    The threshold (1% relative tolerance) is much wider than the
    actual numerical residual, but it captures the meaningful claim:
    the regression must recover the constant level it sees.
    """
    cycles = np.arange(1, 21, dtype=float)
    fluor = np.full(20, 0.5)
    slope, intercept = pre_estimate_background(cycles, fluor, 0, 15)
    assert abs(slope) < 1e-10
    assert intercept == pytest.approx(0.5, rel=0.01)


def test_pre_estimate_background_linear_drift():
    """A baseline with known slope+intercept is recovered exactly.

    ``F = 1.0 + 0.01 * cycle`` is the canonical "instrument with
    photobleaching drift" pattern; the regression must read off the
    coefficients within numerical tolerance.
    """
    cycles = np.arange(1, 21, dtype=float)
    fluor = 1.0 + 0.01 * cycles
    slope, intercept = pre_estimate_background(cycles, fluor, 0, 15)
    assert slope == pytest.approx(0.01, rel=1e-6)
    assert intercept == pytest.approx(1.0, rel=1e-6)


def test_pre_estimate_background_too_few_points_falls_back_to_mean():
    """A 1-point window can't be regressed; defensive fallback returns the mean.

    The function is called from many code paths and must not raise on
    pathological inputs (1-point baseline windows happen on bad
    plates with corrupt metadata).
    """
    cycles = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fluor = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    slope, intercept = pre_estimate_background(cycles, fluor, 0, 1)
    assert slope == 0.0
    assert intercept == pytest.approx(np.mean(fluor))


def test_pre_estimate_background_returns_floats_not_arrays():
    """Return values must be Python floats, not 0-d numpy arrays.

    Downstream code does arithmetic on the returned values without
    extra type-coercion; if numpy 0-d arrays leak through, they
    propagate quietly and break things like dict serialisation.
    """
    cycles = np.arange(1, 21, dtype=float)
    fluor = np.linspace(0.1, 0.5, 20)
    slope, intercept = pre_estimate_background(cycles, fluor, 0, 15)
    assert isinstance(slope, float)
    assert isinstance(intercept, float)


def test_estimate_baseline_end_flat_then_amplification():
    """Synthetic curve: flat baseline → exponential takeoff → flat plateau.

    The detector should pick up the takeoff cycle. Threshold is loose
    (within ±3 cycles of the synthetic takeoff at cycle 20) because
    the algorithm is iterative and the precise verdict depends on the
    noise floor of the regression — anywhere in the takeoff region
    is correct.
    """
    cycles = np.arange(1, 41, dtype=float)
    rng = np.random.default_rng(42)  # deterministic for the test
    noise = rng.normal(0, 0.005, 40)
    # Baseline cycles 1-19, exponential 20-30, plateau 30-40
    flat = np.full(40, 0.1) + noise
    exp = np.where(cycles >= 20, 0.1 * (1.8 ** (cycles - 20)), 0)
    fluor = np.minimum(flat + exp, 5.0)  # cap plateau at 5.0

    bl_end = estimate_baseline_end(cycles, fluor, first_cycle_idx=2)
    # bl_end is an exclusive index — the boundary should be in the
    # cycle range where amplification visibly starts.
    assert 14 <= bl_end <= 25, (
        f"baseline_end_idx={bl_end} not in plausible range [14, 25]; "
        "the detector is missing the amplification onset."
    )


def test_estimate_baseline_end_pure_baseline_no_amplification():
    """A flat baseline with no amplification returns a sensible default.

    The algorithm should not raise. The exact return value is the
    initial window end (no extension iterations triggered).
    """
    cycles = np.arange(1, 41, dtype=float)
    rng = np.random.default_rng(42)
    fluor = np.full(40, 0.1) + rng.normal(0, 0.005, 40)

    bl_end = estimate_baseline_end(
        cycles, fluor, first_cycle_idx=2, window_size=12
    )
    # No amplification → no extension. Initial window is [2, 14).
    assert bl_end == 14
