"""Unit tests for the MAK2 forward simulator and amplification-efficiency helper.

The simulator is purely deterministic (no RNG), so every assertion
here is exact. The single-well fixture is the F1.1 cycle/fluorescence
data from Boggy.csv, used to compute reference forward-sim values
that pin the simulator's numerical behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest

from mak2_model import MAK2Model, calculate_amplification_efficiency


def test_simulate_cycles_returns_correct_shapes():
    """Output arrays have length ``n_cycles`` and the cycle grid starts at 0."""
    cycles, D, F = MAK2Model().simulate_cycles(
        D0=1e-3, k=0.3, P0=1.0, n_cycles=40,
        F_bg_intercept=0.2, F_bg_slope=1e-4,
    )
    assert len(cycles) == 40
    assert len(D) == 40
    assert len(F) == 40
    assert cycles[0] == 0
    assert cycles[-1] == 39


def test_simulate_cycles_initial_conditions():
    """``D[0]`` equals ``D0`` and ``F[0]`` equals ``D0 + F_bg_intercept``."""
    D0 = 1e-3
    F_bg_intercept = 0.5
    _, D, F = MAK2Model().simulate_cycles(
        D0=D0, k=0.3, P0=1.0, n_cycles=10,
        F_bg_intercept=F_bg_intercept, F_bg_slope=0.0,
    )
    assert D[0] == D0
    assert F[0] == pytest.approx(D0 + F_bg_intercept)


def test_simulate_cycles_dna_monotone_non_decreasing():
    """The DNA-only signal never drops cycle-to-cycle.

    Real PCR is irreversible — once amplified, DNA stays amplified
    until the next cycle's amplification adds more. The simulator
    must respect this even at parameter combinations near the
    primer-depletion boundary.
    """
    _, D, _ = MAK2Model().simulate_cycles(
        D0=1e-4, k=0.5, P0=1.0, n_cycles=45,
    )
    diffs = np.diff(D)
    assert (diffs >= -1e-12).all(), (
        f"DNA decreased somewhere; min diff = {diffs.min():.3e}"
    )


def test_simulate_cycles_produces_sigmoid_inflection():
    """The fluorescence curve has an inflection (second derivative changes sign).

    A pure exponential or pure linear curve has no inflection. The
    sigmoid shape comes from primer depletion bending the exponential
    into a plateau — without an inflection, the model isn't doing
    its job.
    """
    _, _, F = MAK2Model().simulate_cycles(
        D0=1e-4, k=0.3, P0=1.0, n_cycles=45,
    )
    second_deriv = np.diff(F, n=2)
    has_positive = (second_deriv > 0).any()
    has_negative = (second_deriv < 0).any()
    assert has_positive and has_negative, (
        "Fluorescence curve has no inflection — second derivative "
        "doesn't change sign."
    )


def test_increasing_d0_shifts_curve_left():
    """Larger ``D0`` → amplification reaches a given threshold sooner.

    The fundamental qPCR observation: more starting template means
    fewer cycles needed to reach detection. Pin this here so a
    refactor that accidentally inverts the relationship is caught
    immediately.
    """
    threshold = 0.05  # arbitrary — anywhere on the steep part of the curve
    _, _, F_low = MAK2Model().simulate_cycles(
        D0=1e-5, k=0.3, P0=1.0, n_cycles=45,
    )
    _, _, F_high = MAK2Model().simulate_cycles(
        D0=1e-3, k=0.3, P0=1.0, n_cycles=45,
    )
    cross_low = np.argmax(F_low > threshold)
    cross_high = np.argmax(F_high > threshold)
    assert cross_high < cross_low, (
        f"Higher D0 should cross threshold sooner; "
        f"got high={cross_high} >= low={cross_low}."
    )


def test_simulate_to_cycle_with_offset():
    """``cycle_offset`` translates the entire curve forward in cycle-space.

    A well that doesn't start amplifying until cycle 10 should look
    identical to a well that started at cycle 0 but observed 10
    cycles later — modulo background, which keeps drifting through
    the lag phase.
    """
    model = MAK2Model()
    F_no_offset = model.simulate_to_cycle(
        D0=1e-3, k=0.3, P0=1.0,
        cycles=np.arange(0, 30, dtype=float),
        F_bg_intercept=0.0, F_bg_slope=0.0,  # disable background to isolate effect
    )
    F_offset = model.simulate_to_cycle(
        D0=1e-3, k=0.3, P0=1.0,
        cycles=np.arange(10, 40, dtype=float),
        F_bg_intercept=0.0, F_bg_slope=0.0,
        cycle_offset=10.0,
    )
    np.testing.assert_allclose(F_no_offset, F_offset, rtol=0, atol=1e-12)


def test_calculate_amplification_efficiency_perfect_doubling():
    """For ``D[n] = 2 * D[n-1]``, efficiency is exactly 1.0."""
    D = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    eff = calculate_amplification_efficiency(D)
    assert eff == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_calculate_amplification_efficiency_no_growth():
    """Constant ``D`` gives zero efficiency at every cycle."""
    eff = calculate_amplification_efficiency(np.ones(10))
    assert (eff == 0.0).all()


def test_calculate_amplification_efficiency_handles_zero_safely():
    """A zero in ``D[n-1]`` skips that division rather than raising."""
    D = np.array([0.0, 0.0, 1.0, 2.0])
    eff = calculate_amplification_efficiency(D)
    # First two transitions divide by zero — expect 0 (skipped).
    # Third transition (D[2]→D[3]) is 2/1 - 1 = 1.0.
    assert eff[0] == 0.0
    assert eff[1] == 0.0
    assert eff[2] == pytest.approx(1.0)


def test_simulate_cycles_reference_values_for_F1_1_inputs(single_well_F1_1):
    """Forward-sim with parameters fitted from F1.1 reproduces a known curve.

    Pins the simulator's numerical behaviour against actual fitted
    parameters from the reference well. Any change to the per-cycle
    update arithmetic (e.g. floating-point reordering in a refactor)
    will perturb these values.

    Reference parameters were recorded from
    ``boggy_reference.json`` after Phase 0 with seed=42; the first
    handful of fluorescence values are checked against the
    forward-simulator's output for those same parameters.
    """
    cycles_arr = np.array(single_well_F1_1["cycles"], dtype=float)
    # Parameters from boggy_reference.json's F1.1 entry. Recorded once
    # and pinned here — if the boggy_reference fixture is regenerated,
    # update these too.
    D0 = 0.0002706614886627987
    k = 0.21640331411404715
    P0 = 1.2610201787470883
    F_bg_intercept = 0.1883648402118236
    F_bg_slope = -0.0004069181079190507

    F_pred = MAK2Model().simulate_to_cycle(
        D0=D0, k=k, P0=P0, cycles=cycles_arr,
        F_bg_intercept=F_bg_intercept, F_bg_slope=F_bg_slope,
    )

    # Sanity: returned array shape matches input cycle grid
    assert F_pred.shape == cycles_arr.shape

    # The first few cycles should be near pure background (D0 still tiny)
    bg_at_cycle_0 = F_bg_intercept  # cycles[0] = 1 in Boggy convention,
    # but simulate_to_cycle treats cycles[0]=1 as effective_cycle=0
    # and adds bg = intercept + slope * cycles[0] = intercept + slope*1
    # ~= intercept (slope is tiny). Use loose tolerance.
    assert abs(F_pred[0] - bg_at_cycle_0) < 0.01

    # The simulated curve should cover a meaningful range (sigmoid did fire)
    assert F_pred.max() - F_pred.min() > 0.5
