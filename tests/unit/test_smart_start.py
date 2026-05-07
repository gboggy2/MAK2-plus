"""Unit tests for the inflection-detection helpers.

``mak2_model.find_slope_threshold_cycle`` is the deterministic
truncation helper used by the optimizer to cut off the noisy tail.
The tests exercise it on synthetic curves where the truth is known.
"""
from __future__ import annotations

import numpy as np
import pytest

from mak2_model import find_slope_threshold_cycle


def _synthetic_sigmoid(n_cycles: int, inflection: int, sharpness: float = 0.5) -> np.ndarray:
    """Build a clean logistic sigmoid with its inflection at ``inflection``."""
    cycles = np.arange(n_cycles, dtype=float)
    return 1.0 / (1.0 + np.exp(-sharpness * (cycles - inflection)))


def test_find_slope_threshold_cycle_locates_known_inflection():
    """Synthetic sigmoid with inflection at cycle 20 → cutoff is near 20+offset.

    The function returns ``inflection_cycle + cycles_after_max``;
    with the synthetic sigmoid the inflection should land within ±1
    cycle of the truth (some imprecision from the discrete max).
    """
    fluor = _synthetic_sigmoid(n_cycles=40, inflection=20)
    cutoff = find_slope_threshold_cycle(fluor, cycles_after_max=3)
    # Expected: inflection ≈ 20, so cutoff ≈ 23. Allow ±1 for argmax slop.
    assert 22 <= cutoff <= 24, f"cutoff={cutoff} outside expected band [22, 24]"


def test_find_slope_threshold_cycle_late_inflection():
    """Sigmoid with very late inflection → cutoff near the end of the array.

    Late amplifiers (low template) inflect close to the run end.
    The function should still return a valid index, capped at the
    last cycle.
    """
    fluor = _synthetic_sigmoid(n_cycles=45, inflection=38)
    cutoff = find_slope_threshold_cycle(fluor, cycles_after_max=3)
    # Inflection at 38, +3 = 41, but capped at 44 (last index).
    assert 40 <= cutoff <= 44


def test_find_slope_threshold_cycle_flat_array():
    """A flat array (no amplification) returns the last cycle.

    The safety net: a curve with no positive max-slope is a failed
    well; the function tells the caller "use everything" so the
    optimizer can fail gracefully on its own quality gates rather
    than getting confused by a zero-length truncation.
    """
    fluor = np.full(40, 0.1)
    cutoff = find_slope_threshold_cycle(fluor)
    assert cutoff == 39


def test_find_slope_threshold_cycle_too_short_returns_last_index():
    """Arrays shorter than the 5-point stencil return the last index defensively."""
    fluor = np.array([0.1, 0.2, 0.3])  # only 3 points
    cutoff = find_slope_threshold_cycle(fluor)
    assert cutoff == 2  # last valid index


def test_find_slope_threshold_cycle_cycles_after_max_offset():
    """The ``cycles_after_max`` offset is added to the inflection.

    Larger ``cycles_after_max`` should push the cutoff later. The
    direction of the shift is what matters here, not the precise
    value (which is bounded by the array length).
    """
    fluor = _synthetic_sigmoid(n_cycles=40, inflection=20)
    cutoff_3 = find_slope_threshold_cycle(fluor, cycles_after_max=3)
    cutoff_8 = find_slope_threshold_cycle(fluor, cycles_after_max=8)
    assert cutoff_8 > cutoff_3, (
        f"Larger cycles_after_max should give later cutoff; "
        f"got {cutoff_3} vs {cutoff_8}."
    )


def test_find_slope_threshold_cycle_real_well(boggy_input_df):
    """On a real qPCR curve (Boggy F1.1), the cutoff lands inside the curve.

    The strongest claim here is just "doesn't return nonsense" — a
    real plate is the integration test. F1.1 is undiluted, so it
    amplifies early; cutoff should be in the first half of the curve
    plus a few cycles.
    """
    fluor = boggy_input_df["F1.1"].to_numpy(dtype=float)
    cutoff = find_slope_threshold_cycle(fluor, cycles_after_max=3)
    assert 5 <= cutoff <= len(fluor) - 1
