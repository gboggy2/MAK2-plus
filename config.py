"""Project-wide configuration constants.

Currently exposes a single setting — the random seed used by every
stochastic component in the engine (Latin Hypercube Sampling and
Differential Evolution inside the MAK2 optimizer, plus the bootstrap
resampling in ``bootstrap.py`` and the limited-dilution bootstrap in
``calibration.py``).

Usage:

  - **Production** (default): leave ``MAK2_RANDOM_SEED`` unset. The
    engine pulls fresh entropy on every run, so replicate fits of
    the same well show the optimizer's natural stochastic spread.
    This honestly conveys fit-confidence: when 5 runs of the same
    well produce 5 nearly-identical D0 values, the user knows the
    fit is robust; when they spread by 5%, the user knows there's
    real ambiguity.

  - **Testing / CI**: set ``MAK2_RANDOM_SEED=42`` (or any int). The
    engine seeds every stochastic call deterministically, so the
    full pipeline produces bit-identical output across runs and
    machines. This is what the Phase-0.5 regression test
    (``test_plate_regression.py``) relies on for its exact-equality
    assertions.

Implementation note: ``RANDOM_SEED`` here is the *base* seed.
Individual call sites in the optimizer derive their per-call seed
from this base plus a constant offset, so different LHS / DE / retry
loops don't share an identical sample sequence (which would degrade
exploration). See ``optimizer.py`` for the per-site offsets.
"""

import os
from dataclasses import dataclass

_seed_env = os.environ.get("MAK2_RANDOM_SEED")
RANDOM_SEED = int(_seed_env) if _seed_env is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Quality-gate thresholds
# ─────────────────────────────────────────────────────────────────────────────
#
# Each per-well fit is run through a series of gates that decide whether the
# fit's status is PASS / FAIL / INDETERMINATE. See ARCHITECTURE.md §7 for
# what each gate does and why; this module is the single place every
# threshold is *named*.
#
# Defaults below match the historical hardcoded values in run_batch.py at
# the moment QualityGateConfig was introduced — replacing those literals
# with reads from this config must be a no-op on existing fixtures. The
# subsequent gate-tuning work (see tuning/) will derive new defaults from
# the PCRedux labelled dataset and replace the values below.
#
# Tests and tuning experiments construct alternative configs; production
# code reads ``DEFAULT_GATES`` unless explicitly overridden.


@dataclass(frozen=True)
class QualityGateConfig:
    """Numeric and structural knobs for the per-well quality gates."""

    # ── Gate 0: R² floor ────────────────────────────────────────────────
    # The fitted R² must clear ``r2_floor_standard`` for a normal well, or
    # ``r2_floor_late_amplifier`` for a well whose fit window ended within
    # ``late_amplifier_tail_window`` cycles of the last data cycle. Late
    # amplifiers have less plateau information so the threshold relaxes.
    #
    # Floors were tuned against the PCRedux labelled dataset under the
    # restored fit_well preprocessing pipeline:
    #
    #   - ``r2_floor_standard = 0.98``: PCRedux's lowest-R² real amplifier
    #     among non-late wells is 0.984, so 0.98 admits every real
    #     amplifier (including dual-stage curves like maro1.299 whose
    #     single-sigmoid fit honestly maxes out at ~0.98). The original
    #     0.99 was set under the broken-pipeline regime where fits were
    #     artificially "perfect" because the long baseline dominated SSR.
    #
    #   - ``r2_floor_late_amplifier = 0.92``: PCRedux's lowest-R² real
    #     late amplifier is 0.946. 0.92 keeps every real late amp while
    #     rejecting the obvious "decreasing-then-rising" false PASSes
    #     (e.g., maro1.356, R²=0.908) that the old 0.85 floor admitted.
    r2_floor_standard: float = 0.98
    r2_floor_late_amplifier: float = 0.92
    late_amplifier_tail_window: int = 5

    # ── Gate 2: fit-window width ────────────────────────────────────────
    # The fit window must include at least this many cycles. Below this,
    # the 5-parameter MAK2 fit becomes over-determined in the wrong
    # direction (many parameter combinations fit equally well).
    min_fit_window_cycles: int = 10

    # ── Gate 2b: MAK2 vs linear discriminator ───────────────────────────
    # On the pre-inflection portion of the fit window, the MAK2 fit's R²
    # must beat a pure-linear fit by at least this much. Catches the case
    # where any monotone curve fits both models passably well.
    min_r2_gap_mak2_vs_linear: float = 0.05
    # Late amplifiers bypass Gate 2b when R² is at least this high (the
    # discriminator is unreliable in the short pre-inflection windows
    # late amplifiers produce).
    gate_2b_late_bypass_r2: float = 0.995

    # ── Gate 4: direction (raw-data rise) ───────────────────────────────
    # The raw fluorescence must end higher than it started. Computed on
    # the median of the first N and last N cycles for noise robustness
    # (``direction_anchor_window``). The growth (last − first) must
    # exceed ``min_growth_pct_of_range`` × total fluorescence range.
    #
    # Why this gate exists: the MAK2 model has a background term
    # (``F_bg_intercept + F_bg_slope × cycle``) that can absorb a real
    # *downward* trend in non-amplifying wells (photobleaching, dye
    # degradation, NTC drift), producing a fit with high R² and a
    # spurious tiny D0. Visual inspection of PCRedux false-PASSes
    # revealed exactly this pattern: 6 monotonically decreasing curves
    # passed the post-fit quality gates with R² > 0.995. Gate 4 catches
    # them on raw data — it's model-independent, so the optimizer can't
    # game it by laundering signal into background.
    #
    # Complements ``data_processing.detect_no_signal_samples`` (which
    # uses plate-wide range comparison); Gate 4 is per-well and works
    # equally well for single-curve usage (PCRedux scoring, future
    # API single-sample mode).
    direction_anchor_window: int = 5
    min_growth_pct_of_range: float = 0.05

    # ── Gate 3: sigmoid-shape (second-derivative sign change) ──────────
    # The fitted curve's second derivative must change sign within the
    # fit window — confirms there's an inflection point. The threshold
    # for "significant" magnitude is expressed as a fraction of the
    # in-window fluorescence range.
    inflection_threshold_pct_of_range: float = 0.01
    # High-R² wells with a wide-enough window bypass Gate 3 (the sigmoid
    # shape is implied by the fit quality).
    gate_3_high_r2_bypass_r2: float = 0.999
    gate_3_high_r2_bypass_min_window: int = 10
    # Late amplifiers bypass Gate 3 when R² is at least this high.
    gate_3_late_bypass_r2: float = 0.995


# The active gate parameters used by run_batch.run_quality_gates() and any
# downstream caller. Tuning experiments construct alternative
# QualityGateConfig instances and pass them explicitly; production reads
# this default.
DEFAULT_GATES = QualityGateConfig()
