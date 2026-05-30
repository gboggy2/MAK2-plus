"""Staged exponential pre-fitting for MAK2+.

Stage 0: empirical exponential fit ``D0_emp * r^n`` over
``[window_start, inflection]`` -> ``Ct_stage0`` via chord-slope-delta.

Stage 1: one-parameter doubling fit ``D0_toe * 2^(n - n_ref)`` over the toe
window ``[Ct_stage0 - TOE_WINDOW_CYCLES, Ct_stage0]`` -> ``D0_toe``,
``toe_fit_r2``.

In the MAK2 toe region (primers ~undepleted, template scarce) the model
reduces analytically to perfect doubling, so the base of 2 is physics, not
a regression choice. This lets Stage 1 estimate D0 with one parameter and
gives Stage 2 a tight, principled D0 prior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Defaults (exposed as module constants so callers / tests can override).
TOE_WINDOW_CYCLES = 4
TOE_MIN_SNR = 2.0
TOE_FIT_R2_MIN = 0.80
TOE_D0_BOUND_FACTOR = 4.0
TOE_RESIDUAL_THRESHOLD = 0.10
# Stage 0 Ct definition: fixed offset before the inflection cycle. The
# MAK2 toe region reduces analytically to perfect doubling (r=2), and with
# r=2 the slope-fraction Ct rule ``s(n) >= k * s_max`` collapses to
# ``Ct = inflection - log2(1/k)``. We pick the offset directly (2 cycles
# before inflection) rather than carry the empirical exponential fit, which
# was biased low (r ~ 1.4) because the regression averaged across the
# primer-depletion region. Stage 1 still does the actual one-parameter
# doubling fit over [Ct-4, Ct]; Stage 0 is now just window placement.
CT_INFLECTION_OFFSET = 2


@dataclass
class Stage0Result:
    success: bool
    Ct_stage0: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class Stage3Result:
    evaluated: bool
    passed: Optional[bool] = None
    mean_residual: Optional[float] = None
    rel_residual: Optional[float] = None
    local_range: Optional[float] = None
    sign: int = 0  # +1 model under data, -1 model over data
    reason: Optional[str] = None


@dataclass
class Stage1Result:
    success: bool
    D0_toe: Optional[float] = None
    toe_fit_r2: Optional[float] = None
    toe_window_start: Optional[float] = None
    toe_window_end: Optional[float] = None
    n_points: int = 0
    snr: Optional[float] = None
    reason: Optional[str] = None


def _background_corrected(
    cycles: np.ndarray,
    fluor_data: np.ndarray,
    bg_int: float,
    bg_slope: float,
) -> np.ndarray:
    return fluor_data - (bg_int + bg_slope * cycles)


def stage0_ct_from_inflection(
    cycles: np.ndarray,
    *,
    inflection_idx: int,
    offset: int = CT_INFLECTION_OFFSET,
) -> Stage0Result:
    """Locate Ct as a fixed cycle-offset before the inflection.

    With r=2 (the MAK2 toe-region limit) the slope-fraction criterion
    ``s(n) >= k * s_max`` collapses to ``Ct = inflection - log2(1/k)``, so
    we set Ct deterministically rather than fitting an empirical
    exponential whose r is biased by the primer-depletion tail of the
    pre-inflection rise.
    """
    cycles = np.asarray(cycles, dtype=float)
    if inflection_idx < offset:
        return Stage0Result(success=False, reason="inflection too early for offset")
    Ct_idx = inflection_idx - offset
    return Stage0Result(success=True, Ct_stage0=float(cycles[Ct_idx]))


def stage1_toe_fit(
    cycles: np.ndarray,
    fluor_data: np.ndarray,
    *,
    bg_int: float,
    bg_slope: float,
    Ct_stage0: float,
    baseline_std: float,
    toe_window_cycles: int = TOE_WINDOW_CYCLES,
    min_snr: float = TOE_MIN_SNR,
) -> Stage1Result:
    """Fit ``D0_toe * 2^(n - n_ref)`` over [Ct - W, Ct] via constant-only OLS.

    The base is fixed at 2 (MAK2 toe-region limit). Skips if mean signal in
    the toe window is below ``min_snr * baseline_std`` above zero (post bg
    subtraction), since fitting a doubling model into noise is meaningless.
    """
    cyc_all = np.asarray(cycles, dtype=float)
    win_lo = Ct_stage0 - float(toe_window_cycles)
    win_hi = Ct_stage0
    mask = (cyc_all >= win_lo) & (cyc_all <= win_hi)
    if int(mask.sum()) < 3:
        return Stage1Result(success=False, reason="<3 cycles in toe window")

    cyc = cyc_all[mask]
    flu = np.asarray(fluor_data, dtype=float)[mask]
    F_corr = flu - (bg_int + bg_slope * cyc)

    mean_corr = float(np.mean(F_corr))
    snr = mean_corr / baseline_std if baseline_std > 0 else float("inf")
    if not np.isfinite(snr) or snr < min_snr:
        return Stage1Result(
            success=False, snr=snr, reason=f"SNR {snr:.2f} < {min_snr}",
            toe_window_start=float(cyc[0]), toe_window_end=float(cyc[-1]),
        )

    # Linear-space one-parameter fit (was log-space until 2026-05).
    #
    # Model: F_corr(n) = D0_toe * 2^n, with r = 2 fixed by the MAK2 toe-
    # limit physics. Linear-in-D0_toe → solved analytically by OLS
    # through origin on the transformed variable m_i = 2^(n_i - n_ref):
    #
    #     F_corr_i = (D0_toe * 2^n_ref) * m_i  =:  beta * m_i
    #     beta_hat = sum(F_corr_i * m_i) / sum(m_i^2)
    #     D0_toe   = beta_hat / 2^n_ref
    #
    # The n_ref shift is purely for numerical hygiene (keeps m_i values
    # in the range [1, 2^window_cycles] instead of [2^13, 2^17] for a
    # typical toe). The estimator is mathematically identical.
    #
    # Why linear space (the old log-space fit's failure mode): log(F)
    # amplifies low-signal noise enormously — a 0.005 RFU baseline-
    # subtraction error on a F=0.01 point becomes a ±0.5 swing in
    # log2(F), dominating the slope fit. In linear space the same error
    # stays a 0.005 RFU residual with leverage proportional only to its
    # distance from the fit line. The change also lets us include
    # cycles where F_corr <= 0 (which log() can't represent), so n_pts
    # equals the configured window size.
    n = cyc
    y = F_corr
    n_ref = float(n[0])
    m = np.power(2.0, n - n_ref)
    denom = float(np.sum(m * m))
    if denom <= 0:
        return Stage1Result(
            success=False, snr=snr, reason="degenerate fit window",
            toe_window_start=float(cyc[0]), toe_window_end=float(cyc[-1]),
        )
    beta = float(np.sum(y * m) / denom)
    D0_toe = beta / (2.0 ** n_ref)

    # If the fit collapsed to a non-positive D0 (only possible when most
    # toe-window points are negative — i.e. nearly pure noise), skip
    # Stage 1 rather than hand the optimizer a bogus prior.
    if not np.isfinite(D0_toe) or D0_toe <= 0:
        return Stage1Result(
            success=False, snr=snr, reason="fit returned non-positive D0",
            toe_window_start=float(cyc[0]), toe_window_end=float(cyc[-1]),
        )

    # R^2 in linear F space — directly comparable to Stage 3's
    # toe_rel_residual (both live in raw bg-corrected fluorescence units).
    y_hat = beta * m
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    toe_fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return Stage1Result(
        success=True,
        D0_toe=D0_toe,
        toe_fit_r2=toe_fit_r2,
        toe_window_start=float(cyc[0]),
        toe_window_end=float(cyc[-1]),
        n_points=int(len(n)),
        snr=snr,
    )


def stage3_toe_gate(
    cycles: np.ndarray,
    fluor_data: np.ndarray,
    fit_params: dict,
    *,
    Ct_stage0: float,
    toe_window_cycles: int = TOE_WINDOW_CYCLES,
    threshold: float = TOE_RESIDUAL_THRESHOLD,
) -> Stage3Result:
    """Compute mean (observed - predicted) in the toe window.

    Fails the gate when ``|mean_residual| / local_signal_range > threshold``.
    Sign is +1 when the model sits below the data (D0 likely too low /
    window too late), -1 when the model sits above (D0 too high).
    """
    # Local import: MAK2Model lives in mak2_model and importing at module
    # scope would create a cycle (mak2_model -> ... -> toe_prefit).
    from mak2_model import MAK2Model

    cyc_all = np.asarray(cycles, dtype=float)
    win_lo = Ct_stage0 - float(toe_window_cycles)
    win_hi = Ct_stage0
    mask = (cyc_all >= win_lo) & (cyc_all <= win_hi)
    if int(mask.sum()) < 3:
        return Stage3Result(evaluated=False, reason="<3 cycles in toe window")

    cyc = cyc_all[mask]
    obs = np.asarray(fluor_data, dtype=float)[mask]

    model = MAK2Model()
    pred = model.simulate_to_cycle(
        fit_params['D0'], fit_params['k'], fit_params['P0'], cyc,
        F_bg_intercept=fit_params['F_bg_intercept'],
        F_bg_slope=fit_params['F_bg_slope'],
    )

    residuals = obs - pred
    mean_resid = float(np.mean(residuals))
    local_range = float(obs.max() - obs.min())
    if local_range <= 0:
        return Stage3Result(evaluated=False, reason="zero local signal range")
    rel = abs(mean_resid) / local_range
    return Stage3Result(
        evaluated=True,
        passed=(rel <= threshold),
        mean_residual=mean_resid,
        rel_residual=rel,
        local_range=local_range,
        sign=int(np.sign(mean_resid)),
    )


def baseline_std_from_prefit(
    cycles: np.ndarray,
    fluor_data: np.ndarray,
    bg_int: float,
    bg_slope: float,
    floor_idx: int,
    baseline_end_idx: int,
) -> float:
    """Std of background-corrected fluorescence over the pre-amplification region.

    Used as the noise floor for the Stage 1 SNR precondition.
    """
    if baseline_end_idx - floor_idx < 3:
        return 0.0
    cyc = np.asarray(cycles[floor_idx:baseline_end_idx], dtype=float)
    flu = np.asarray(fluor_data[floor_idx:baseline_end_idx], dtype=float)
    F_corr = flu - (bg_int + bg_slope * cyc)
    return float(np.std(F_corr))
