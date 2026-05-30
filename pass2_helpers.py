"""Shared helpers for the channel-aware Pass 2 retry.

Pass 1 fits each well in isolation. After Pass 1 finishes, the
distribution of ``k`` / ``P0`` / ``F_bg_*`` across the plate is
informative — wells that landed in bad local minima or had pathological
fits can be rescued by re-fitting with the channel-typical values as
priors. This module owns the two pieces of that logic that are
mathematically identical across every Pass 2 caller (app.py batch and
``run_batch.run_pass2``):

  - ``compute_channel_priors``: per-channel and plate-wide medians of
    the kinetic + background parameters, computed over the subset of
    Pass 1 fits that look reliable.

  - ``identify_retry_candidates``: predicates flag a Pass 1 result for
    a retry. Includes the "hopeless wells" skip step that exempts late
    amplifiers from the R² < 0.85 cutoff.

The per-well retry *fitting* itself is not extracted here — the retry
loop has UI integration (app.py) vs CLI integration (run_batch)
differences that don't simplify cleanly. Both call sites still do the
fit themselves, but using the prior dicts and the index list that
these two functions produce.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


# Threshold below which we won't retry — pure noise/drift fits don't
# benefit from channel priors and just consume time. Late amplifiers
# are exempted because their priors-based retries do recover them.
HOPELESS_R2 = 0.85


def _channel_of(name: str) -> str:
    """Channel prefix extraction matching app.py / run_batch.py helpers."""
    if '::' in name:
        return name.split('::')[0]
    if '_' in name:
        return name.split('_')[0]
    return 'default'


def compute_channel_priors(results_list: list[dict]) -> tuple[dict, dict]:
    """Return ``(channel_medians, plate_medians)`` from Pass 1 results.

    "Reliable" Pass 1 fits (R² > 0.95, k < 0.5, Success ✓) contribute
    to the per-channel and plate-wide medians of k / P0 / F_bg_intercept
    / F_bg_slope. Channels need ≥ 2 reliable fits to enter
    ``channel_medians``; everything else falls back to ``plate_medians``
    when a retry needs a prior.

    The plate-wide fallback uses sensible defaults (k=0.15, P0=1e5,
    F_bg_intercept=1e5, F_bg_slope=0) when no reliable fits exist at
    all (e.g. early in a debug run).
    """
    ch_k: dict[str, list[float]] = {}
    ch_P0: dict[str, list[float]] = {}
    ch_Fbg: dict[str, list[float]] = {}
    ch_slope: dict[str, list[float]] = {}

    for r in results_list:
        if (r.get('k') is not None and r.get('R2') is not None
                and r['R2'] > 0.95 and r['k'] < 0.5
                and str(r.get('Success', '')).startswith('✓')):
            ch = _channel_of(r['Sample'])
            ch_k.setdefault(ch, []).append(r['k'])
            ch_P0.setdefault(ch, []).append(r['P0'])
            ch_Fbg.setdefault(ch, []).append(r['F_bg_intercept'])
            if r.get('F_bg_slope') is not None:
                ch_slope.setdefault(ch, []).append(r['F_bg_slope'])

    channel_medians: dict[str, dict] = {}
    for ch, ks in ch_k.items():
        if len(ks) >= 2:
            channel_medians[ch] = {
                'k':              float(np.median(ks)),
                'P0':             float(np.median(ch_P0[ch])),
                'F_bg_intercept': float(np.median(ch_Fbg[ch])),
                'F_bg_slope':     float(np.median(ch_slope.get(ch, [0.0]))),
                'n':              len(ks),
            }

    all_k    = [v for vs in ch_k.values()    for v in vs]
    all_P0   = [v for vs in ch_P0.values()   for v in vs]
    all_Fbg  = [v for vs in ch_Fbg.values()  for v in vs]
    all_Sl   = [v for vs in ch_slope.values() for v in vs]
    plate_medians = {
        'k':              float(np.median(all_k))   if all_k   else 0.15,
        'P0':             float(np.median(all_P0))  if all_P0  else 1e5,
        'F_bg_intercept': float(np.median(all_Fbg)) if all_Fbg else 1e5,
        'F_bg_slope':     float(np.median(all_Sl))  if all_Sl  else 0.0,
    }
    return channel_medians, plate_medians


def identify_retry_candidates(
    results_list: list[dict],
    cycles: np.ndarray,
    cycles_after_max: int,
) -> tuple[list[int], int]:
    """Return sorted indices of Pass 1 wells worth re-fitting in Pass 2.

    Five inclusion predicates (any one triggers a retry):

      (a) High SSR relative to fluorescence range, combined with
          R² < 0.999.
      (b) Optimisation failed entirely (k is None).
      (c) Degenerate k > 0.5 (unphysical for qPCR), combined with
          R² < 0.999.
      (d) R² below the 0.999 target.
      (e) "Tail overshoot": the model is consistently above the data
          in the last 3 cycles of the fit window, which happens when
          smart truncation cuts the window before the plateau and the
          optimizer ends up unconstrained on the late-cycle k.

    Wells with R² < 0.85 are exempted from retry (pure noise/drift
    can't be rescued) *unless* they are late amplifiers, defined as
    fits whose end cycle is within ``cycles_after_max`` of the last
    cycle — those benefit from extended-baseline retries.
    """
    # Local import avoids a circular import: pass2_helpers used by app.py
    # and run_batch.py, both of which sit above mak2_model in dep order.
    from mak2_model import MAK2Model

    last_cyc = float(cycles[-1]) if len(cycles) else 0.0
    late_margin = min(max(1, cycles_after_max), 5)
    retry: set[int] = set()

    for i, r in enumerate(results_list):
        fd = r.get('fluor_data')
        r2 = r.get('R2')

        # (a) high SSR with sub-target R²
        if (r.get('SSR') is not None and fd is not None
                and (r2 is None or r2 < 0.999)):
            F_rng = float(np.max(fd) - np.min(fd))
            if r['SSR'] > 0.01 * F_rng ** 2:
                retry.add(i)

        # (b) total optimisation failure
        if r.get('k') is None:
            retry.add(i)

        # (c) degenerate k
        if (r.get('k') is not None and r['k'] > 0.5
                and (r2 is None or r2 < 0.999)):
            retry.add(i)

        # (d) below target R²
        if r2 is not None and r2 < 0.999:
            retry.add(i)

        # (e) tail overshoot — model below data at end of fit window.
        if (r2 is not None and r2 < 0.999
                and fd is not None and r.get('error') is None):
            try:
                fe = r.get('fit_end_cycle')
                fs = r.get('fit_start_cycle')
                if (fe is not None and fs is not None
                        and not (isinstance(fe, float) and np.isnan(fe))):
                    c_arr = (cycles[:len(fd)] if len(cycles) >= len(fd)
                             else np.arange(1, len(fd) + 1, dtype=float))
                    win_mask = (c_arr >= fs) & (c_arr <= fe)
                    fd_win = fd[win_mask]
                    if r.get('D0') is not None and not np.isnan(r['D0']):
                        f_pred = MAK2Model().simulate_to_cycle(
                            D0=r['D0'], k=r['k'], P0=r['P0'],
                            cycles=c_arr[win_mask],
                            F_bg_intercept=r['F_bg_intercept'],
                            F_bg_slope=r['F_bg_slope'],
                        )
                        resid = fd_win - f_pred
                        last3 = float(np.mean(resid[-3:])) if len(resid) >= 3 else 0.0
                        F_rng = float(np.max(fd) - np.min(fd))
                        if F_rng > 0 and last3 < -0.03 * F_rng:
                            retry.add(i)
            except Exception:
                pass

    # Hopeless-well exemption: keep late amplifiers, drop everything
    # else below R² 0.85. Skip count returned so callers can surface
    # "skipped N wells" feedback in their UI.
    skipped = 0
    for i in list(retry):
        r2_i = results_list[i].get('R2')
        if r2_i is not None and r2_i < HOPELESS_R2:
            fe_i = results_list[i].get('fit_end_cycle')
            is_late = (fe_i is not None and fe_i >= last_cyc - late_margin)
            if not is_late:
                retry.discard(i)
                skipped += 1

    return sorted(retry), skipped
