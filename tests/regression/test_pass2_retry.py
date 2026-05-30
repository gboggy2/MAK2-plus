"""Pass 2 retry regression — locks current behavior of the channel-aware
retry pass before it gets unified.

The Boggy fixture fits cleanly on every well, so Pass 2 is a no-op there.
This test uses 10 ``y``-labelled curves from the competimer dataset; under
the current pipeline, Pass 1 leaves ~2 wells with R² < 0.999 that Pass 2
attempts to rescue. The reference JSON captures the *post-Pass-2* state
of all 10 wells (which fields Pass 1 + Pass 2 set, what tier they end up
in, and the kinetic numbers).

When the Pass 2 retry loop is refactored, this test catches any drift in
output: status must match exactly; D0 / R² / Ct must match within the
same tolerance the plate regression uses.
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
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_batch import run_pass1, run_pass2  # noqa: E402

REF = ROOT / "tests" / "fixtures" / "competimer_pass2_reference.json"
TOLERANCE_MODE = os.environ.get("REGRESSION_TOLERANCE_MODE") == "1"


def _load_competimer_subset():
    """Load the 10 y-labelled competimer curves used by the reference."""
    with open(REF) as fh:
        ref_doc = json.load(fh)
    sample_ids = ref_doc["samples"]
    cycles = np.array(ref_doc["cycles"], dtype=float)
    cdir = ROOT / "tests" / "fixtures" / "pcredux_extended" / "competimer"
    curves = pd.read_csv(cdir / "curves.csv")
    samples = {}
    for cid in sample_ids:
        sub = curves.loc[curves["curve_id"] == cid].sort_values("cycle")
        samples[cid] = sub["fluorescence"].to_numpy(dtype=float)
    return samples, cycles, ref_doc["results"]


def _run_pipeline(samples, cycles):
    """Run Pass 1 + Pass 2 with the same empty-metadata invocation the
    reference was captured under."""
    os.environ["MAK2_RANDOM_SEED"] = "42"
    with contextlib.redirect_stdout(io.StringIO()):
        results = run_pass1(
            samples, cycles,
            sample_metadata={}, rox_by_well={}, channel_thresholds={},
            global_threshold=None,
            channel_baseline_means={}, global_baseline_mean=None,
        )
        run_pass2(
            results, cycles,
            sample_metadata={}, rox_by_well={}, channel_thresholds={},
            global_threshold=None,
            channel_baseline_means={}, global_baseline_mean=None,
        )
    return results


@pytest.fixture(scope="module")
def competimer_results():
    samples, cycles, ref = _load_competimer_subset()
    got = _run_pipeline(samples, cycles)
    by_sample = {r["Sample"]: r for r in got}
    return by_sample, ref


def test_pass2_status_unchanged(competimer_results):
    """Pass/fail status per well must match the reference exactly."""
    got, ref = competimer_results
    mismatches = []
    for ref_row in ref:
        s = ref_row["Sample"]
        if got[s].get("Success") != ref_row["Success"]:
            mismatches.append(f"{s}: got Success={got[s].get('Success')!r} "
                              f"vs ref={ref_row['Success']!r}")
    assert not mismatches, "Status drift:\n  " + "\n  ".join(mismatches)


def test_pass2_kinetics_within_tolerance(competimer_results):
    """D0, R², Ct, Tier must match within the configured tolerance.

    In default (exact) mode: bit-equality on D0/R²/Ct, exact Tier match.
    In tolerance mode (REGRESSION_TOLERANCE_MODE=1): D0 within ±5%, R²
    within ±0.001, Ct within ±0.1 cycles, Tier exact.
    """
    got, ref = competimer_results
    fails = []
    for ref_row in ref:
        s = ref_row["Sample"]
        g = got[s]
        if g.get("Success") != ref_row["Success"]:
            continue  # status mismatch already covered

        # Tier always exact when both fits succeeded.
        if ref_row["Tier"] is not None and g.get("Tier") != ref_row["Tier"]:
            fails.append(f"{s}: Tier {g.get('Tier')!r} vs ref {ref_row['Tier']!r}")

        for field, tol in [("D0", 0.05), ("R2", 0.001), ("Ct", 0.1)]:
            ref_v, got_v = ref_row.get(field), g.get(field)
            if ref_v is None or got_v is None:
                if ref_v != got_v:
                    fails.append(f"{s}: {field} None mismatch "
                                 f"(got={got_v!r}, ref={ref_v!r})")
                continue
            ref_v, got_v = float(ref_v), float(got_v)
            if not np.isfinite(ref_v) or not np.isfinite(got_v):
                if np.isfinite(ref_v) != np.isfinite(got_v):
                    fails.append(f"{s}: {field} finite-ness mismatch")
                continue
            if TOLERANCE_MODE:
                if field == "D0":
                    rel = abs(got_v - ref_v) / abs(ref_v) if ref_v != 0 else 0
                    if rel > tol:
                        fails.append(f"{s}: D0 {got_v:.4e} vs {ref_v:.4e} "
                                     f"(rel diff {rel:.2%} > {tol:.0%})")
                else:
                    if abs(got_v - ref_v) > tol:
                        fails.append(f"{s}: {field} {got_v:.4f} vs {ref_v:.4f} "
                                     f"(|diff| {abs(got_v-ref_v):.4f} > {tol})")
            else:
                if got_v != ref_v:
                    fails.append(f"{s}: {field} {got_v!r} != {ref_v!r}")
    assert not fails, f"{len(fails)} regression failure(s):\n  " + "\n  ".join(fails)


def test_pass2_actually_runs_retries():
    """Sanity: this fixture is only meaningful if Pass 1 alone leaves at
    least one well with R²<0.999 (which is what triggers Pass 2). If the
    fixture or Pass 1 changes such that nothing needs retrying, this test
    loses its safety-net value — flag that loudly here.
    """
    samples, cycles, _ = _load_competimer_subset()
    os.environ["MAK2_RANDOM_SEED"] = "42"
    with contextlib.redirect_stdout(io.StringIO()):
        pass1_only = run_pass1(
            samples, cycles,
            sample_metadata={}, rox_by_well={}, channel_thresholds={},
            global_threshold=None,
            channel_baseline_means={}, global_baseline_mean=None,
        )
    n_below_target = sum(
        1 for r in pass1_only
        if r.get("R2") is not None and float(r["R2"]) < 0.999
    )
    assert n_below_target >= 1, (
        f"Pass 1 alone produced 0 wells with R²<0.999 on the competimer "
        f"fixture, so Pass 2 had nothing to retry. The other tests in this "
        f"file no longer guard the retry code path."
    )
