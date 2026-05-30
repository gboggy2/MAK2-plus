"""Guard against re-introduced duplication of the workflow-canonical code.

Per ARCHITECTURE.md ("The no-divergence rule"), the per-well fitting
pipeline lives in:

    - fit_well.py                     (fit_well, prepare_fit_inputs)
    - pass2_helpers.py                (retry_one_well, channel priors)
    - toe_prefit.py                   (toe stages)
    - run_batch.run_quality_gates     (gates)

Every driver — app.py, run_batch.py — calls into those. When a future
change tries to inline fitting logic into a driver file again (the
historical failure mode that produced three drifted copies of the
pipeline), this test fails CI and points the contributor at the
canonical module.

If you have a *legitimate* reason to bypass the canonical file
(probably you don't — talk to a maintainer first), add the call site
to the per-pattern allowlist below with a comment explaining the
exception.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

import pytest

ROOT = Path(__file__).resolve().parents[2]

# ─── Files that are DRIVERS (must not inline fit logic) ──────────────────
DRIVER_FILES = ["app.py", "run_batch.py"]

# ─── Files where the patterns below are EXPECTED ─────────────────────────
# Canonical files own the fit logic by definition; tests may instantiate
# the optimizer directly for narrow-scope unit testing.
CANONICAL_FILES = {
    "fit_well.py", "pass2_helpers.py", "toe_prefit.py",
    "optimizer.py", "mak2_model.py", "run_batch.py",
}


def _read(path: Path) -> str:
    return path.read_text()


# ─────────────────────────────────────────────────────────────────────────
# RULE 1 — Drivers must not instantiate MAK2Optimizer directly.
# (Single exception: run_batch.py owns run_quality_gates, which doesn't
# use the optimizer; but it also has bootstrap/calibration helpers that
# legitimately do. We allow run_batch.py overall but flag app.py strictly.)
# ─────────────────────────────────────────────────────────────────────────
def test_app_py_does_not_instantiate_optimizer_directly():
    """app.py must route per-well fits through fit_well, not call
    MAK2Optimizer directly. Allowlisted exceptions: helpers that need
    the optimizer's state for plot reconstruction or post-fit
    inspection (predict/calculate_ct/etc), or bootstrap re-fits."""
    src = _read(ROOT / "app.py")
    # Allow `MAK2Optimizer(model)` instances that only set state for
    # display purposes (no .fit() call on the same line family).
    # Pattern to flag: `optimizer.fit(` or `.fit(` immediately following
    # a MAK2Optimizer instance used for primary fitting.
    forbidden = re.compile(
        r"^\s*\w+_(?:opt|optimizer)\.fit\s*\(", re.MULTILINE,
    )
    matches = forbidden.findall(src)

    # Known-legitimate exceptions: bootstrap re-fits and standard-curve
    # calibration use their own optimizer instances. They are NOT the
    # main per-well batch fit.
    allowed = {
        # bootstrap dialog re-fits one well many times with perturbed
        # data — fundamentally a different operation from a per-well
        # production fit. See `# BOOTSTRAP` markers in app.py.
        "_bo_opt.fit(",
        "_boot_opt.fit(",
    }
    actual_violations = [m for m in matches if m.lstrip() not in allowed]
    assert not actual_violations, (
        f"app.py instantiates MAK2Optimizer and calls .fit() in "
        f"{len(actual_violations)} place(s):\n  "
        + "\n  ".join(actual_violations)
        + "\n\nPer-well fitting must route through fit_well. If you "
        "have a real exception, add the call-site name to the allowlist "
        "in this test with a comment explaining why."
    )


# ─────────────────────────────────────────────────────────────────────────
# RULE 2 — No driver file may define its own copy of the four canonical
# entry points. Duplication-by-name catches the most common drift mode:
# someone copy-pastes a function back into app.py / run_batch.py / etc.
# ─────────────────────────────────────────────────────────────────────────
CANONICAL_FUNCTIONS = [
    ("fit_well",                  "fit_well.py"),
    ("prepare_fit_inputs",        "fit_well.py"),
    ("retry_one_well",            "pass2_helpers.py"),
    ("compute_channel_priors",    "pass2_helpers.py"),
    ("identify_retry_candidates", "pass2_helpers.py"),
    ("stage0_ct_from_inflection", "toe_prefit.py"),
    ("stage1_toe_fit",            "toe_prefit.py"),
    ("stage3_toe_gate",           "toe_prefit.py"),
    ("run_quality_gates",         "run_batch.py"),
    ("channel_of",                "pass2_helpers.py"),
    ("well_pos_of",               "pass2_helpers.py"),
]


def _toplevel_function_defs(path: Path) -> set[str]:
    """Return the set of top-level function names defined in a .py file."""
    try:
        tree = ast.parse(_read(path))
    except SyntaxError:
        return set()
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


@pytest.mark.parametrize("fn_name, canonical_file", CANONICAL_FUNCTIONS)
def test_canonical_function_defined_only_once(fn_name, canonical_file):
    """For each canonical function, exactly one file in the repo may
    define it as a top-level `def`. If another file defines the same
    name, it's a duplication."""
    py_files = [
        p for p in ROOT.glob("*.py")
        if p.name not in {"setup.py", "conftest.py"}
    ]
    definers = []
    for p in py_files:
        if fn_name in _toplevel_function_defs(p):
            definers.append(p.name)
    assert definers == [canonical_file], (
        f"`{fn_name}` is defined in {definers!r}; should be only in "
        f"`{canonical_file}`. If another file legitimately needs to "
        f"override it, import + wrap instead of redefining."
    )


# ─────────────────────────────────────────────────────────────────────────
# RULE 3 — The driver files must import the canonical functions they use.
# This is the positive form of rule 2: if a driver uses, say, fit_well
# semantics, it must do so via the import path, not via a local copy.
# ─────────────────────────────────────────────────────────────────────────
def test_app_py_imports_fit_well():
    src = _read(ROOT / "app.py")
    assert re.search(r"from\s+fit_well\s+import|import\s+fit_well", src), (
        "app.py must import from fit_well — the per-well fitting entry "
        "point lives there."
    )


def test_run_batch_py_imports_fit_well():
    src = _read(ROOT / "run_batch.py")
    assert re.search(r"from\s+fit_well\s+import|import\s+fit_well", src), (
        "run_batch.py must import from fit_well — run_pass1 routes "
        "through fit_well."
    )


def test_drivers_import_pass2_helpers():
    """Both batch driver paths must import from pass2_helpers (channel
    priors, retry candidates, retry_one_well)."""
    for fname in ("app.py", "run_batch.py"):
        src = _read(ROOT / fname)
        assert re.search(r"from\s+pass2_helpers\s+import", src), (
            f"{fname} must import from pass2_helpers — Pass 2 retry "
            "and channel-prior logic lives there."
        )
