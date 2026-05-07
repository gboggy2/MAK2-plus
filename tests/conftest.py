"""Shared pytest fixtures and path constants.

Importing the engine modules from the repo root requires the root to
be on ``sys.path`` — pytest doesn't add it automatically when tests
live in a subdirectory. Doing it here once means individual test
files can ``from mak2_model import ...`` without per-file boilerplate.

**Seed-before-import**: ``config.RANDOM_SEED`` is read from the
``MAK2_RANDOM_SEED`` env var at config-module import time. Any
attempt to seed via ``monkeypatch.setattr(config, ...)`` later
fixes only the config module's attribute — the engine modules that
already imported ``RANDOM_SEED`` via ``from config import RANDOM_SEED``
keep their stale (None) reference. So we set the env var here,
before pytest's collection imports any test file that imports the
engine, guaranteeing every engine module sees a seeded value at
its first import.
"""
from __future__ import annotations

import os

# MUST be set before any engine module is imported below.
os.environ.setdefault("MAK2_RANDOM_SEED", "42")

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def boggy_input_csv() -> Path:
    """Filesystem path to the input plate CSV (deterministic test fixture)."""
    return FIXTURES_DIR / "boggy_input.csv"


@pytest.fixture(scope="session")
def boggy_input_df(boggy_input_csv) -> pd.DataFrame:
    """Boggy plate as a DataFrame (Cycles + 12 sample columns).

    Loaded once per pytest session for speed — tests must not mutate
    the returned DataFrame.
    """
    return pd.read_csv(boggy_input_csv)


@pytest.fixture(scope="session")
def single_well_F1_1() -> dict:
    """The F1.1 well's cycles + fluorescence + nominal channel.

    Schema mirrors the ``POST /fit/single`` request body planned for
    Phase 1 — see CLAUDE.md.
    """
    path = FIXTURES_DIR / "single_well_F1_1.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def boggy_reference() -> dict:
    """Seeded MAK2 fit results for every Boggy well.

    Captured with ``MAK2_RANDOM_SEED=42`` by
    ``tests/fixtures/build_fixtures.py``. Used by the regression test
    for exact-equality assertions; do NOT regenerate on a whim, the
    diff between old and new is the permanent record of intentional
    behaviour changes.
    """
    path = FIXTURES_DIR / "boggy_reference.json"
    return json.loads(path.read_text())


@pytest.fixture(autouse=True)
def _confirm_seeded():
    """Sanity-check that the engine sees a seeded RANDOM_SEED.

    Detects the failure mode where someone runs ``pytest`` from an
    environment that has ``MAK2_RANDOM_SEED=`` (empty) or otherwise
    unset, in which case the conftest's module-level ``setdefault``
    won't override an explicitly-empty env value. Better to fail
    loudly here than to silently report bogus regression failures
    from unseeded fits.
    """
    import config
    assert config.RANDOM_SEED == 42, (
        f"Tests must run seeded; got config.RANDOM_SEED={config.RANDOM_SEED}. "
        "Check MAK2_RANDOM_SEED is unset (or 42) when invoking pytest."
    )
