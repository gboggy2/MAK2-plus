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

_seed_env = os.environ.get("MAK2_RANDOM_SEED")
RANDOM_SEED = int(_seed_env) if _seed_env is not None else None
