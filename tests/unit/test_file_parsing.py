"""Unit tests for ``QPCRDataConverter`` on the simple-CSV format.

The Boggy fixture is in the simple format (column 0 = cycles, other
columns = wells). Only that format is tested here; the QuantStudio /
ABI / CFX formats are exercised via integration tests in Phase 1
when real instrument files are available.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from qpcr_data_converter import QPCRDataConverter


def test_load_from_file_returns_canonical_triple(boggy_input_csv):
    """Top-level ``load_from_file`` returns ``(cycles, samples, metadata)``.

    The triple is the engine's canonical input shape — every
    downstream function expects exactly this. Pin it here so an
    accidental shape change is caught immediately.
    """
    converter = QPCRDataConverter()
    cycles, samples, metadata = converter.load_from_file(str(boggy_input_csv))
    assert isinstance(cycles, np.ndarray)
    assert isinstance(samples, dict)
    assert isinstance(metadata, dict)


def test_loaded_cycle_count_matches_input(boggy_input_csv, boggy_input_df):
    """Loaded cycle array has the same length as the input file's row count."""
    converter = QPCRDataConverter()
    cycles, _, _ = converter.load_from_file(str(boggy_input_csv))
    assert len(cycles) == len(boggy_input_df)


def test_loaded_sample_count_matches_input(boggy_input_csv, boggy_input_df):
    """Sample dict has one entry per non-cycle column."""
    converter = QPCRDataConverter()
    _, samples, _ = converter.load_from_file(str(boggy_input_csv))
    expected_samples = [c for c in boggy_input_df.columns if c != "Cycles"]
    assert sorted(samples.keys()) == sorted(expected_samples)
    assert len(samples) == 12  # Boggy.csv has 12 wells


def test_loaded_fluorescence_arrays_have_correct_length(boggy_input_csv):
    """Each sample's fluorescence array matches the cycles array length."""
    converter = QPCRDataConverter()
    cycles, samples, _ = converter.load_from_file(str(boggy_input_csv))
    for name, fluor in samples.items():
        assert len(fluor) == len(cycles), (
            f"Sample {name} has {len(fluor)} points but cycles has {len(cycles)}"
        )


def test_no_nan_in_loaded_fluorescence(boggy_input_csv):
    """Loaded arrays have no NaN values for non-empty wells.

    Some parsers return NaN for missing data; downstream code
    (especially the optimizer) does not handle NaN gracefully.
    Pin this so a parser change that introduces NaN is caught.
    """
    converter = QPCRDataConverter()
    _, samples, _ = converter.load_from_file(str(boggy_input_csv))
    for name, fluor in samples.items():
        assert not np.isnan(fluor).any(), f"NaN values in {name}"


def test_metadata_format_field_set(boggy_input_csv):
    """Metadata dict carries a ``format`` key identifying the parsed format."""
    converter = QPCRDataConverter()
    _, _, metadata = converter.load_from_file(str(boggy_input_csv))
    assert metadata["format"] == "simple"


def test_detect_format_returns_simple_for_boggy(boggy_input_csv):
    """``detect_format`` recognises the simple format directly."""
    converter = QPCRDataConverter()
    df = pd.read_csv(boggy_input_csv)
    assert converter.detect_format(df) == "simple"


def test_offset_is_applied_when_enabled(boggy_input_csv, boggy_input_df):
    """``add_offset=True`` (default) raises every fluorescence point by ``offset_value``.

    Default offset is 1e-5 — invisible against any real signal but
    non-zero, preventing log(0) in downstream code.
    """
    converter = QPCRDataConverter(add_offset=True, offset_value=1e-5)
    _, samples_with_offset, _ = converter.load_from_file(str(boggy_input_csv))
    converter_no = QPCRDataConverter(add_offset=False)
    _, samples_no_offset, _ = converter_no.load_from_file(str(boggy_input_csv))
    # Pick any well; the offset should be uniformly added.
    well = next(iter(samples_with_offset))
    diff = samples_with_offset[well] - samples_no_offset[well]
    np.testing.assert_allclose(diff, 1e-5)
