"""qPCR file-format parsers.

Reads the half-dozen file shapes that real qPCR instruments produce
and emits the same canonical pair: a ``cycles`` array and a
``samples`` dict from sample name to fluorescence array. The
downstream pipeline (``run_batch.py``, the Streamlit app) is then
format-agnostic.

Supported formats:

- **Simple CSV** — Column 1 = cycle numbers, other columns =
  fluorescence per sample. The most common output of academic /
  open-source tools (qpcR R package, MIQE-compliant exports).
- **Wide format** — Rows = cycles, columns = wells/samples. Common
  CSV export from generic plate readers.
- **CFX (Bio-Rad CFX Manager)** — `Cycle` column at column 0, then
  per-well columns.
- **QuantStudio (Applied Biosystems)** — Multi-sheet XLSX export
  with a separate Sample Setup sheet for the well-to-sample map.
- **ABI multicomponent CSV** — Sectioned CSV with ``[Sample Setup]``,
  ``[Multicomponent Data]``, ``[Amplification Data]``, ``[Results]``
  blocks. The richest format — preserves per-channel fluorescence,
  ROX passive reference, and instrument metadata.

Format detection is heuristic (column names, presence of section
headers); see ``detect_format``.

Two top-level entry points are exposed:
- ``QPCRDataConverter`` class — full API, allows re-use of the same
  configured converter across many files.
- ``load_abi_results_csv`` — module-level helper for the Results-CSV
  metadata extract used by ``run_batch.py``.

Author: Greg Boggy, PhD
Date: January 28, 2026
"""

import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class QPCRDataConverter:
    """Format-detect and parse qPCR data files into a canonical shape.

    Holds light parser configuration (offset behaviour) plus diagnostic
    state from the most recent ``load_from_file`` call (detected
    format, sample count, cycle count). The class can be re-used
    across multiple files.

    Canonical output of every parse method:

    - ``cycles`` (np.ndarray): cycle numbers, typically 1..N.
    - ``samples`` (Dict[str, np.ndarray]): well/sample name → per-cycle
      fluorescence array.
    - ``metadata`` (Dict): format-specific extras (channels list,
      passive reference, sample names, sample setup table, etc.).
      Keyed lookups are stable across formats; absent keys mean the
      format doesn't carry that information.
    """

    def __init__(self, add_offset: bool = True, offset_value: float = 1e-5):
        """Configure parsing behaviour.

        Args:
            add_offset: When True, add ``offset_value`` to every
                fluorescence point. Some downstream code (and the
                MAK2 model) takes log of fluorescence in places; the
                offset prevents log(0) when an instrument reports a
                literal 0.0 baseline. Default True.
            offset_value: Magnitude of the offset. Default 1e-5,
                small enough to be invisible against any real signal
                but non-zero everywhere.
        """
        self.add_offset = add_offset
        self.offset_value = offset_value
        self.detected_format = None
        self.n_samples = 0
        self.n_cycles = 0
        
    def detect_format(self, df: pd.DataFrame) -> str:
        """Heuristically classify a parsed DataFrame as one of the supported formats.

        Detection order matters — checks are tried in priority sequence
        and the first match wins:

        1. ``'simple'``: column 0 looks like cycle numbers
           (sequential integers starting near 1).
        2. ``'wide'``: every column name matches a well-position
           regex (``A1``-``H12``, ``Well_N``, ``Sample_N``).
        3. ``'cfx'``: presence of ``Well`` and ``Cq`` columns
           (Bio-Rad convention).
        4. ``'quantstudio'``: presence of ``Well Position`` or
           ``Target Name`` columns (Applied Biosystems convention).

        Falls back to ``'simple'`` if nothing matches — the simple
        parser is the most permissive and will give a useful error
        if it then fails on bad data.

        Args:
            df: Raw DataFrame from ``pd.read_csv`` / ``pd.read_excel``.

        Returns:
            One of ``'simple'``, ``'wide'``, ``'cfx'``, ``'quantstudio'``.
        """
        # Check for simple format (col 0 = cycles, others = samples)
        if df.shape[1] >= 2:
            first_col = df.iloc[:, 0]
            # Check if first column looks like cycle numbers
            if self._is_cycle_column(first_col):
                return 'simple'
        
        # Check for wide format (no explicit cycle column, rows are cycles)
        if all(df.columns.astype(str).str.match(r'^[A-H]\d{1,2}$|^Well_\d+$|^Sample_\d+$')):
            return 'wide'
        
        # Check for Bio-Rad CFX format
        if 'Well' in df.columns and 'Cq' in df.columns:
            return 'cfx'
        
        # Check for QuantStudio format  
        if 'Well Position' in df.columns or 'Target Name' in df.columns:
            return 'quantstudio'
        
        # Default to simple if we can't detect
        return 'simple'
    
    def _is_cycle_column(self, col: pd.Series) -> bool:
        """Heuristic test: does this column look like sequential cycle numbers?

        Three checks: <10% NaN after numeric coercion, first value
        in [0, 2] (some files index from 0, most from 1), and a mean
        cycle-to-cycle increment in [0.8, 1.2] (handles any small
        repeat-cycle artefacts).
        """
        try:
            # Try to convert to numeric
            numeric_col = pd.to_numeric(col, errors='coerce')
            
            # Should be mostly integers starting from 1
            if numeric_col.isna().sum() > len(col) * 0.1:  # More than 10% NaN
                return False
            
            # Should start near 1 and increment
            first_val = numeric_col.iloc[0]
            if not (0 <= first_val <= 2):
                return False
                
            # Check for roughly sequential values
            diffs = numeric_col.diff().dropna()
            if not (diffs.mean() > 0.8 and diffs.mean() < 1.2):
                return False
                
            return True
        except:
            return False
    
    def load_from_file(
        self,
        filepath: Union[str, Path, io.BytesIO]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
        """Top-level dispatch: load any supported qPCR file into canonical form.

        The decision tree (in order):

        1. If input is a ``BytesIO`` (e.g. Streamlit upload), try
           the ABI sectioned-CSV parser first (fastest reject for
           non-ABI data), then the QuantStudio multi-sheet XLSX
           parser, then a generic CSV/Excel read.
        2. If input is a ``.csv`` path, try ABI sectioned-CSV then
           generic CSV.
        3. Otherwise (Excel path), try QuantStudio multi-sheet,
           then generic Excel (openpyxl, fallback to xlrd).

        For multi-target QuantStudio files (``has_targets=True``),
        the function returns ``cycles=array, samples={}`` and sets
        ``metadata['requires_target_selection']=True`` so the UI can
        prompt the user to pick a target before re-loading.

        The configured ``offset_value`` is added to all fluorescence
        arrays unless ``add_offset=False``.

        Args:
            filepath: Path string, ``Path``, or in-memory ``BytesIO``.

        Returns:
            ``(cycles, samples, metadata)`` triple. ``metadata`` always
            contains ``format``, ``n_samples``, ``n_cycles``,
            ``sample_names``, ``cycle_range``; format-specific keys
            (``extra_info``, ``requires_target_selection``) appear
            when relevant.

        Raises:
            ValueError: If the file cannot be parsed by any path.
        """
        extra_info = None
        
        # Read file
        try:
            if isinstance(filepath, io.BytesIO):
                # Streamlit uploaded file - try both formats
                # First, check for ABI CSV format (# comments + [Section] markers)
                try:
                    result = self._try_abi_csv(filepath)
                    filepath.seek(0)
                    section_name, section_df, header_meta = result
                    return self._load_from_abi_csv(section_name, section_df, header_meta)
                except Exception:
                    filepath.seek(0)
                    pass

                # Next, check if it's an Excel file with multiple sheets (QuantStudio)
                try:
                    cycles, samples, extra_info = self._try_quantstudio_multisheet(filepath)
                    self.detected_format = 'quantstudio'
                    
                    # Check if we need target selection
                    if extra_info and extra_info.get('has_targets'):
                        # Return with extra_info for target selection
                        self.n_samples = 0  # Will be set after target selection
                        self.n_cycles = len(cycles)
                        
                        metadata = {
                            'format': 'quantstudio',
                            'n_samples': 0,
                            'n_cycles': self.n_cycles,
                            'sample_names': [],
                            'cycle_range': (cycles.min(), cycles.max()),
                            'extra_info': extra_info,
                            'requires_target_selection': True
                        }
                        
                        return cycles, samples, metadata
                    
                    # Single target - process normally
                    # Add offset if requested
                    if self.add_offset:
                        for name in samples:
                            samples[name] = samples[name] + self.offset_value
                    
                    # Store metadata
                    self.n_samples = len(samples)
                    self.n_cycles = len(cycles)
                    
                    metadata = {
                        'format': 'quantstudio',
                        'n_samples': self.n_samples,
                        'n_cycles': self.n_cycles,
                        'sample_names': list(samples.keys()),
                        'cycle_range': (cycles.min(), cycles.max()),
                        'requires_target_selection': False
                    }
                    
                    return cycles, samples, metadata
                    
                except:
                    # Not a multi-sheet QuantStudio, try normal reading
                    filepath.seek(0)
                    pass
                
                # Try reading as CSV or Excel
                try:
                    df = pd.read_csv(filepath)
                except:
                    filepath.seek(0)
                    try:
                        df = pd.read_excel(filepath, engine='openpyxl')
                    except:
                        filepath.seek(0)
                        df = pd.read_excel(filepath, engine='xlrd')
                        
            elif str(filepath).endswith('.csv'):
                # Try ABI CSV format first
                try:
                    section_name, section_df, header_meta = self._try_abi_csv(filepath)
                    return self._load_from_abi_csv(section_name, section_df, header_meta)
                except Exception:
                    pass
                df = pd.read_csv(filepath)
            else:
                # Excel file - try multi-sheet QuantStudio first
                try:
                    cycles, samples, extra_info = self._try_quantstudio_multisheet(filepath)
                    self.detected_format = 'quantstudio'
                    
                    # Check if we need target selection
                    if extra_info and extra_info.get('has_targets'):
                        # Return with extra_info for target selection
                        self.n_samples = 0  # Will be set after target selection
                        self.n_cycles = len(cycles)
                        
                        metadata = {
                            'format': 'quantstudio',
                            'n_samples': 0,
                            'n_cycles': self.n_cycles,
                            'sample_names': [],
                            'cycle_range': (cycles.min(), cycles.max()),
                            'extra_info': extra_info,
                            'requires_target_selection': True
                        }
                        
                        return cycles, samples, metadata
                    
                    # Single target - process normally
                    # Add offset if requested
                    if self.add_offset:
                        for name in samples:
                            samples[name] = samples[name] + self.offset_value
                    
                    # Store metadata
                    self.n_samples = len(samples)
                    self.n_cycles = len(cycles)
                    
                    metadata = {
                        'format': 'quantstudio',
                        'n_samples': self.n_samples,
                        'n_cycles': self.n_cycles,
                        'sample_names': list(samples.keys()),
                        'cycle_range': (cycles.min(), cycles.max()),
                        'requires_target_selection': False
                    }
                    
                    return cycles, samples, metadata
                    
                except:
                    # Not a multi-sheet format, read normally
                    pass
                
                # Try openpyxl first (for .xlsx), then xlrd (for .xls)
                try:
                    df = pd.read_excel(filepath, engine='openpyxl')
                except:
                    df = pd.read_excel(filepath, engine='xlrd')
                    
        except Exception as e:
            raise ValueError(f"Could not read file: {str(e)}")
        
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("File is empty or contains no data")
        
        # Check for minimum size
        if df.shape[0] < 10 or df.shape[1] < 2:
            raise ValueError(
                f"File too small: {df.shape[0]} rows, {df.shape[1]} columns. "
                f"Need at least 10 cycles and 2 columns (cycle + fluorescence)"
            )
        
        # Detect format
        format_type = self.detect_format(df)
        self.detected_format = format_type
        
        # Convert based on format
        try:
            if format_type == 'simple':
                cycles, samples = self._convert_simple_format(df)
            elif format_type == 'wide':
                cycles, samples = self._convert_wide_format(df)
            elif format_type == 'cfx':
                cycles, samples = self._convert_cfx_format(df)
            elif format_type == 'quantstudio':
                cycles, samples = self._convert_quantstudio_format(df)
            else:
                # Fallback to simple
                cycles, samples = self._convert_simple_format(df)
        except Exception as e:
            raise ValueError(
                f"Could not parse data in {format_type} format: {str(e)}\n"
                f"Expected format: First column = cycle numbers (1, 2, 3, ...), "
                f"other columns = fluorescence values"
            )
        
        # Validate extracted data
        if len(samples) == 0:
            raise ValueError("No samples found in file")
        
        if len(cycles) < 10:
            raise ValueError(f"Too few cycles: {len(cycles)}. Need at least 10 for qPCR analysis")
        
        # Check for valid fluorescence values
        for name, fluor in samples.items():
            if len(fluor) != len(cycles):
                raise ValueError(f"Sample '{name}' has {len(fluor)} values but {len(cycles)} cycles")
            if not np.isfinite(fluor).all():
                raise ValueError(f"Sample '{name}' contains invalid values (NaN or Inf)")
        
        # Add offset if requested
        if self.add_offset:
            for name in samples:
                samples[name] = samples[name] + self.offset_value
        
        # Store metadata
        self.n_samples = len(samples)
        self.n_cycles = len(cycles)
        
        metadata = {
            'format': format_type,
            'n_samples': self.n_samples,
            'n_cycles': self.n_cycles,
            'sample_names': list(samples.keys()),
            'cycle_range': (cycles.min(), cycles.max())
        }
        
        return cycles, samples, metadata
    
    def _convert_simple_format(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Parse a simple-format DataFrame: column 0 cycles, other columns samples.

        Example::

            Cycle | Sample_1 | Sample_2
            1     | 0.12     | 0.15
            2     | 0.18     | 0.22

        Skips columns with >10% NaN entries (likely metadata or
        comment columns mixed into the data); within accepted
        columns, missing values are linearly interpolated and
        backward/forward-filled at the edges to give the optimizer a
        contiguous array.

        Raises:
            ValueError: If fewer than 10 valid cycle rows are found,
                or if every fluorescence column was rejected.
        """
        # Try to convert first column to numeric, coercing errors
        cycles = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        
        # Find valid rows (where cycle is a valid number)
        valid_rows = ~cycles.isna()
        
        if valid_rows.sum() < 10:
            raise ValueError(
                f"Only {valid_rows.sum()} valid cycle rows found. "
                f"Need at least 10 rows with numeric cycle values"
            )
        
        # Extract valid cycles
        cycles = cycles[valid_rows].values
        
        # Extract samples - only use valid rows
        samples = {}
        for col in df.columns[1:]:
            # Convert to numeric, coercing errors
            fluor = pd.to_numeric(df[col][valid_rows], errors='coerce')
            
            # Skip if too many NaN values
            if fluor.isna().sum() / len(fluor) > 0.1:  # More than 10% NaN
                continue
            
            # Fill any remaining NaN with interpolation
            if fluor.isna().any():
                fluor = fluor.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            
            samples[str(col)] = fluor.values
        
        if len(samples) == 0:
            raise ValueError("No valid sample columns found with numeric fluorescence data")
        
        return cycles, samples
    
    def _convert_wide_format(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Parse a wide-format DataFrame: every column a sample, rows are cycles.

        Cycle numbers are not stored in the file; they're synthesised
        as ``1, 2, ..., len(df)``. Example::

            A1   | A2   | A3
            0.12 | 0.15 | 0.11
            0.18 | 0.22 | 0.17
        """
        # Generate cycle numbers
        cycles = np.arange(1, len(df) + 1)
        
        # Extract samples
        samples = {}
        for col in df.columns:
            samples[str(col)] = df[col].values
        
        return cycles, samples
    
    def _convert_cfx_format(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Parse Bio-Rad CFX Manager export.

        Long-format DataFrame with one row per (well, cycle) pair;
        groupby-pivots into the canonical (cycles, samples) shape.
        Fluorescence column name is autodetected by case-insensitive
        substring match against ``fluor`` or ``rfu``; falls back to
        the third column.
        """
        # Find fluorescence column (might be named differently)
        fluor_cols = [c for c in df.columns if 'fluor' in c.lower() or 'rfu' in c.lower()]
        if not fluor_cols:
            # Fallback: use third column
            fluor_col = df.columns[2] if len(df.columns) > 2 else df.columns[1]
        else:
            fluor_col = fluor_cols[0]
        
        # Group by well
        samples = {}
        for well, group in df.groupby('Well'):
            samples[str(well)] = group[fluor_col].values
        
        # Get cycles from first well
        first_well = df['Well'].iloc[0]
        cycles = df[df['Well'] == first_well]['Cycle'].values
        
        return cycles, samples
    
    def _convert_quantstudio_format(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Parse Applied Biosystems QuantStudio export (single-sheet variant).

        Long-format DataFrame keyed by ``Well Position`` (or ``Well``)
        and ``Cycle`` (or ``Cycle Number``). Fluorescence column
        autodetected by ``rn`` / ``fluor`` substring; falls back to
        the last column. For multi-sheet QuantStudio Excel files,
        ``_try_quantstudio_multisheet`` does the heavier work first.
        """
        # Find fluorescence column
        well_col = 'Well Position' if 'Well Position' in df.columns else 'Well'
        cycle_col = 'Cycle' if 'Cycle' in df.columns else 'Cycle Number'
        
        fluor_cols = [c for c in df.columns if 'rn' in c.lower() or 'fluor' in c.lower()]
        if fluor_cols:
            fluor_col = fluor_cols[0]
        else:
            fluor_col = df.columns[-1]  # Last column as fallback
        
        # Group by well
        samples = {}
        for well, group in df.groupby(well_col):
            samples[str(well)] = group[fluor_col].values
        
        # Get cycles from first well
        first_well = df[well_col].iloc[0]
        cycles = df[df[well_col] == first_well][cycle_col].values
        
        return cycles, samples
    
    def _parse_sample_setup(self, xls: pd.ExcelFile) -> Optional[pd.DataFrame]:
        """Parse the Sample Setup sheet from a QuantStudio multi-sheet export.

        Recovers per-well metadata: ``Sample Name``, ``Task`` (UNKNOWN /
        STANDARD / NTC), ``Quantity`` (the known copy number for
        standards). This metadata feeds the calibration step
        downstream (see ``calibration.build_standard_curve``).

        The header row position varies between QuantStudio versions
        (typically around row 46), so we scan the first 60 rows for
        the canonical column names rather than assuming a fixed
        offset. Tolerates the trailing-space sheet name variant
        ("Sample Setup ").

        Args:
            xls: An open ``pd.ExcelFile`` handle.

        Returns:
            DataFrame with the relevant subset of columns, or
            ``None`` if the sheet is absent or doesn't parse.
        """
        setup_sheets = ['Sample Setup', 'Sample Setup ']  # trailing space variant
        for sheet in setup_sheets:
            if sheet in xls.sheet_names:
                try:
                    df_raw = pd.read_excel(xls, sheet_name=sheet, header=None)
                    # Find header row (contains "Well" and "Sample Name")
                    for i in range(min(60, len(df_raw))):
                        row_vals = [str(v).strip() for v in df_raw.iloc[i].values]
                        if 'Well' in row_vals and 'Sample Name' in row_vals:
                            # Use this row as column names
                            col_names = [str(v).strip() for v in df_raw.iloc[i].values]
                            # Read data rows below the header
                            df = df_raw.iloc[i+1:].copy()
                            df.columns = col_names
                            df = df.reset_index(drop=True)

                            # Keep relevant columns
                            keep = ['Well', 'Well Position', 'Sample Name', 'Target Name',
                                    'Task', 'Quantity']
                            keep = [c for c in keep if c in df.columns]
                            df = df[keep]

                            # Drop rows where Well is empty/NaN
                            df = df.dropna(subset=['Well'])

                            # Convert Quantity to numeric
                            if 'Quantity' in df.columns:
                                df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

                            return df
                except Exception:
                    pass
        return None

    def _parse_results_sheet(self, xls: pd.ExcelFile) -> Optional[pd.DataFrame]:
        """Parse the Results sheet from a QuantStudio multi-sheet export.

        Recovers the instrument's own per-well analysis: ``CT`` and
        ``Ct Threshold``. The Streamlit app uses these as the
        comparison baseline against the MAK2-fit Ct (so the user can
        sanity-check that the fit isn't wildly disagreeing with the
        instrument).

        Same dynamic header-row scan as ``_parse_sample_setup`` —
        the row number isn't fixed across QuantStudio versions.

        Args:
            xls: An open ``pd.ExcelFile`` handle.

        Returns:
            DataFrame with ``Well Position``, ``Target Name``, ``CT``,
            ``Ct Threshold`` columns, or ``None`` if absent or
            unparseable.
        """
        results_sheets = ['Results', 'Results ']
        for sheet in results_sheets:
            if sheet in xls.sheet_names:
                try:
                    df_raw = pd.read_excel(xls, sheet_name=sheet, header=None)
                    for i in range(min(60, len(df_raw))):
                        row_vals = [str(v).strip() for v in df_raw.iloc[i].values]
                        if 'Well Position' in row_vals and 'CT' in row_vals:
                            col_names = [str(v).strip() for v in df_raw.iloc[i].values]
                            df = df_raw.iloc[i+1:].copy()
                            df.columns = col_names
                            df = df.reset_index(drop=True)

                            keep = ['Well Position', 'Target Name', 'CT', 'Ct Threshold']
                            keep = [c for c in keep if c in df.columns]
                            df = df[keep]
                            df = df.dropna(subset=['Well Position'])

                            if 'CT' in df.columns:
                                df['CT'] = pd.to_numeric(df['CT'], errors='coerce')
                            if 'Ct Threshold' in df.columns:
                                df['Ct Threshold'] = pd.to_numeric(
                                    df['Ct Threshold'], errors='coerce'
                                )

                            return df
                except Exception:
                    pass
        return None

    def _try_quantstudio_multisheet(self, filepath: Union[str, Path, io.BytesIO]) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[Dict]]:
        """Parse a QuantStudio multi-sheet XLSX export.

        Three sheets get read together because they're complementary:

        - **Amplification Data**: per-(well, cycle) fluorescence
          (``Rn`` and/or ``Delta_Rn``). The numeric source of truth.
        - **Sample Setup**: per-well metadata (sample name, task,
          known quantity). Needed for downstream calibration.
        - **Results**: instrument-computed CT and threshold. Used
          as a sanity-check baseline.

        Multiplexed plates carry a ``Target`` column. When present,
        this method does NOT pivot the data — instead it stashes the
        raw long-form DataFrame in ``extra_info['raw_df']`` and sets
        ``has_targets=True`` so the UI can offer a target selector
        before the actual per-target pivot is done by
        ``filter_by_target``.

        Args:
            filepath: Excel file path or BytesIO.

        Returns:
            ``(cycles, samples, extra_info)``. For single-target
            plates, ``samples`` is the canonical per-well dict; for
            multiplexed plates, ``samples`` is empty and
            ``extra_info`` carries the raw data plus the target list.

        Raises:
            ValueError: If no recognisable amplification sheet is
                present.
        """
        try:
            # Check if it's an Excel file with multiple sheets
            if isinstance(filepath, io.BytesIO):
                filepath.seek(0)
                xls = pd.ExcelFile(filepath)
            else:
                xls = pd.ExcelFile(filepath)
            
            # Look for Amplification Data sheet
            target_sheets = ['Amplification Data', 'Amplification', 'Amp Data', 'RawData']
            sheet_name = None
            
            for sheet in target_sheets:
                if sheet in xls.sheet_names:
                    sheet_name = sheet
                    break
            
            if not sheet_name:
                raise ValueError("No amplification data sheet found")
            
            # Read the sheet
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Find where the data starts - look for "Cycle" or "Well" header
            data_start_row = None
            for i in range(min(100, len(df))):
                row_val = str(df.iloc[i, 0]).lower()
                if 'well' in row_val or 'cycle' in row_val:
                    data_start_row = i
                    break
            
            if data_start_row is None:
                raise ValueError("Could not find data start in amplification sheet")
            
            # Re-read with correct header
            if isinstance(filepath, io.BytesIO):
                filepath.seek(0)
            df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=data_start_row+1, header=None)
            
            # Set column names based on typical QuantStudio structure
            if df.shape[1] >= 6:
                df.columns = ['Well', 'Well_Position', 'Cycle', 'Target', 'Rn', 'Delta_Rn']
            elif df.shape[1] >= 3:
                df.columns = ['Well_Position', 'Cycle', 'Delta_Rn'] + [f'Col_{i}' for i in range(df.shape[1]-3)]
            
            # Clean data
            df = df.dropna(subset=['Cycle'])
            df['Cycle'] = pd.to_numeric(df['Cycle'], errors='coerce')
            
            # Use Delta_Rn or Rn for fluorescence
            fluor_col = 'Delta_Rn' if 'Delta_Rn' in df.columns else 'Rn'
            df[fluor_col] = pd.to_numeric(df[fluor_col], errors='coerce')
            df = df.dropna(subset=['Cycle', fluor_col])
            
            # Find well position column
            well_col = 'Well_Position' if 'Well_Position' in df.columns else 'Well'
            
            # Parse Sample Setup sheet for metadata (Sample Name, Task, Quantity)
            sample_setup = self._parse_sample_setup(xls)

            # Parse Results sheet for instrument-reported CT values
            results_sheet = self._parse_results_sheet(xls)

            # Check if we have Target column (multiplexed data)
            has_targets = 'Target' in df.columns and df['Target'].notna().any()

            if has_targets:
                # Get unique targets
                targets = df['Target'].dropna().unique().tolist()

                # Return data organized by target
                extra_info = {
                    'has_targets': True,
                    'targets': targets,
                    'raw_df': df,  # Store for later filtering
                    'well_col': well_col,
                    'fluor_col': fluor_col,
                    'sample_setup': sample_setup,  # Well metadata (may be None)
                    'results_sheet': results_sheet,  # Instrument CT values (may be None)
                }
                
                # For now, return empty samples dict - will be populated after target selection
                cycles = np.arange(1, int(df['Cycle'].max()) + 1)
                samples = {}
                
                return cycles, samples, extra_info
            else:
                # Single target - process normally
                pivot = df.pivot_table(index='Cycle', columns=well_col, values=fluor_col, aggfunc='first')
                
                cycles = pivot.index.values
                samples = {str(col): pivot[col].values for col in pivot.columns}
                
                return cycles, samples, None
            
        except Exception as e:
            raise ValueError(f"QuantStudio multi-sheet parsing failed: {str(e)}")
    
    def _load_from_abi_csv(
        self,
        section_name: str,
        section_df: pd.DataFrame,
        header_meta: Dict[str, str]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
        """Wrap a parsed ABI section into the canonical ``load_from_file`` return.

        Two paths depending on which section was found:

        - **Multicomponent**: rich format with per-channel
          fluorescence + a passive reference dye. Returns
          ``samples={}`` and stashes the per-channel data in
          ``metadata['extra_info']`` so the UI can offer a channel
          selector before the actual fit. Sets
          ``requires_channel_selection=True``.
        - **Amplification Data** (or other single-channel section):
          single-channel ΔRn data. Returns the canonical samples
          dict directly, ready to fit. Sets
          ``requires_channel_selection=False``.
        """
        if section_name == 'Multicomponent':
            cycles, samples_by_channel, dye_columns, passive_reference = \
                self._parse_abi_multicomponent(section_df, header_meta)

            # Channels available for fitting = all channels except passive reference
            fitting_channels = [c for c in dye_columns if c != passive_reference]

            self.detected_format = 'abi_multicomponent'
            self.n_cycles = len(cycles)
            self.n_samples = 0  # populated after channel selection

            extra_info = {
                'has_channels': True,
                'channels': fitting_channels,
                'passive_reference': passive_reference,
                'all_channels': dye_columns,
                'samples_by_channel': samples_by_channel,
                'cycles': cycles,
                'header_meta': header_meta,
                'section': section_name,
            }

            metadata = {
                'format': 'abi_multicomponent',
                'n_samples': 0,
                'n_cycles': self.n_cycles,
                'sample_names': [],
                'cycle_range': (cycles.min(), cycles.max()),
                'extra_info': extra_info,
                'requires_channel_selection': True,
                'requires_target_selection': False,
            }

            return cycles, {}, metadata

        else:
            # Amplification Data or other single-channel section
            cycles, samples = self._parse_abi_amplification(section_df, header_meta)

            if self.add_offset:
                for name in samples:
                    samples[name] = samples[name] + self.offset_value

            self.detected_format = 'abi_amplification'
            self.n_samples = len(samples)
            self.n_cycles = len(cycles)

            metadata = {
                'format': 'abi_amplification',
                'n_samples': self.n_samples,
                'n_cycles': self.n_cycles,
                'sample_names': list(samples.keys()),
                'cycle_range': (cycles.min(), cycles.max()),
                'requires_channel_selection': False,
                'requires_target_selection': False,
            }

            return cycles, samples, metadata

    def _try_abi_csv(
        self,
        filepath: Union[str, Path, io.BytesIO]
    ) -> Tuple[str, pd.DataFrame, Dict[str, str]]:
        """Detect and split an ABI sectioned CSV into header + chosen section.

        ABI CSVs look like::

            # Instrument Type: 7500 Fast System
            # Passive Reference: ROX
            # ...
            [Sample Setup]
            <CSV rows>
            [Multicomponent]
            <CSV rows>
            [Amplification Data]
            <CSV rows>
            [Results]
            <CSV rows>

        This method:
          1. Parses the ``#`` comment block into a metadata dict.
          2. Splits the body into sections by ``[Name]`` markers.
          3. Picks the highest-priority section for fitting:
             ``Multicomponent`` > ``Amplification Data`` > ``Raw Data``
             (Multicomponent preferred because it preserves all
             channels including the ROX passive reference, which the
             pipeline uses for normalization).
          4. Parses the chosen section as a CSV.

        Args:
            filepath: Path or BytesIO.

        Returns:
            ``(section_name, section_df, header_meta)``.

        Raises:
            ValueError: If the file has no ``#`` headers (not ABI) or
                no ``[Section]`` markers.
        """
        # Read raw lines
        if isinstance(filepath, io.BytesIO):
            filepath.seek(0)
            raw = filepath.read().decode('utf-8', errors='replace')
            filepath.seek(0)
        else:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                raw = f.read()

        lines = raw.splitlines()

        # Must have at least some # comment lines to be an ABI CSV
        comment_lines = [l for l in lines if l.startswith('#')]
        if not comment_lines:
            raise ValueError("Not an ABI CSV: no # comment header lines found")

        # Parse header metadata
        header_meta: Dict[str, str] = {}
        for line in comment_lines:
            # Format: "# Key: Value"
            content = line[1:].strip()
            if ':' in content:
                key, _, value = content.partition(':')
                header_meta[key.strip()] = value.strip()

        # Split non-comment lines into sections by [Section Name] markers
        # A section marker is a line that, when stripped of quotes and whitespace,
        # matches \[.*\]
        sections: Dict[str, List[str]] = {}
        current_section: Optional[str] = None
        current_lines: List[str] = []

        for line in lines:
            if line.startswith('#') or line.strip() == '':
                continue
            stripped = line.strip().strip('"')
            m = re.match(r'^\[(.+)\]$', stripped)
            if m:
                # Save previous section
                if current_section is not None and current_lines:
                    sections[current_section] = current_lines
                current_section = m.group(1)
                current_lines = []
            else:
                if current_section is not None:
                    current_lines.append(line)

        # Save last section
        if current_section is not None and current_lines:
            sections[current_section] = current_lines

        if not sections:
            raise ValueError("Not an ABI CSV: no [Section] markers found")

        # Priority order for sections to use for fitting
        priority = ['Multicomponent', 'Amplification Data', 'Raw Data']
        chosen_section = None
        for name in priority:
            if name in sections:
                chosen_section = name
                break

        if chosen_section is None:
            # Fall back to whatever we found
            chosen_section = next(iter(sections))

        # Parse the chosen section as CSV
        section_text = '\n'.join(sections[chosen_section])
        section_df = pd.read_csv(io.StringIO(section_text))

        return chosen_section, section_df, header_meta

    def _parse_abi_multicomponent(
        self,
        section_df: pd.DataFrame,
        header_meta: Dict[str, str]
    ) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]], List[str], Optional[str]]:
        """Pivot the ``[Multicomponent]`` section into per-channel per-well arrays.

        Layout of the input section: long-format with ``Well Position``,
        ``Cycle Number``, and one column per dye channel (FAM, JOE,
        ROX, …). This function pivots it into nested dicts keyed by
        channel and then by well position.

        Notable handling:

        - **Cycle deduplication.** Some ABI instruments append
          post-amplification reads (melt curve, dissociation) at the
          same cycle numbers. We keep only the first occurrence of
          each cycle so that downstream code sees the amplification
          phase only.
        - **Passive reference identification.** Read from the
          ``Passive Reference`` header field (typically ``ROX``);
          downstream code uses it to ROX-normalise per-channel data.

        Args:
            section_df: Parsed DataFrame from
                ``_try_abi_csv``.
            header_meta: Header metadata dict.

        Returns:
            ``(cycles, samples_by_channel, dye_columns, passive_reference)``.
            ``samples_by_channel`` is keyed by channel name then by
            well position. ``passive_reference`` may be ``None`` if
            the header doesn't declare one.

        Raises:
            ValueError: If required columns (cycle, well, any dye)
                are missing.
        """
        # Identify structural columns
        non_dye = {'Well', 'Well Position', 'Stage Number', 'Step Number', 'Cycle Number',
                   'well', 'well position', 'stage number', 'step number', 'cycle number'}

        # Find the cycle column
        cycle_col = None
        for col in section_df.columns:
            if col.strip().lower() in ('cycle number', 'cycle'):
                cycle_col = col
                break
        if cycle_col is None:
            raise ValueError("No 'Cycle Number' column found in [Multicomponent] section")

        # Find well position column
        well_col = None
        for col in section_df.columns:
            if col.strip().lower() in ('well position',):
                well_col = col
                break
        if well_col is None:
            for col in section_df.columns:
                if col.strip().lower() == 'well':
                    well_col = col
                    break
        if well_col is None:
            raise ValueError("No 'Well Position' or 'Well' column found in [Multicomponent] section")

        # Dye columns = everything else
        dye_columns = [
            col for col in section_df.columns
            if col.strip().lower() not in non_dye
        ]
        if not dye_columns:
            raise ValueError("No dye channel columns found in [Multicomponent] section")

        # Convert to numeric
        section_df = section_df.copy()
        section_df[cycle_col] = pd.to_numeric(section_df[cycle_col], errors='coerce')
        for col in dye_columns:
            section_df[col] = pd.to_numeric(section_df[col], errors='coerce')
        section_df = section_df.dropna(subset=[cycle_col, well_col])

        # Get cycle array from first well
        first_well = section_df[well_col].iloc[0]
        raw_cycles = section_df[section_df[well_col] == first_well][cycle_col].values.astype(float)

        # Some instruments append post-amplification reads (e.g. melt curve)
        # at the same cycle number.  Keep only the first occurrence of each
        # cycle to get the amplification-phase data.
        _, unique_idx = np.unique(raw_cycles, return_index=True)
        unique_idx.sort()  # preserve original order
        cycles = raw_cycles[unique_idx]
        n_keep = len(unique_idx)

        # Build samples_by_channel, truncating to amplification cycles only
        samples_by_channel: Dict[str, Dict[str, np.ndarray]] = {}
        for dye in dye_columns:
            well_dict: Dict[str, np.ndarray] = {}
            for well_pos, group in section_df.groupby(well_col):
                group_sorted = group.sort_values(cycle_col)
                arr = group_sorted[dye].values.astype(float)
                well_dict[str(well_pos)] = arr[unique_idx] if len(arr) > n_keep else arr
            samples_by_channel[dye] = well_dict

        # Passive reference from header
        passive_reference = header_meta.get('Passive Reference', None)
        if passive_reference:
            passive_reference = passive_reference.strip()

        return cycles, samples_by_channel, dye_columns, passive_reference

    def _parse_abi_amplification(
        self,
        section_df: pd.DataFrame,
        header_meta: Dict[str, str]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Pivot the ``[Amplification Data]`` section into per-well arrays.

        Single-channel cousin of ``_parse_abi_multicomponent``.
        Long-format DataFrame with ``Well Position``, ``Cycle Number``,
        and a ``ΔRn`` (or similar) value column. Used when the ABI
        export doesn't contain a Multicomponent section (older
        instruments or single-channel workflows).
        """
        # Find cycle column
        cycle_col = None
        for col in section_df.columns:
            if col.strip().lower() in ('cycle number', 'cycle'):
                cycle_col = col
                break
        if cycle_col is None:
            raise ValueError("No cycle column found in [Amplification Data] section")

        # Find well position column
        well_col = None
        for col in section_df.columns:
            if col.strip().lower() in ('well position',):
                well_col = col
                break
        if well_col is None:
            for col in section_df.columns:
                if col.strip().lower() == 'well':
                    well_col = col
                    break

        # Find fluorescence column (ΔRn preferred over Rn)
        fluor_col = None
        for candidate in section_df.columns:
            c = candidate.strip().lower()
            if 'delta' in c and 'rn' in c:
                fluor_col = candidate
                break
        if fluor_col is None:
            for candidate in section_df.columns:
                c = candidate.strip().lower()
                if 'rn' in c or 'fluor' in c:
                    fluor_col = candidate
                    break
        if fluor_col is None:
            fluor_col = section_df.columns[-1]

        section_df = section_df.copy()
        section_df[cycle_col] = pd.to_numeric(section_df[cycle_col], errors='coerce')
        section_df[fluor_col] = pd.to_numeric(section_df[fluor_col], errors='coerce')
        section_df = section_df.dropna(subset=[cycle_col])

        first_well = section_df[well_col].iloc[0]
        cycles = section_df[section_df[well_col] == first_well][cycle_col].values.astype(float)

        samples: Dict[str, np.ndarray] = {}
        for well_pos, group in section_df.groupby(well_col):
            group_sorted = group.sort_values(cycle_col)
            samples[str(well_pos)] = group_sorted[fluor_col].values.astype(float)

        return cycles, samples

    def filter_by_channel(
        self,
        extra_info: Dict,
        channel_name: str,
        normalize_by_reference: bool = False
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], None]:
        """Extract one channel's per-well data from a Multicomponent load.

        Optionally divides by the passive-reference channel (ROX) to
        correct for well-to-well loading volume variation. ROX
        normalisation is the conventional first-line correction on
        ABI instruments — without it, well-volume differences
        propagate into the D0 estimates as systematic per-well bias.

        Args:
            extra_info: ``extra_info`` dict from ``load_from_file``
                when ``requires_channel_selection`` was True.
            channel_name: Dye to extract (e.g. ``'FAM'``, ``'JOE'``).
            normalize_by_reference: When True, divide the chosen
                channel by the passive reference (default False
                because the per-well pipeline does its own
                channel-aware normalisation downstream).

        Returns:
            ``(cycles, samples, None)`` — third element is unused
            but preserved for API compatibility with other parsers.

        Raises:
            ValueError: If ``extra_info`` lacks channel data or if
                ``channel_name`` isn't present.
        """
        if not extra_info or not extra_info.get('has_channels'):
            raise ValueError("No channel information available in extra_info")

        samples_by_channel = extra_info['samples_by_channel']
        if channel_name not in samples_by_channel:
            raise ValueError(f"Channel '{channel_name}' not found. Available: {list(samples_by_channel.keys())}")

        cycles = extra_info['cycles']
        well_dict = samples_by_channel[channel_name]

        samples: Dict[str, np.ndarray] = {}
        for well_pos, arr in well_dict.items():
            fluor = arr.copy()
            if normalize_by_reference:
                ref = extra_info.get('passive_reference')
                if ref and ref in samples_by_channel:
                    ref_arr = samples_by_channel[ref].get(well_pos)
                    if ref_arr is not None and np.all(ref_arr > 0):
                        fluor = fluor / ref_arr
            if self.add_offset:
                fluor = fluor + self.offset_value
            samples[well_pos] = fluor

        return cycles, samples, None

    def get_sample_info(self, samples: Dict[str, np.ndarray]) -> pd.DataFrame:
        """One-row-per-sample summary table for sanity checks.

        Quick visual triage before fitting: the ``Detectable`` column
        flags wells where ``max < 2 * min`` (i.e. less than a 2× signal
        range — almost certainly a flat NTC or failed well). Doesn't
        do the heavy lifting that ``data_processing.detect_no_signal_samples``
        does; this is just a sanity print.

        Args:
            samples: Sample dict from any ``load_from_file`` path.

        Returns:
            DataFrame with ``Sample``, ``Min``, ``Max``, ``Mean``,
            ``Detectable`` columns.
        """
        info = []
        for name, fluor in samples.items():
            info.append({
                'Sample': name,
                'Min': fluor.min(),
                'Max': fluor.max(),
                'Mean': fluor.mean(),
                'Detectable': 'Yes' if fluor.max() > fluor.min() * 2 else 'Low Signal'
            })
        
        return pd.DataFrame(info)
    
    def filter_by_target(
        self,
        extra_info: Dict,
        target_name: str
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[Dict]]:
        """Pivot one target's data out of a multiplexed QuantStudio plate.

        For multiplexed plates, ``_try_quantstudio_multisheet`` defers
        the pivot until the user has chosen a target. This method
        does that pivot and bundles the per-well sample metadata
        (Sample Name, Task, Quantity from Sample Setup; instrument
        CT from Results) so downstream code can run a complete
        analysis on this single target.

        Args:
            extra_info: ``extra_info`` dict from the deferred
                multi-sheet load (must have ``has_targets=True``).
            target_name: Target to pivot.

        Returns:
            ``(cycles, samples, sample_metadata)``. ``sample_metadata``
            is ``{well_pos: {field: value, ...}}`` with NaN values
            normalised to ``None`` for cleaner downstream handling;
            it's ``None`` when neither Sample Setup nor Results
            sheets were parseable.

        Raises:
            ValueError: If ``extra_info`` lacks target data or no
                rows match ``target_name``.
        """
        if not extra_info or not extra_info.get('has_targets'):
            raise ValueError("No target information available")

        df = extra_info['raw_df']
        well_col = extra_info['well_col']
        fluor_col = extra_info['fluor_col']

        # Filter for selected target
        df_target = df[df['Target'] == target_name].copy()

        if len(df_target) == 0:
            raise ValueError(f"No data found for target '{target_name}'")

        # Pivot to get samples
        pivot = df_target.pivot_table(index='Cycle', columns=well_col, values=fluor_col, aggfunc='first')

        cycles = pivot.index.values
        samples = {str(col): pivot[col].values for col in pivot.columns}

        # Add offset if requested
        if self.add_offset:
            for name in samples:
                samples[name] = samples[name] + self.offset_value

        # Build per-well metadata from Sample Setup (if available)
        sample_metadata = None
        setup_df = extra_info.get('sample_setup')
        if setup_df is not None:
            # Filter setup by target name if Target Name column exists
            if 'Target Name' in setup_df.columns:
                target_setup = setup_df[setup_df['Target Name'] == target_name].copy()
            else:
                target_setup = setup_df.copy()

            if len(target_setup) > 0:
                # Key by Well Position to match sample dict keys
                wp_col = 'Well Position' if 'Well Position' in target_setup.columns else 'Well'
                # Build dict: {well_pos: {col: value, ...}}
                sample_metadata = {}
                for _, row in target_setup.iterrows():
                    well_pos = str(row[wp_col])
                    meta = {}
                    for col in target_setup.columns:
                        if col != wp_col:
                            val = row[col]
                            # Convert NaN to None for cleaner handling
                            if pd.isna(val):
                                meta[col] = None
                            else:
                                meta[col] = val
                    sample_metadata[well_pos] = meta

        # Merge instrument-reported CT values from Results sheet (if available)
        results_df = extra_info.get('results_sheet')
        if results_df is not None and sample_metadata is not None:
            if 'Target Name' in results_df.columns:
                target_results = results_df[results_df['Target Name'] == target_name]
            else:
                target_results = results_df

            for _, row in target_results.iterrows():
                well_pos = str(row.get('Well Position', ''))
                if well_pos in sample_metadata:
                    if 'CT' in results_df.columns and pd.notna(row.get('CT')):
                        sample_metadata[well_pos]['Instrument_CT'] = float(row['CT'])
                    if 'Ct Threshold' in results_df.columns and pd.notna(row.get('Ct Threshold')):
                        sample_metadata[well_pos]['Ct_Threshold'] = float(row['Ct Threshold'])

        return cycles, samples, sample_metadata

    def filter_all_targets(
        self,
        extra_info: Dict,
    ) -> Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray], Optional[Dict]]]:
        """Pivot every target's data out of a multiplexed plate at once.

        Args:
            extra_info: ``extra_info`` dict from
                ``_try_quantstudio_multisheet``.

        Returns:
            Dict keyed by target name; each value is the
            ``(cycles, samples, sample_metadata)`` triple that
            ``filter_by_target`` would return for that target.
            Convenience wrapper for callers that want every target
            at once (e.g. ``app.py`` channel iteration).
        """
        if not extra_info or not extra_info.get('has_targets'):
            raise ValueError("No target information available")

        targets = extra_info['targets']
        result = {}
        for target in targets:
            cycles, samples, metadata = self.filter_by_target(extra_info, target)
            result[target] = (cycles, samples, metadata)
        return result

def load_abi_results_csv(file) -> Dict:
    """Parse the per-well metadata CSV exported alongside an ABI plate.

    The Results CSV (separate from the Multicomponent CSV that holds
    fluorescence) carries per-well metadata that drives several
    pipeline behaviours:

    - ``Sample Name`` / ``Target Name`` / ``Task`` / ``Quantity``:
      identify standards for ``calibration.build_standard_curve``.
    - ``Baseline Start`` / ``Baseline End``: define the per-well
      baseline window used by ``pre_estimate_background``.
    - ``Ct Threshold``: the per-channel Ct threshold the instrument
      computed; used as the threshold value for the MAK2-fitted Ct
      so the MAK2 Ct is comparable to the instrument's number.
    - ``HIGHSD`` / ``NOAMP`` / ``EXPFAIL``: the instrument's quality
      flags, surfaced in the Status column for cross-reference.
    - The Ct column is labelled ``C_`` in ABI exports (with
      fallbacks for ``CT``/``Ct``/``Cq``/``Cp``).

    Args:
        file: Either a path/string or a file-like object with a
            ``read`` method (e.g. a Streamlit upload).

    Returns:
        Dict with:

        - ``'well_meta'`` / ``'sample_metadata'`` (same dict —
          aliased for legacy callers): keyed by ``f"{Reporter}_{Well}"``,
          each value is a per-well dict including the metadata
          fields above.
        - ``'channel_thresholds'``: per-channel median Ct threshold,
          e.g. ``{'FAM': 0.1, 'JOE': 0.04}``.
        - ``'n_wells'``: total wells parsed.

    Raises:
        ValueError: If the CSV lacks ``Well`` or ``Reporter`` columns.
    """
    if hasattr(file, 'read'):
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')
        df = pd.read_csv(io.StringIO(content))
    else:
        df = pd.read_csv(file)

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # The Ct column is labelled "C_" in ABI exports
    ct_col = 'C_'
    if ct_col not in df.columns:
        # Fallback: accept 'CT', 'Ct', 'Cq' etc.
        for candidate in ['CT', 'Ct', 'Cq', 'Cp']:
            if candidate in df.columns:
                ct_col = candidate
                break

    required = {'Well', 'Reporter'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ABI results CSV missing required columns: {missing}")

    well_meta: Dict[str, Dict] = {}
    channel_thresholds: Dict[str, float] = {}

    for _, row in df.iterrows():
        well = str(row.get('Well', '')).strip()
        reporter = str(row.get('Reporter', '')).strip()
        if not well or not reporter:
            continue

        key = f"{reporter}_{well}"

        # Ct value
        raw_ct = row.get(ct_col, '')
        try:
            ct_val = float(raw_ct)
        except (ValueError, TypeError):
            ct_val = None   # "Undetermined"

        # Quantity
        raw_qty = row.get('Quantity', '')
        try:
            qty_val = float(raw_qty)
        except (ValueError, TypeError):
            qty_val = float('nan')

        # Ct threshold (per-channel, constant for all wells of same reporter)
        raw_thresh = row.get('Ct Threshold', '')
        try:
            thresh_val = float(raw_thresh)
            channel_thresholds[reporter] = thresh_val
        except (ValueError, TypeError):
            thresh_val = float('nan')

        task = str(row.get('Task', '')).strip()
        sample_name = str(row.get('Sample Name', '')).strip()
        target_name = str(row.get('Target Name', '')).strip()

        meta_entry = {
            'Sample Name':    sample_name,
            'Target Name':    target_name,
            'Task':           task,
            'Quantity':       qty_val,
            'Ct_instrument':  ct_val,
            'Ct Threshold':   thresh_val,
            'Baseline Start': row.get('Baseline Start', np.nan),
            'Baseline End':   row.get('Baseline End', np.nan),
            'HIGHSD':         str(row.get('HIGHSD', 'N')).strip().upper() == 'Y',
            'NOAMP':          str(row.get('NOAMP', 'N')).strip().upper() == 'Y',
            'EXPFAIL':        str(row.get('EXPFAIL', 'N')).strip().upper() == 'Y',
        }
        well_meta[key] = meta_entry

    return {
        'well_meta':          well_meta,
        'sample_metadata':    well_meta,   # alias — same object
        'channel_thresholds': channel_thresholds,
        'n_wells':            len(well_meta),
    }


if __name__ == "__main__":
    # Test with example data
    print("qPCR Data Converter - Testing")
    print("=" * 50)
    
    # Create test simple format
    test_df_simple = pd.DataFrame({
        'Cycle': range(1, 41),
        'A1': np.random.exponential(0.1, 40).cumsum() + 0.1,
        'A2': np.random.exponential(0.12, 40).cumsum() + 0.1,
        'A3': np.random.exponential(0.08, 40).cumsum() + 0.1,
    })
    
    # Save and test
    test_df_simple.to_csv('/tmp/test_simple.csv', index=False)
    
    converter = QPCRDataConverter()
    cycles, samples, metadata = converter.load_from_file('/tmp/test_simple.csv')
    
    print(f"\nTest Results:")
    print(f"  Format detected: {metadata['format']}")
    print(f"  Samples loaded: {metadata['n_samples']}")
    print(f"  Cycles: {metadata['n_cycles']}")
    print(f"  Sample names: {metadata['sample_names']}")
    print(f"  Cycle range: {metadata['cycle_range']}")
    
    # Show sample info
    info_df = converter.get_sample_info(samples)
    print(f"\nSample Information:")
    print(info_df.to_string(index=False))
    
    print("\n✅ Converter working correctly!")
