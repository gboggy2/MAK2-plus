"""
qPCR Data Converter - Handle various instrument export formats

Supports multiple formats:
- Simple format: Column 1 = cycles, other columns = samples
- Wide format: Rows = cycles, columns = wells/samples
- Plate reader format: Multi-sheet or sectioned data
- CFX format: Bio-Rad CFX Manager exports
- QuantStudio format: Applied Biosystems exports (Excel)
- ABI CSV format: Applied Biosystems CSV exports with [Section] headers
  (Multicomponent, Amplification Data)

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
    """
    Convert various qPCR instrument data formats to MAK2+ standard format.
    
    Standard format:
    - cycles: np.ndarray of cycle numbers
    - samples: Dict[str, np.ndarray] mapping sample names to fluorescence values
    """
    
    def __init__(self, add_offset: bool = True, offset_value: float = 1e-5):
        """
        Initialize converter.
        
        Parameters
        ----------
        add_offset : bool
            Add small offset to avoid zeros (important for log transforms)
        offset_value : float
            Value to add if add_offset=True
        """
        self.add_offset = add_offset
        self.offset_value = offset_value
        self.detected_format = None
        self.n_samples = 0
        self.n_cycles = 0
        
    def detect_format(self, df: pd.DataFrame) -> str:
        """
        Auto-detect the qPCR data format.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw data from file
            
        Returns
        -------
        str
            Format type: 'simple', 'wide', 'cfx', 'quantstudio', etc.
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
        """Check if a column contains cycle numbers."""
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
        """
        Load qPCR data from file and convert to standard format.
        
        Parameters
        ----------
        filepath : str, Path, or BytesIO
            Path to data file or uploaded file buffer
            
        Returns
        -------
        cycles : np.ndarray
            Cycle numbers
        samples : Dict[str, np.ndarray]
            Dictionary mapping sample names to fluorescence values
        metadata : Dict
            Information about the loaded data
            May contain 'extra_info' with target information for multiplexed data
            
        Raises
        ------
        ValueError
            If file cannot be parsed or doesn't contain valid qPCR data
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
        """
        Convert simple format: Column 0 = cycles, others = samples.
        
        Example:
        Cycle | Sample_1 | Sample_2
        1     | 0.12     | 0.15
        2     | 0.18     | 0.22
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
        """
        Convert wide format: All columns are samples, rows are cycles.
        
        Example:
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
        """
        Convert Bio-Rad CFX format.
        
        Expected columns: Well, Cycle, Fluorescence (or similar)
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
        """
        Convert Applied Biosystems QuantStudio format.
        
        Expected columns: Well Position, Cycle, ΔRn (or similar)
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
        """
        Parse Sample Setup sheet for well metadata (Sample Name, Task, Quantity).

        QuantStudio exports include a "Sample Setup" sheet with metadata about each well,
        including sample names, task types (UNKNOWN, STANDARD, NTC), and known quantities
        for standards. The header row position varies (typically around row 46).

        Parameters
        ----------
        xls : pd.ExcelFile
            Open Excel file handle

        Returns
        -------
        pd.DataFrame or None
            DataFrame with columns like Well, Well Position, Sample Name, Target Name,
            Task, Quantity. Returns None if sheet not found or unparseable.
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
        """
        Parse Results sheet for instrument-reported CT values and thresholds.

        QuantStudio exports include a "Results" sheet with per-well CT, Ct Threshold,
        and other analysis results. The header row position varies.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with columns like Well Position, Target Name, CT, Ct Threshold.
            Returns None if sheet not found or unparseable.
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
        """
        Try to read QuantStudio format from multi-sheet Excel file.

        Looks for "Amplification Data" sheet and skips header rows.
        Also reads "Sample Setup" sheet for well metadata (Sample Name, Task, Quantity).

        Returns
        -------
        cycles : np.ndarray
        samples : Dict[str, np.ndarray]
        extra_info : Dict
            Contains 'targets' for multiplexed data and 'sample_setup' for well metadata
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
        """
        Convert a parsed ABI CSV section into the standard load_from_file() return format.
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
        """
        Try to parse an Applied Biosystems CSV export with #-comment headers
        and [Section Name] markers.

        Returns
        -------
        section_name : str
            Name of the section used (e.g. 'Multicomponent', 'Amplification Data')
        section_df : pd.DataFrame
            Parsed data for that section
        header_meta : Dict[str, str]
            Key-value pairs from the # comment header lines
            (e.g. {'Passive Reference': 'ROX', 'Instrument Type': '7500 Fast System'})

        Raises
        ------
        ValueError
            If the file does not look like an ABI CSV or contains no usable section
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
        """
        Parse a [Multicomponent] section DataFrame.

        Returns
        -------
        cycles : np.ndarray
        samples_by_channel : Dict[channel_name → Dict[well_pos → np.ndarray]]
        dye_columns : List[str]
            All dye channel names found (including passive reference)
        passive_reference : str or None
            Name of the passive reference dye (e.g. 'ROX'), or None
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
        """
        Parse an [Amplification Data] section DataFrame (single ΔRn channel).

        Returns
        -------
        cycles : np.ndarray
        samples : Dict[well_pos → np.ndarray]
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
        """
        Extract samples for a single dye channel from ABI Multicomponent data.

        Parameters
        ----------
        extra_info : Dict
            extra_info dict returned by load_from_file() when requires_channel_selection=True
        channel_name : str
            Dye channel to extract (e.g. 'FAM', 'JOE')
        normalize_by_reference : bool
            If True, divide fluorescence by the passive reference channel (e.g. ROX)
            to correct for well-to-well volume variation

        Returns
        -------
        cycles : np.ndarray
        samples : Dict[str, np.ndarray]
        None
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
        """
        Generate summary information about loaded samples.
        
        Parameters
        ----------
        samples : Dict[str, np.ndarray]
            Sample dictionary from load_from_file
            
        Returns
        -------
        pd.DataFrame
            Summary with columns: Sample, Min, Max, Mean, Detectable
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
        """
        Filter multiplexed qPCR data by target name.

        Parameters
        ----------
        extra_info : Dict
            Extra info dict from _try_quantstudio_multisheet containing raw_df
            and optionally sample_setup DataFrame
        target_name : str
            Target name to filter for

        Returns
        -------
        cycles : np.ndarray
            Cycle numbers
        samples : Dict[str, np.ndarray]
            Samples for the selected target only
        sample_metadata : Dict or None
            Per-well metadata dict keyed by well position, e.g.:
            {'A1': {'Sample Name': 'Patient_1', 'Task': 'UNKNOWN', 'Quantity': NaN}, ...}
            None if Sample Setup sheet was not available.
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
        """
        Filter multiplexed qPCR data for ALL targets at once.

        Parameters
        ----------
        extra_info : Dict
            Extra info dict from _try_quantstudio_multisheet.

        Returns
        -------
        dict
            Keyed by target name, values are (cycles, samples, sample_metadata)
            tuples — same as filter_by_target() returns for each target.
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
    """
    Parse an ABI results CSV (exported from QuantStudio / StepOnePlus results table).

    The file has columns:
        Well, Sample Name, Target Name, Task, Reporter, Quencher,
        C_ (Ct), C_ Mean, C_ SD, Quantity, ...,
        Ct Threshold, Baseline Start, Baseline End,
        HIGHSD, NOAMP, EXPFAIL

    Returns
    -------
    dict with keys:
        'well_meta'           : {f"{Reporter}_{Well}": per-well dict}
        'sample_metadata'     : same dict — drop-in for st.session_state.sample_metadata
        'channel_thresholds'  : {'FAM': 0.1, 'JOE': 0.04, ...}
        'n_wells'             : int
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
