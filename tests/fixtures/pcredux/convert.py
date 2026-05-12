"""Convert Roediger et al. supplementary qPCR table to MAK2-readable CSVs.

The PCRedux paper (Roediger et al., bioRxiv 2021; doi:10.1101/2021.03.31.437921)
ships supplementary table `media-1.xlsx` containing:

  - **Studies**          12 contributing studies (chemistry, instrument, ...)
  - **Raw qPCR data**    1024 columns = 1 cycle column + 1023 amplification
                          curves (only 5 of the 12 studies; the other 7 are
                          labeled-only with no raw fluorescence)
  - **Human Rating**     3 expert raters × 3181 curves + consensus column
                          (y / n / a = positive / negative / ambiguous)
  - **Features**         91 pre-computed numeric features per curve (PCRedux's
                          ML pipeline uses these; we ignore them for now)
  - **RF Predictions**   PCRedux's Random-Forest predictions across 100 splits
                          (used by the paper to publish accuracy numbers)

For MAK2 gate tuning we need (raw_fluorescence, consensus_label) per curve.
The intersection is 1023 curves from 5 studies. The other 2158 curves carry
labels but no raw data and are unusable for our purpose.

Output schemas:

  - **pcredux_curves.csv**  (long format, ~46k rows for 1023 curves × 45 cycles)
        curve_id, dataset, cycle, fluorescence

  - **pcredux_labels.csv**  (one row per curve, 1023 rows)
        curve_id, dataset, label, label_rater1, label_rater2, label_rater3,
        rater_conformity

Label mapping for downstream MAK2 work (applied at consumer side, not here):
    y → PASS, n → FAIL, a → INDETERMINATE

Usage:
    python tests/fixtures/pcredux/convert.py <path/to/media-1.xlsx>

If the source xlsx is committed alongside this script (default), the
convert.py path defaults to ``media-1.xlsx`` in the same directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def convert(source_xlsx: Path) -> tuple[int, int]:
    """Read the xlsx and emit the two CSVs. Returns (n_curves, n_labels)."""
    raw = pd.read_excel(source_xlsx, sheet_name="Raw qPCR data")
    hr  = pd.read_excel(source_xlsx, sheet_name="Human Rating", header=[0, 1])

    # The Human Rating sheet is a multi-header with three rater blocks.
    # Each block uses the same Run column; pick rater #3's because it's the
    # one carrying the consensus column too.
    runs = hr[("Rater #3 (ANS)", "Run")]
    r1 = hr[("Rater #1 (SR)", "test.result.1")]
    r2 = hr[("Rater #2 (KB)", "test.result.1")]
    r3 = hr[("Rater #3 (ANS)", "test.result.1")]
    consensus = hr[("Rater #3 (ANS)", "Consens ")]  # trailing space is in the file
    conformity = (r1 == r2) & (r2 == r3)

    labels_df = pd.DataFrame({
        "curve_id":         runs,
        "dataset":          runs.str.split(".").str[0],
        "label":            consensus,
        "label_rater1":     r1,
        "label_rater2":     r2,
        "label_rater3":     r3,
        "rater_conformity": conformity,
    })

    # Restrict to curves that have raw fluorescence too — the intersection
    # is what we can actually run MAK2 against.
    raw_curve_cols = set(raw.columns[1:])
    keep = labels_df["curve_id"].isin(raw_curve_cols)
    labels_df = labels_df[keep].reset_index(drop=True)

    # Long-format curve table: one row per (curve_id, cycle). Drop trailing
    # NaN rows that the source xlsx pads to 50 cycles even when the actual
    # run was 40 or 45.
    cycles_arr = raw["Cycles"].astype(int)
    curve_rows = []
    for curve_id in labels_df["curve_id"]:
        fluor = raw[curve_id]
        valid = fluor.notna()
        sub = pd.DataFrame({
            "curve_id":     curve_id,
            "dataset":      curve_id.split(".")[0],
            "cycle":        cycles_arr[valid].values,
            "fluorescence": fluor[valid].values,
        })
        curve_rows.append(sub)
    curves_df = pd.concat(curve_rows, ignore_index=True)

    # Write outputs alongside this script
    curves_path = HERE / "pcredux_curves.csv"
    labels_path = HERE / "pcredux_labels.csv"
    curves_df.to_csv(curves_path, index=False)
    labels_df.to_csv(labels_path, index=False)

    return len(labels_df), len(curves_df)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "media-1.xlsx"
    if not src.exists():
        sys.exit(
            f"source xlsx not found: {src}\n"
            "Pass the path to Roediger et al.'s media-1.xlsx as the first arg."
        )
    n_curves, n_rows = convert(src)
    print(f"wrote {n_curves} curves, {n_rows} (curve, cycle) rows")


if __name__ == "__main__":
    main()
