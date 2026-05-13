#!/usr/bin/env Rscript
# Extract PCRedux + chipPCR labelled amplification-curve datasets to CSV.
#
# Produces, under tests/fixtures/pcredux_extended/<dataset>/:
#   - curves.csv  (long format: curve_id, cycle, fluorescence)
#   - labels.csv  (curve_id + derived consensus label + raw rater columns)
#
# Datasets harvested (everything in PCRedux/chipPCR that's both
# amplification-curve data AND has accessible rater decisions):
#   - PCRedux::kbqPCR   (400 wells, 7 raters + conformity column)
#   - PCRedux::RAS002   (192 wells, single y/n factor)
#
# Usage:
#   Rscript tests/fixtures/pcredux/extract.R 2>&1 | tee /tmp/extract.log

suppressMessages({ library(PCRedux); library(chipPCR) })
options(width = 120)

out_root <- "tests/fixtures/pcredux_extended"
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

# ── Helpers ─────────────────────────────────────────────────────────────

# Convert a wide curves matrix (first col = cycle, rest = wells) to long format.
wide_to_long <- function(df, cycle_col, well_cols) {
  curves <- data.frame(curve_id = character(0), cycle = numeric(0),
                       fluorescence = numeric(0), stringsAsFactors = FALSE)
  cycle_vec <- df[[cycle_col]]
  for (w in well_cols) {
    curves <- rbind(curves, data.frame(
      curve_id     = rep(w, nrow(df)),
      cycle        = cycle_vec,
      fluorescence = df[[w]],
      stringsAsFactors = FALSE
    ))
  }
  curves
}

# Consensus from a decision row: if PCRedux's own "conformity" flag is TRUE
# (all 7 raters agreed), use the unanimous label; otherwise mark "a".
# Cleaner than majority-vote and matches the package's own semantics.
consensus_from_row <- function(decision_row, rater_cols) {
  conformity <- isTRUE(as.logical(decision_row$conformity))
  if (!conformity) return("a")
  # All raters agree — read any of them
  as.character(decision_row[[rater_cols[1]]])
}

# ── kbqPCR (400 wells, 7 raters) ────────────────────────────────────────

cat("=== kbqPCR ===\n")
kb <- PCRedux::kbqPCR
kb_dec <- PCRedux::decision_res_kbqPCR
cat("kbqPCR dim     :", dim(kb), "\n")
cat("decision cols  :", paste(colnames(kb_dec), collapse = ", "), "\n")

well_cols <- setdiff(colnames(kb), "cyc")
stopifnot(length(well_cols) == 400)
stopifnot(nrow(kb_dec) == 400)

# Sanity: kb_dec$kbqPCR should match column names of kb
if (!all(kb_dec$kbqPCR == well_cols)) {
  # Some datasets list curves in decision-table order rather than column order;
  # reindex if needed.
  cat("  Note: well order in decision table differs from curves; reindexing.\n")
  ord <- match(well_cols, kb_dec$kbqPCR)
  if (anyNA(ord)) stop("kbqPCR: some wells have no decision row")
  kb_dec <- kb_dec[ord, ]
}

curves_kb <- wide_to_long(kb, "cyc", well_cols)

rater_cols <- grep("^test\\.result", colnames(kb_dec), value = TRUE)
cat("rater cols     :", paste(rater_cols, collapse = ", "), "\n")
labels_kb <- data.frame(
  curve_id = well_cols,
  label    = vapply(seq_len(nrow(kb_dec)),
                    function(i) consensus_from_row(kb_dec[i, ], rater_cols),
                    character(1)),
  conformity = as.character(kb_dec$conformity),
  stringsAsFactors = FALSE
)
# Append raw rater columns for downstream per-rater analysis
for (rc in rater_cols) labels_kb[[rc]] <- as.character(kb_dec[[rc]])

dir.create(file.path(out_root, "kbqPCR"), showWarnings = FALSE)
write.csv(curves_kb, file.path(out_root, "kbqPCR", "curves.csv"), row.names = FALSE)
write.csv(labels_kb, file.path(out_root, "kbqPCR", "labels.csv"), row.names = FALSE)
cat("wrote          :", nrow(curves_kb), "curve rows,",
    nrow(labels_kb), "labels →", file.path(out_root, "kbqPCR/"), "\n")
cat("label dist     :", table(labels_kb$label), "\n\n")

# ── RAS002 (192 wells, 1 rater) ─────────────────────────────────────────

cat("=== RAS002 ===\n")
ras <- PCRedux::RAS002
ras_dec <- PCRedux::RAS002_decisions
cat("RAS002 dim     :", dim(ras), "\n")
cat("RAS002 decision class:", class(ras_dec), " len:", length(ras_dec), "\n")

well_cols <- setdiff(colnames(ras), "cyc")
stopifnot(length(well_cols) == length(ras_dec))

curves_ras <- wide_to_long(ras, "cyc", well_cols)
labels_ras <- data.frame(
  curve_id = well_cols,
  label    = as.character(ras_dec),
  stringsAsFactors = FALSE
)

dir.create(file.path(out_root, "RAS002"), showWarnings = FALSE)
write.csv(curves_ras, file.path(out_root, "RAS002", "curves.csv"), row.names = FALSE)
write.csv(labels_ras, file.path(out_root, "RAS002", "labels.csv"), row.names = FALSE)
cat("wrote          :", nrow(curves_ras), "curve rows,",
    nrow(labels_ras), "labels →", file.path(out_root, "RAS002/"), "\n")
cat("label dist     :", table(labels_ras$label), "\n\n")

# ── Manifest ────────────────────────────────────────────────────────────

manifest <- data.frame(
  dataset    = c("kbqPCR", "RAS002"),
  source     = c("PCRedux::kbqPCR + decision_res_kbqPCR",
                 "PCRedux::RAS002 + RAS002_decisions"),
  n_curves   = c(length(unique(curves_kb$curve_id)),
                 length(unique(curves_ras$curve_id))),
  n_cycles   = c(length(unique(curves_kb$cycle)),
                 length(unique(curves_ras$cycle))),
  n_raters   = c(7, 1),
  consensus  = c("PCRedux conformity flag (all 7 agree → label, else 'a')",
                 "single decision"),
  stringsAsFactors = FALSE
)
write.csv(manifest, file.path(out_root, "manifest.csv"), row.names = FALSE)
cat("=== manifest ===\n")
print(manifest)
cat("\nDONE.\n")
