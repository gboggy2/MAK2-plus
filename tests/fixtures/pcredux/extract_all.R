#!/usr/bin/env Rscript
# Mass extraction of every labelled dataset referenced in
# PCRedux::data_sample. Labels come from data_sample (decision column);
# raw curves come from whichever package owns the dataset of that name
# (qpcR for most, chipPCR for some, PCRedux for RAS002/RAS003).
#
# Output: tests/fixtures/pcredux_extended/<dataset>/
#   - curves.csv  (curve_id, cycle, fluorescence)
#   - labels.csv  (curve_id, label, dataset)
#
# Skips RAS002 (already extracted directly from PCRedux::RAS002 with
# correct cycle labels; data_sample's RAS002 rows refer to the same data).
#
# Usage:
#   Rscript tests/fixtures/pcredux/extract_all.R 2>&1 | tee /tmp/extract_all.log

suppressMessages({
  library(PCRedux); library(chipPCR)
  if (!requireNamespace("qpcR", quietly = TRUE)) {
    stop("qpcR not installed. Run install.packages('qpcR') first.")
  }
  library(qpcR)
})
options(width = 120)

out_root <- "tests/fixtures/pcredux_extended"
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

SKIP <- c("RAS002")  # already extracted with cleaner labels via extract.R

# ── Source-package lookup ───────────────────────────────────────────────
# Try each package in order. Returns (object, package_name) or (NULL, NULL).
find_dataset <- function(name) {
  for (pkg in c("qpcR", "PCRedux", "chipPCR")) {
    obj <- tryCatch(get(name, envir = asNamespace(pkg)),
                    error = function(e) NULL)
    if (!is.null(obj)) return(list(obj = obj, pkg = pkg))
  }
  list(obj = NULL, pkg = NULL)
}

# Heuristic: find the column holding the cycle index. Look for common
# names first; fall back to "first numeric column whose values are
# monotonically increasing integers."
detect_cycle_col <- function(df) {
  candidates <- c("cyc", "Cycle", "Cycles", "cycle", "cycles", "Index")
  for (c in candidates) {
    if (c %in% colnames(df)) return(c)
  }
  for (c in colnames(df)) {
    v <- df[[c]]
    if (is.numeric(v) && all(!is.na(v)) && all(diff(v) > 0)) return(c)
  }
  NULL
}

# ── Per-dataset extraction ──────────────────────────────────────────────
extract_one <- function(dataset_name, labels_for_ds) {
  cat("=== ", dataset_name, " ===\n", sep = "")

  if (dataset_name %in% SKIP) {
    cat("  SKIP (already extracted via extract.R)\n")
    return(invisible())
  }

  found <- find_dataset(dataset_name)
  if (is.null(found$obj)) {
    cat("  NOT FOUND in qpcR / PCRedux / chipPCR — skipping\n")
    return(invisible())
  }
  cat("  source: ", found$pkg, "::", dataset_name,
      "  class: ", paste(class(found$obj), collapse = "/"), "\n", sep = "")

  obj <- found$obj
  if (is.matrix(obj)) obj <- as.data.frame(obj, check.names = FALSE)
  if (!is.data.frame(obj)) {
    cat("  UNHANDLED TYPE — skipping\n")
    return(invisible())
  }

  cycle_col <- detect_cycle_col(obj)
  if (is.null(cycle_col)) {
    cat("  cannot detect cycle column — skipping\n")
    return(invisible())
  }
  cat("  cycle column: ", cycle_col, "  dim: ",
      paste(dim(obj), collapse = " x "), "\n", sep = "")

  # Match labels' run identifiers to data columns
  wanted <- as.character(labels_for_ds$runs)
  have <- intersect(wanted, colnames(obj))
  missing <- setdiff(wanted, colnames(obj))
  cat("  matched columns: ", length(have), "/", length(wanted), "\n", sep = "")
  if (length(missing) > 0 && length(missing) <= 8) {
    cat("    missing: ", paste(missing, collapse = ", "), "\n", sep = "")
  } else if (length(missing) > 0) {
    cat("    missing: ", paste(head(missing, 5), collapse = ", "),
        " ... (+", length(missing) - 5, " more)\n", sep = "")
  }
  if (length(have) == 0) {
    cat("  no usable curves — skipping\n")
    return(invisible())
  }

  # Build long-format curves table
  cyc_vec <- obj[[cycle_col]]
  curves <- do.call(rbind, lapply(have, function(w) {
    data.frame(curve_id = w, cycle = cyc_vec, fluorescence = obj[[w]],
               stringsAsFactors = FALSE)
  }))

  # Pair labels (data_sample$decision is the consensus label as a factor;
  # convert to character; subset to matched runs).
  lab <- labels_for_ds[labels_for_ds$runs %in% have, c("runs", "decision", "dataset")]
  colnames(lab) <- c("curve_id", "label", "dataset")
  lab$label <- as.character(lab$label)

  # Write out
  out_dir <- file.path(out_root, dataset_name)
  dir.create(out_dir, showWarnings = FALSE)
  write.csv(curves, file.path(out_dir, "curves.csv"), row.names = FALSE)
  write.csv(lab,    file.path(out_dir, "labels.csv"), row.names = FALSE)
  cat("  wrote: ", nrow(curves), " curve rows + ", nrow(lab),
      " labels  (y=", sum(lab$label == "y"),
      ", n=", sum(lab$label == "n"), ")\n", sep = "")
}

# ── Main loop ───────────────────────────────────────────────────────────
ds_table <- PCRedux::data_sample
cat("data_sample dim: ", paste(dim(ds_table), collapse = " x "),
    "  unique datasets: ", length(unique(ds_table$dataset)), "\n\n", sep = "")

# Process largest datasets last so smaller ones get cleaner per-dataset
# reporting up top.
ds_order <- names(sort(table(ds_table$dataset), decreasing = FALSE))
for (d in ds_order) {
  extract_one(d, ds_table[ds_table$dataset == d, ])
}

# ── Update manifest ─────────────────────────────────────────────────────
existing_dirs <- list.dirs(out_root, recursive = FALSE)
m <- do.call(rbind, lapply(existing_dirs, function(d) {
  lf <- file.path(d, "labels.csv")
  if (!file.exists(lf)) return(NULL)
  lab <- read.csv(lf, stringsAsFactors = FALSE)
  data.frame(
    dataset    = basename(d),
    n_curves   = nrow(lab),
    n_y        = sum(lab$label == "y"),
    n_n        = sum(lab$label == "n"),
    n_a        = sum(lab$label == "a"),
    stringsAsFactors = FALSE
  )
}))
write.csv(m, file.path(out_root, "manifest.csv"), row.names = FALSE)
cat("\n=== final manifest ===\n")
print(m)
cat("\nDONE.\n")
