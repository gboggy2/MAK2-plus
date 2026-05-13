#!/usr/bin/env Rscript
# Second-pass discovery: enumerate the datasets referenced in
# PCRedux::data_sample and what's available in the qpcR package.
#
# Usage:
#   Rscript tests/fixtures/pcredux/discover_more.R 2>&1 | tee /tmp/discover_more.log

options(width = 120)
suppressMessages({ library(PCRedux); library(chipPCR) })

# ── data_sample: what datasets does it cover? ───────────────────────────
cat("========== PCRedux::data_sample ==========\n")
ds <- PCRedux::data_sample
cat("dim:", dim(ds), "\n")
cat("columns (first 15):\n  ", paste(head(colnames(ds), 15), collapse=", "), "\n\n")
cat("dataset × decision crosstab:\n")
print(table(ds$dataset, ds$decision))
cat("\nrows per dataset:\n")
print(sort(table(ds$dataset), decreasing = TRUE))

# Sample 'runs' values to see what the curve identifier looks like per dataset
cat("\nsample 'runs' values per dataset (first 5):\n")
for (d in unique(ds$dataset)) {
  cat(sprintf("  %-15s: %s\n", d,
              paste(head(ds$runs[ds$dataset == d], 5), collapse = ", ")))
}

# ── qpcR package: is it installed? what raw datasets are available? ─────
cat("\n========== package: qpcR ==========\n")
if (!requireNamespace("qpcR", quietly = TRUE)) {
  cat("qpcR NOT INSTALLED. Run install.packages('qpcR') to harvest from it.\n")
} else {
  suppressMessages(library(qpcR))
  cat("qpcR version:", as.character(packageVersion("qpcR")), "\n\n")
  items <- tryCatch(data(package = "qpcR")$results[, "Item"],
                    error = function(e) character(0))
  for (item in items) {
    name <- sub("\\s*\\(.*\\)$", "", item)
    obj <- tryCatch(get(name, envir = asNamespace("qpcR")),
                    error = function(e) NULL)
    if (is.null(obj)) next
    cls <- paste(class(obj), collapse="/")
    if (is.data.frame(obj) || is.matrix(obj)) {
      cat(sprintf("  %-25s %-15s dim=%s\n", name, cls,
                  paste(dim(obj), collapse=" x ")))
    } else if (is.list(obj)) {
      cat(sprintf("  %-25s %-15s len=%d\n", name, cls, length(obj)))
    } else {
      cat(sprintf("  %-25s %-15s\n", name, cls))
    }
  }
}

cat("\nDONE.\n")
