#!/usr/bin/env Rscript
# Discovery script — enumerate everything in PCRedux + chipPCR and print
# enough structure to write targeted extractors.
#
# Usage:
#   Rscript tests/fixtures/pcredux/discover.R 2>&1 | tee discover.log
#
# Then paste discover.log into the next conversation turn.

options(width = 120)

# ── Load packages ───────────────────────────────────────────────────────
for (pkg in c("PCRedux", "chipPCR")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("INSTALL NEEDED: install.packages('", pkg, "')\n", sep = "")
  } else {
    cat("Loaded:", pkg, as.character(packageVersion(pkg)), "\n")
  }
}
suppressMessages({
  library(PCRedux)
  library(chipPCR)
})
cat("\n")

# ── For each data item, print enough to know how to extract it ─────────
describe <- function(pkg) {
  cat("\n========== package:", pkg, "==========\n")
  items <- tryCatch(
    data(package = pkg)$results[, "Item"],
    error = function(e) character(0)
  )
  if (length(items) == 0) {
    cat("  (no data items)\n"); return(invisible())
  }
  for (item in items) {
    # Strip "(...)" decoration that data() sometimes adds
    name <- sub("\\s*\\(.*\\)$", "", item)
    cat("\n--- ", pkg, "::", name, " ---\n", sep = "")
    obj <- tryCatch(
      get(name, envir = asNamespace(pkg)),
      error = function(e) { cat("  get() failed:", conditionMessage(e), "\n"); NULL }
    )
    if (is.null(obj)) next

    cat("  class       :", paste(class(obj), collapse = "/"), "\n")
    if (is.data.frame(obj) || is.matrix(obj)) {
      cat("  dim         :", paste(dim(obj), collapse = " x "), "\n")
      cn <- colnames(obj)
      cat("  first cols  :", paste(head(cn, 10), collapse = ", "),
          if (length(cn) > 10) sprintf(" ... (+%d more)", length(cn) - 10) else "", "\n")
      cat("  head 2x6    :\n")
      tryCatch(print(head(obj, 2)[, seq_len(min(6, ncol(obj))), drop = FALSE]),
               error = function(e) cat("    (could not print)\n"))
    } else if (is.list(obj)) {
      cat("  length      :", length(obj), "\n")
      nm <- names(obj)
      if (!is.null(nm)) {
        cat("  first names :", paste(head(nm, 10), collapse = ", "),
            if (length(nm) > 10) sprintf(" ... (+%d more)", length(nm) - 10) else "", "\n")
      }
      if (length(obj) >= 1) {
        cat("  obj[[1]] class:", paste(class(obj[[1]]), collapse = "/"), "\n")
        if (is.data.frame(obj[[1]]) || is.matrix(obj[[1]])) {
          cat("  obj[[1]] dim  :", paste(dim(obj[[1]]), collapse = " x "), "\n")
        }
      }
    } else {
      cat("  (unhandled type — str dump follows)\n")
      tryCatch(str(obj, max.level = 1), error = function(e) cat("    (str failed)\n"))
    }
  }
}

describe("PCRedux")
describe("chipPCR")

cat("\n\nDONE.\n")
