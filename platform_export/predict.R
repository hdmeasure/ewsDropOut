#!/usr/bin/env Rscript
# Contoh skoring siswa baru memakai model .rds.
# Mereproduksi: p_raw -> kalibrasi -> koreksi prior -> threshold.
#
# Jalankan:
#   Rscript predict.R aspd_num contoh_input.csv
# contoh_input.csv harus memuat kolom sesuai 'features' pada model.

suppressMessages({ library(xgboost); library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
name      <- if (length(args) >= 1) args[1] else "aspd_num"
input_csv <- if (length(args) >= 2) args[2] else NA

model <- readRDS(file.path("models", paste0(name, ".rds")))

logit   <- function(p) { p <- pmin(pmax(p, 1e-6), 1 - 1e-6); log(p / (1 - p)) }
sigmoid <- function(z) 1 / (1 + exp(-z))

calibrate <- function(p_raw, cal) {
  if (cal$method == "sigmoid") {
    sigmoid(cal$a * logit(p_raw) + cal$b)
  } else { # isotonic: interpolasi linier dgn clip di ujung
    approx(x = cal$x, y = cal$y, xout = p_raw, rule = 2)$y
  }
}

score <- function(model, df) {
  X <- as.matrix(df[, model$features, drop = FALSE])   # urutan fitur WAJIB
  storage.mode(X) <- "double"
  p_raw <- predict(model$booster, X)
  p_cal <- calibrate(p_raw, model$calibration)
  offset <- log(model$pi_pop / (1 - model$pi_pop)) -
            log(model$pi_train / (1 - model$pi_train))
  p_adj <- sigmoid(logit(p_cal) + offset)
  data.frame(prob_do = round(p_adj, 4),
             risiko_do = ifelse(p_adj >= model$threshold, "BERISIKO", "Tidak"))
}

if (!is.na(input_csv) && file.exists(input_csv)) {
  df <- read.csv(input_csv)
  out <- score(model, df)
  print(utils::head(out, 20))
  cat(sprintf("\nTotal: %d siswa, berisiko: %d (%.2f%%)\n",
              nrow(out), sum(out$risiko_do == "BERISIKO"),
              100 * mean(out$risiko_do == "BERISIKO")))
} else {
  # demo dengan baris dummy (semua fitur = 0) bila tak ada input
  cat("Tidak ada input CSV; contoh skoring 1 baris dummy.\n")
  df <- as.data.frame(matrix(0, nrow = 1, ncol = length(model$features)))
  names(df) <- model$features
  print(score(model, df))
  cat("\nUntuk pakai data nyata: Rscript predict.R", name, "input.csv\n")
}
