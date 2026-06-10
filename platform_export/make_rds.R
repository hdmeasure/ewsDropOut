#!/usr/bin/env Rscript
# Membangun file .rds dari artefak ekspor (booster.json + spec.json).
# Hasil .rds berisi list: booster (xgb.Booster) + spec lengkap (fitur, kalibrasi,
# koreksi prior, threshold), siap dipakai platform berbasis R.
#
# Jalankan:
#   Rscript make_rds.R                 # proses semua model di ./models
#   Rscript make_rds.R aspd_num        # satu model

suppressMessages({
  library(xgboost)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
models_dir <- "models"
names <- if (length(args) > 0) args else c("aspd_num", "tanpa_aspd")

build_rds <- function(name) {
  booster_path <- file.path(models_dir, paste0(name, "_booster.json"))
  spec_path    <- file.path(models_dir, paste0(name, "_spec.json"))
  if (!file.exists(booster_path) || !file.exists(spec_path)) {
    cat(sprintf("[LEWAT] %s: artefak belum lengkap\n", name)); return(invisible())
  }
  booster <- xgb.load(booster_path)
  spec    <- fromJSON(spec_path, simplifyVector = TRUE)

  model <- list(
    name        = spec$name,
    booster     = booster,
    features    = spec$features,        # urutan WAJIB
    calibration = spec$calibration,
    pi_train    = spec$pi_train,
    pi_pop      = spec$pi_pop,
    threshold   = spec$threshold
  )
  out <- file.path(models_dir, paste0(name, ".rds"))
  saveRDS(model, out)
  cat(sprintf("[%s] -> %s (fitur=%d, kalibrasi=%s)\n",
              name, out, length(spec$features), spec$calibration$method))
}

invisible(lapply(names, build_rds))
cat("Selesai. File .rds siap dipakai (lihat predict.R untuk contoh skoring).\n")
