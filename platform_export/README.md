# Ekspor Model untuk Platform

Model final tersedia dalam beberapa format agar mudah diintegrasikan di lingkungan apa pun.

## Format yang disediakan (folder `models/`)

| File | Untuk | Keterangan |
|---|---|---|
| `<name>.rds` | **R** | list berisi booster XGBoost + spec (fitur, kalibrasi, threshold) |
| `<name>_booster.json` | R / Python / lainnya | model XGBoost native (lintas-bahasa) |
| `<name>_spec.json` | semua runtime | spesifikasi skoring (sumber kebenaran) |
| `<name>.joblib` | **Python** | bundle siap pakai |

`<name>` = `aspd_num` (utama) dan `tanpa_aspd` (fallback).

## Alur skoring (sama di semua format)

```
p_raw  = booster.predict(X[features])              # urutan fitur sesuai spec
p_cal  = calibrate(p_raw)                           # sigmoid: 1/(1+exp(-(a*logit(p)+b)))
                                                    # isotonic: interpolasi linier (x,y)
offset = logit(pi_pop) - logit(pi_train)
p_adj  = sigmoid(logit(p_cal) + offset)             # koreksi prior ke base rate
label  = p_adj >= threshold
```

> **Penting:** urutan kolom input **wajib** mengikuti `features` pada `*_spec.json`.

## Memakai di R

```r
Rscript predict.R aspd_num input.csv
```
`input.csv` memuat kolom sesuai `features`. Lihat `predict.R` untuk fungsi `score()`.

## Memakai di Python

```bash
python3 predict.py aspd_num input.csv          # jalur portabel (booster.json + spec.json)
```

## Regenerasi artefak (bila model dilatih ulang)

```bash
# 1. ekspor dari bundle joblib hasil pelatihan
python3 export_model.py --in-dir /path/ke/output --out-dir models

# 2. bangun .rds
Rscript make_rds.R
```

## Threshold operasional

`threshold` pada spec dihitung dari data latih. Saat diterapkan ke populasi penuh,
sebaiknya **hitung ulang threshold** sebagai kuantil `1 - pi_pop` dari `p_adj` populasi
tersebut, agar proporsi siswa ter-flag tepat = `pi_pop` (mis. 2,5%). Lihat
`workflow/m6_implementasi_tiered.py`.

## Dependensi
- R: `xgboost`, `jsonlite`
- Python: `xgboost`, `numpy`, `pandas` (lihat `../requirements.txt`)
