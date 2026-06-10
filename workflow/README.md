# Workflow Pemodelan Prediksi DO

Implementasi dari `PLAN_PEMODELAN_DO.md`.

**Skenario yang tersedia:**
| Skenario | Fitur ASPD yang dipakai | Catatan |
|---|---|---|
| `dengan_aspd` | read, num, sci (mentah) | ASPD bias: DO justru bernilai tinggi |
| `tanpa_aspd` | — (tanpa ASPD) | faktor risiko logis (SES) |
| `aspd_mean` | rerata 3 skor | uji apakah rerata lebih baik |
| `aspd_dev` | rerata + deviasi (std) | uji apakah kerataan skor bersinyal |
| `aspd_pola` | level R/S/T per mapel | uji apakah pola skor bersinyal |

`--scenario both` = `dengan_aspd`+`tanpa_aspd`; `--scenario all` = kelima skenario.

## Prasyarat
File berikut harus sudah ada di folder project utama:
- `datasetDO.xlsx`, `datasetNonDO_SMP.xlsx` (untuk training)
- `DATA_SISWA_DO_SMP_EKSTRAK.xlsx`, `DATA_SISWA_NON_DO_EKSTRAK.xlsx` (sumber NPSN)
- `dataset_implementasi_SMP.xlsx` (untuk m5)

Library: numpy, pandas, scikit-learn, xgboost, shap, statsmodels, matplotlib, openpyxl, joblib (semua sudah terpasang).

## Urutan menjalankan

```bash
cd "Workflow Pemodelan"

# Tahap 0-1: siapkan input (gabung label, join NPSN, encoding)
python3 m0_siapkan_input.py

# Tahap 3-4: nested CV (FS voting + tuning) -> pilih family terbaik
python3 m1_nested_cv.py                      # kedua skenario, n_iter=25
#   opsi: --scenario tanpa_aspd --n-iter 40

# Tahap 5-6: finalisasi (refit + kalibrasi + koreksi prior + threshold)
#   WAJIB isi --pi dengan base rate DO SMP riil (mis. 0.03 = 3%)
python3 m2_finalisasi_model.py --pi 0.03
#   opsi: --scenario tanpa_aspd --family xgb --pi 0.03

# Tahap 7: evaluasi hold-out test
python3 m3_evaluasi_test.py

# Tahap 8: interpretasi SHAP
python3 m4_shap_interpretasi.py

# Tahap 9: implementasi ke Dapodik (proporsi ter-flag = pi_pop)
python3 m5_implementasi_prediksi.py --scenario tanpa_aspd
```

## Implementasi FINAL (tiered) — m6

Konfigurasi deployment yang direkomendasikan: **model utama `aspd_num`, fallback `tanpa_aspd`**.
- `aspd_num` (XGBoost) dipakai untuk siswa yang punya nilai numerasi ASPD → diskriminasi
  setara ASPD penuh (ROC-AUC ~0,93) tapi faktor risikonya logis (SES rendah).
- `tanpa_aspd` (RF) dipakai untuk siswa tanpa ASPD.
- Setiap siswa berisiko diberi **alasan** (3 faktor pendorong terbesar via SHAP).

```bash
# prasyarat: model_aspd_num.joblib & model_tanpa_aspd.joblib sudah dibuat oleh m2
python3 m2_finalisasi_model.py --scenario aspd_num --pi 0.025
python3 m2_finalisasi_model.py --scenario tanpa_aspd --pi 0.025
python3 m6_implementasi_tiered.py --pi 0.025
#   bisa diganti: --primary aspd_num --fallback tanpa_aspd
```
Output: `output/prediksi_tiered.xlsx` (kolom `model_dipakai`, `prob_do`, `risiko_do`, `alasan_risiko`).

## Catatan penting

- **`--pi` (base rate populasi)** menentukan proporsi siswa yang akhirnya ditandai
  berisiko. Default proyek = **0.025 (2,5%)**. Jika belum ada angka resmi, mulai dengan
  estimasi dan sesuaikan. Tanpa `--pi`, m2 memakai prevalensi train (tanpa koreksi) dan memberi peringatan.
- **Imputasi & data tidak lengkap (m5):** training TIDAK pernah diimputasi (complete-case).
  Pada implementasi:
  - **Sulingjar yang kosong → diimputasi median** (boleh, karena level sekolah & ordinal).
  - **Fitur INTI** (jenis kelamin, pendidikan/penghasilan ortu, ASPD) yang kosong **>
    `--max-missing`** (default **0**) → siswa diberi label **"Data Tidak Lengkap"** dan **tidak
    diprediksi** — mencegah bias dari imputasi berlebihan pada variabel individu.
  - Kolom diagnostik di output: `n_inti_kosong`, `n_sulingjar_kosong`, `n_fitur_kosong`.
  - Naikkan `--max-missing` hanya bila ingin menoleransi sedikit kekosongan fitur inti.
- **Proporsi rasional dijamin di m5** mode `baserate` (default): threshold dihitung di antara
  siswa yang **diprediksi** (data lengkap) sehingga fraksi ter-flag = `pi`. Ranking risiko dari
  probabilitas model terkalibrasi.
  - Catatan: model pohon (RF) menghasilkan probabilitas diskret → proporsi bisa sedikit
    melenceng dari target karena nilai kembar (ties) di batas threshold (mis. 3,0% vs 2,5%).
    Model XGBoost/logreg lebih kontinu sehingga lebih presisi.

## Untuk mengubah prevalensi target (mis. ke 2,5%)
Cara paling koheren (probabilitas ikut terkalibrasi ke 2,5%):
```bash
python3 m2_finalisasi_model.py --pi 0.025
python3 m3_evaluasi_test.py
python3 m5_implementasi_prediksi.py --scenario both
```
Cara cepat (hanya mengubah proporsi flag, tanpa melatih ulang):
```bash
python3 m5_implementasi_prediksi.py --scenario both --pi 0.025
```
- **`tingkat_pendidikan` sengaja dibuang** (artefak sampling: DO diamati kelas 7–9,
  NonDO kini kelas 10–11).
- **GroupKFold per NPSN** dipakai di semua split & CV (cegah kebocoran Sulingjar).
- **Tanpa SMOTE** — imbalance via `class_weight`/`scale_pos_weight` + kalibrasi + koreksi prior.

## Output (folder `output/`)
| File | Isi |
|---|---|
| `model_input_<sc>.csv` | data training tergabung + berlabel |
| `split_<sc>_train/test.csv` | pembagian per sekolah (test terkunci) |
| `cv_metrics_<sc>.csv`, `cv_summary_<sc>.csv` | hasil nested CV per family |
| `feature_stability_<sc>.csv` | frekuensi terpilihnya tiap fitur antar-fold |
| `recommendation_<sc>.txt` | family terbaik |
| `fs_final_<sc>.csv` | detail voting fitur final |
| `model_<sc>.joblib` + `_info.txt` | model final (bundle) |
| `eval_test_<sc>.txt`, `curve_*_<sc>.png` | evaluasi & kurva |
| `shap_summary_<sc>.png`, `shap_importance_<sc>.csv` | interpretasi |
| `prediksi_implementasi_<sc>.xlsx` | hasil prediksi siswa SMP + ringkasan proporsi |

## Modul pendukung
- `common.py` — konfigurasi kolom, loader, encoding, metrik.
- `fs_voting.py` — feature selection hybrid voting (LR p<0.05 + RF imp + SHAP).
- `models.py` — definisi family & ruang hyperparameter.
- `scorer.py` — bundle: estimator + kalibrator + koreksi prior + threshold.
