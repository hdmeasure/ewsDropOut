# ewsDropOut — Early Warning System for Student Dropout (SMP)

Pipeline machine learning untuk memprediksi **risiko putus sekolah (drop out / DO)**
siswa jenjang SMP, beserta workflow penyiapan data, pelatihan, evaluasi, dan ekspor
model multi-format untuk integrasi platform.

Repositori ini berisi **kode, dokumentasi, dan model terlatih** — **tidak** menyertakan
data mentah siswa (data bersifat rahasia dan dikelola terpisah).

---

## Apa yang diprediksi

Model menghasilkan, untuk tiap siswa:
- `prob_do` — probabilitas risiko DO (terkalibrasi, sudah dikoreksi ke base rate populasi),
- `risiko_do` — label `BERISIKO` / `Tidak`,
- `alasan_risiko` — faktor pendorong utama (berbasis SHAP) untuk tiap siswa berisiko.

Proporsi siswa yang ditandai berisiko dijaga **rasional** (mengikuti base rate populasi
yang dapat diatur, mis. 2,5%), sehingga jumlah sasaran intervensi tetap masuk akal.

---

## Sumber variabel (generik)

| Kelompok | Variabel |
|---|---|
| Demografi | jenis kelamin |
| Sosial-ekonomi (administratif) | pendidikan & penghasilan ayah/ibu (ordinal) |
| Asesmen standar | skor numerasi (ASPD) |
| Mutu sekolah | indikator survei lingkungan belajar (ordinal 1–3) |

Variabel akademik lain (literasi membaca & sains) **sengaja tidak dipakai** pada model
utama karena pada data kami menunjukkan hubungan yang tidak valid (artefak cakupan
sampel); lihat [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

---

## Dua model (pendekatan tiered)

| Model | Dipakai untuk | Catatan |
|---|---|---|
| `aspd_num` (utama) | siswa yang memiliki skor numerasi | diskriminasi tinggi, faktor risiko logis (sosial-ekonomi) |
| `tanpa_aspd` (fallback) | siswa tanpa skor asesmen | hanya variabel administratif + mutu sekolah |

Keduanya **XGBoost** (portabel lintas R/Python), dengan kalibrasi probabilitas +
koreksi prior + threshold berbasis base rate.

---

## Struktur repositori

```
workflow/         Skrip pipeline (data -> model final): m0..m6 + modul pendukung
platform_export/  Ekspor model ke .rds / .json / .joblib + contoh skoring R & Python
docs/             METHODOLOGY, CODEBOOK, FLOWCHART
requirements.txt  Dependensi Python
```

- Alur lengkap & diagram: [docs/FLOWCHART.md](docs/FLOWCHART.md)
- Metodologi & dasar literatur: [docs/METHODOLOGY.md](docs/METHODOLOGY.md)
- Kamus variabel: [docs/CODEBOOK.md](docs/CODEBOOK.md)
- Cara pakai model di platform: [platform_export/README.md](platform_export/README.md)

---

## Menjalankan pelatihan (dengan data Anda sendiri)

Data tidak disertakan. Sediakan dataset berlabel sesuai [docs/CODEBOOK.md](docs/CODEBOOK.md),
lalu:

```bash
pip install -r requirements.txt
cd workflow
python3 m0_siapkan_input.py
python3 m1_nested_cv.py --scenario all
python3 m2_finalisasi_model.py --scenario aspd_num  --pi 0.025
python3 m2_finalisasi_model.py --scenario tanpa_aspd --family xgb --pi 0.025
python3 m3_evaluasi_test.py
python3 m6_implementasi_tiered.py --pi 0.025
```

## Memakai model terlatih (platform)

```bash
cd platform_export
Rscript predict.R aspd_num input.csv      # R, dari .rds
python3 predict.py aspd_num input.csv     # Python, dari spec.json + booster.json
```

---

## Lisensi & data

Kode untuk keperluan internal tim platform. **Tidak ada data pribadi siswa** di
repositori ini.
