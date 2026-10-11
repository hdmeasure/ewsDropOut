# Rencana Paket R `ewsDropOut` → SAGE *Applied Psychological Measurement* (APM)

Status: draf ide (belum ada kode paket). Dibuat dari hasil riset + isi repo saat ini.

## 0. Temuan riset kunci

1. **Fit ke APM adalah risiko utama.** Scope APM = psikometri (IRT, DIF, reliabilitas, CAT, linking). Halaman guideline hanya menyebut *computer program review* yang **berdasarkan undangan** (Computer Software Review Editor, ±6 halaman) — bukan artikel paket R reguler. Prediksi DO murni (XGBoost + SHAP) akan dianggap di luar scope.
   → Naskah harus berupa **artikel metodologis psikometrik** dengan paket sebagai pendamping, atau hubungi editor lebih dulu. *(Saya tidak bisa membuka halaman aims & scope langsung; verifikasi manual diperlukan.)*
2. **Celah paket:** tidak ada paket CRAN untuk EWS DO sekolah. `dropout` (CRAN) = survei, `EWS` = krisis ekonomi. Terdekat: `EWStools` (Wisconsin DEWS; Knowles, 2015, *JEDM*) — berbasis ROC, tanpa kalibrasi/SHAP/fairness, tidak aktif.
3. **Literatur pendukung:** Bowers et al. (2013) – 110 indikator DO, banyak presisi tinggi tapi akurasi rendah; Bird et al. (2025, *JPAM*) – bias algoritmik pada prediksi keberhasilan siswa, sangat bergantung definisi "at-risk"; Yu et al. – menyertakan atribut terproteksi hampir tak mengubah fairness; koreksi intercept untuk case-control (King & Zeng, 2001); rmda untuk net benefit (CRAN, 2018, tua).
4. **Aset unik repo ini yang tidak ada di paket lain:** (a) koreksi prior case-control → base rate populasi + threshold berbasis base rate, (b) CV bersarang per-sekolah, (c) deployment *tiered* (data lengkap/parsial), (d) **temuan artefak cakupan sampel** (skor literasi/sains berkorelasi terbalik dgn DO karena hanya sebagian siswa DO punya skor) — ini isu **seleksi/range restriction**, topik klasik psikometri.

## Rencana A — "Validitas skor risiko": prediksi diferensial, seleksi, & kalibrasi

**Klaim ilmiah:** Skor EWS adalah *tes* dengan keputusan berisiko tinggi; harus diuji validitas prediktif & invariansinya seperti tes lain. Paket menyediakan perangkat psikometrik untuk itu.

| Modul | Isi | Akar psikometri |
|---|---|---|
| `ews_fit()` | XGBoost/logit native R, nested CV per-sekolah (grup), koreksi prior, threshold base rate | King & Zeng; Cawley & Talbot |
| `ews_calibrate()` | kalibrasi + diagnosis (slope/intercept, ECE, Brier) pada prior-shift | validitas prediktif |
| `ews_diff_prediction()` | uji prediksi diferensial lintas gender/SES/sekolah (Cleary-style: intercept & slope per grup; kalibrasi per grup; recall/FPR per grup pada threshold) | bias tes / differential prediction |
| `ews_selection_check()` | diagnostik artefak cakupan (missing-not-at-random pada skor asesmen): bandingkan DO ber-skor vs tanpa skor, koreksi seleksi (Heckman/IPW), tanda kebalikan | range restriction, Thorndike |
| `ews_tiered()` | model utama/fallback + "Data Tidak Lengkap" | missing data design |
| `ews_explain()` | SHAP per siswa + alasan teks (Indonesia/Inggris) | — |
| `ews_decision()` | net benefit / kurva kapasitas intervensi (top-k%) | utilitas keputusan |

**Kontribusi untuk APM:** kerangka + simulasi Monte Carlo (prior shift × seleksi × bias kelompok) + studi kasus data SMP (anonim/sintetis). **Pertanyaan riset:** seberapa besar bias seleksi pada skor asesmen mengubah kalibrasi & fairness EWS, dan apakah diagnostik yang diusulkan mendeteksinya?
**Risiko:** moderat-rendah; paling realistis diterima. Data rahasia → wajib `ews_simulate()` (data sintetis) agar reproducible.

## Rencana B — "Pengukuran → risiko": EWS yang sadar galat pengukuran (IRT)

**Klaim ilmiah:** Prediktor EWS adalah *hasil pengukuran* (skor numerasi ASPD, indikator survei mutu sekolah ordinal 1–3, SES ordinal). Memperlakukannya sebagai angka mentah mengabaikan galat ukur & ketidakpastian. Paket mengintegrasikan model pengukuran ke pipeline prediksi.

| Modul | Isi |
|---|---|
| `ews_scale()` | skala ordinal → laten via Graded Response Model (pembungkus `mirt`), reliabilitas/SE per siswa & sekolah (multilevel: mutu sekolah level-2) |
| `ews_pv()` | *plausible values* / EAP+SE dipropagasi ke model risiko (multiple imputation style, aturan Rubin) |
| `ews_risk_interval()` | interval ketidakpastian prob_do (propagasi galat ukur + konformal) |
| `ews_dif_check()` | DIF butir survei/asesmen lintas gender/wilayah sebelum dipakai prediksi |
| `ews_tiered()` | skor asesmen hilang = desain data tidak lengkap, bukan imputasi naif |
| + `ews_fit/calibrate/explain` | seperti Rencana A (inti dipakai bersama) |

**Pertanyaan riset:** apakah koreksi galat pengukuran memperbaiki kalibrasi, stabilitas peringkat siswa berisiko, dan fairness dibanding skor mentah? Simulasi: variasi reliabilitas, jumlah butir, prevalensi.
**Kontribusi untuk APM:** jelas psikometri (IRT/DIF/reliabilitas) → fit scope sangat baik. **Risiko:** lebih tinggi — butuh data level butir (belum ada di repo; repo hanya skor/indikator agregat), dependensi `mirt`, kerja metodologis lebih berat.

## Perbandingan & rekomendasi

| | A | B |
|---|---|---|
| Fit APM | sedang (lewat validitas/bias/seleksi) | kuat (IRT, DIF, galat ukur) |
| Kesiapan dari repo | tinggi (±70% sudah ada) | sedang (perlu data butir) |
| Kebaruan | menengah | tinggi |
| Risiko | rendah | sedang-tinggi |

**Rekomendasi:** mulai **A sebagai fondasi** (modul inti `fit/calibrate/diff_prediction/selection_check` dipakai keduanya), lalu tambahkan modul skala `mirt` dari B bila data level butir tersedia → artikel gabungan "A + sebagian B" paling kuat untuk APM. Lakukan **email ke editor APM dulu** (pre-submission inquiry) soal jenis artikel.

## Langkah teknis & publikasi

1. Port workflow Python → R native (`xgboost`, `rsample` group CV, `shapviz`/`fastshap`); model `.rds` sekarang jadi contoh vignette.
2. `ews_simulate()` data sintetis (mereplikasi prevalensi case-control, artefak seleksi).
3. Dokumentasi: roxygen2, `testthat`, vignette, pkgdown; cek `R CMD check`, nama bebas di CRAN, lisensi (MIT/GPL-3).
4. Arsip Zenodo (DOI) + replikasi kode manuskrip; pertimbangkan *Journal of Statistical Software*/*R Journal* sebagai opsi cadangan bila APM menolak scope.
5. Etika: izin data siswa, tidak menyertakan data mentah, deklarasi penggunaan (bukan untuk sanksi siswa).

## Referensi riset
- APM submission guidelines: https://journals.sagepub.com/author-instructions/APM
- Knowles (2015) Wisconsin DEWS: https://jedm.educationaldatamining.org/index.php/JEDM/article/view/JEDM082
- Bird, Castleman & Song (2025), *JPAM*, doi:10.1002/pam.22569 (EdWorkingPaper 23-717)
- Bowers, Sprott & Taff (2013), *The High School Journal*
- CRAN `dropout`, `EWS`, `rmda` (cek keberadaan/kompetitor)
