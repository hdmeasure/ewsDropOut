# Metodologi

Ringkasan pendekatan pemodelan EWS Drop Out, beserta dasar literatur dan keputusan desain.

## 1. Definisi populasi & label

- **DO (positif)**: siswa yang putus sekolah saat jenjang SMP.
- **Non-DO (negatif)**: siswa yang menyelesaikan SMP (kini berada di jenjang lanjutan).
- Variabel `tingkat_pendidikan` **dibuang** sebagai prediktor: karena DO diamati pada
  kelas SMP sedangkan Non-DO kini di kelas lebih tinggi, memasukkannya menimbulkan
  kebocoran/artefak sampling (model memisahkan kelas secara trivial).

## 2. Penanganan ketidakseimbangan kelas

Data latih bersifat **enriched (case–control)** — prevalensi DO pada data latih jauh
lebih tinggi daripada di populasi nyata. Pendekatan:

- **Tanpa SMOTE.** Literatur menunjukkan oversampling sintetis merusak kalibrasi
  (menggelembungkan probabilitas kelas minoritas). Sebagai gantinya digunakan
  `class_weight` / `scale_pos_weight`.
- **Kalibrasi probabilitas** (sigmoid/Platt atau isotonic) pada data tertahan.
- **Koreksi prior** dari prevalensi latih (`pi_train`) ke base rate populasi (`pi_pop`):
  `offset = logit(pi_pop) − logit(pi_train)` ditambahkan pada log-odds.
- **Threshold** diset pada base rate sehingga proporsi siswa ter-flag rasional.

## 3. Pemilihan fitur (hybrid voting)

Fitur dipertahankan bila lolos **≥2 dari 3** metode (dihitung di dalam fold CV agar tak bocor):
1. signifikansi regresi logistik (p < 0.05),
2. importance Random Forest di atas median,
3. mean |SHAP| (XGBoost) di atas median.

## 4. Validasi (nested cross-validation)

- **Outer loop** GroupKFold (per sekolah) → estimasi performa tak-bias.
- **Inner loop** GroupKFold → feature selection + tuning hyperparameter
  (RandomizedSearch, metrik PR-AUC).
- Mencegah bias optimistik akibat memakai CV yang sama untuk tuning & seleksi.

## 5. Metrik

Karena data imbalanced, fokus pada **PR-AUC, recall, F1 kelas minoritas, dan kalibrasi
(Brier)** — bukan accuracy.

## 6. Temuan penting: validitas variabel asesmen

Pada data kami, skor **literasi membaca & sains** memiliki hubungan **terbalik** dengan DO
(siswa DO yang memiliki skor asesmen justru cenderung bernilai tinggi). Ini hampir pasti
**artefak cakupan sampel** — hanya sebagian siswa DO yang memiliki skor asesmen, dan subset
itu cenderung berprestasi. Model yang memakai ketiga skor akhirnya menandai *siswa
berprestasi* sebagai berisiko — tidak valid untuk intervensi.

**Solusi:** gunakan hanya skor **numerasi** (hubungan lemah & non-monotonik, tidak
mendominasi), sehingga model digerakkan oleh faktor **sosial-ekonomi** yang arah
pengaruhnya logis (pendidikan/penghasilan orang tua rendah → risiko naik). Hasil: daftar
siswa berisiko didominasi profil sosial-ekonomi rendah — actionable untuk program bantuan.

## 7. Deployment tiered

- **Model utama `aspd_num`** untuk siswa yang memiliki skor numerasi.
- **Model fallback `tanpa_aspd`** untuk siswa tanpa skor asesmen.
- Siswa dengan variabel **inti** tidak lengkap diberi label *Data Tidak Lengkap* dan tidak
  diprediksi (menghindari bias imputasi). Variabel mutu sekolah yang kosong boleh diimputasi
  median (level sekolah).

## Acuan literatur (ringkas)

- Hybrid statistical + ML + XAI untuk prediksi DO (feature selection voting; class weight, bukan SMOTE).
- Sistem peringatan dini berbasis XGBoost dengan variabel sosial-ekonomi/demografi; prioritas recall.
- Studi kalibrasi: oversampling sintetis merusak kalibrasi probabilitas.
- Nested cross-validation untuk estimasi performa tak-bias (menghindari bias seleksi+tuning).
- Koreksi prior untuk kejadian langka (rare-events) agar probabilitas sesuai base rate.
