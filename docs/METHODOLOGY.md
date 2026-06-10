# Metodologi

Ringkasan pendekatan pemodelan EWS Drop Out, beserta dasar literatur dan keputusan desain.
Setiap pemilihan metode/tahapan disertai sitasi (APA 7); daftar lengkap di bagian
[Referensi](#referensi).

## 1. Definisi populasi & label

- **DO (positif)**: siswa yang putus sekolah saat jenjang SMP.
- **Non-DO (negatif)**: siswa yang menyelesaikan SMP (kini berada di jenjang lanjutan).
- Variabel `tingkat_pendidikan` **dibuang** sebagai prediktor: karena DO diamati pada
  kelas SMP sedangkan Non-DO kini di kelas lebih tinggi, memasukkannya menimbulkan
  kebocoran/artefak sampling (model memisahkan kelas secara trivial).

## 2. Penanganan ketidakseimbangan kelas

Data latih bersifat **enriched (case–control)** — prevalensi DO pada data latih jauh
lebih tinggi daripada di populasi nyata.

- **Tanpa SMOTE.** Meskipun oversampling sintetis seperti SMOTE (Chawla et al., 2002)
  lazim dipakai pada prediksi DO (Villar & de Andrade, 2024), simulasi menunjukkan
  koreksi ketidakseimbangan **merusak kalibrasi** model risiko — menggelembungkan
  probabilitas kelas minoritas tanpa memperbaiki diskriminasi (van den Goorbergh et al.,
  2022). Karena itu digunakan pembobotan kelas (`class_weight` / `scale_pos_weight`),
  sejalan dengan praktik pada sistem prediksi DO berbasis pohon (Khatun et al., 2025;
  Carballo-Mendívil et al., 2025).
- **Kalibrasi probabilitas** (sigmoid/Platt atau isotonic) pada data tertahan, untuk
  menjaga probabilitas dapat ditafsirkan sebagai risiko (van den Goorbergh et al., 2022).
- **Koreksi prior** dari prevalensi latih (`pi_train`) ke base rate populasi (`pi_pop`):
  `offset = logit(pi_pop) − logit(pi_train)` ditambahkan pada log-odds — pendekatan untuk
  data kejadian langka (King & Zeng, 2001).
- **Threshold** diset pada base rate sehingga proporsi siswa ter-flag rasional.

## 3. Pemilihan fitur (hybrid voting)

Fitur dipertahankan bila lolos **≥2 dari 3** metode, mengikuti kerangka voting pada
prediksi DO (Khatun et al., 2025): (1) signifikansi regresi logistik (p < 0.05),
(2) importance Random Forest (Breiman, 2001) di atas median, dan (3) mean |SHAP|
(Lundberg & Lee, 2017) dari model XGBoost di atas median. Seluruhnya dihitung **di dalam
fold CV** agar tidak terjadi kebocoran (Cawley & Talbot, 2010).

## 4. Validasi (nested cross-validation)

- **Outer loop** GroupKFold (per sekolah) → estimasi performa tak-bias.
- **Inner loop** GroupKFold → feature selection + tuning hyperparameter
  (RandomizedSearch, metrik PR-AUC).
- Pemisahan ini mencegah bias optimistik akibat memakai CV yang sama untuk tuning dan
  seleksi model (Cawley & Talbot, 2010). Pembagian **per sekolah** mencegah kebocoran
  karena variabel mutu sekolah bersifat level-sekolah.

## 5. Algoritma & metrik

- Model: **XGBoost** (Chen & Guestrin, 2016) dan **Random Forest** (Breiman, 2001), yang
  konsisten unggul pada prediksi DO (Villar & de Andrade, 2024; Khatun et al., 2025).
- Karena data imbalanced, fokus pada **PR-AUC, recall, F1 kelas minoritas, dan kalibrasi
  (Brier)** — bukan accuracy; prioritas pada recall sejalan dengan tujuan peringatan dini
  (Carballo-Mendívil et al., 2025).
- Interpretasi tiap prediksi memakai **SHAP** (Lundberg & Lee, 2017).

## 6. Temuan penting: validitas variabel asesmen

Pada data kami, skor **literasi membaca & sains** memiliki hubungan **terbalik** dengan DO
(siswa DO yang memiliki skor asesmen justru cenderung bernilai tinggi). Ini hampir pasti
**artefak cakupan sampel** — hanya sebagian siswa DO yang memiliki skor asesmen, dan subset
itu cenderung berprestasi. Model yang memakai ketiga skor akhirnya menandai *siswa
berprestasi* sebagai berisiko — tidak valid untuk intervensi.

**Solusi:** gunakan hanya skor **numerasi** (hubungan lemah & non-monotonik, tidak
mendominasi), sehingga model digerakkan oleh faktor **sosial-ekonomi** yang arah
pengaruhnya logis (pendidikan/penghasilan orang tua rendah → risiko naik), konsisten
dengan prediktor DO yang dilaporkan literatur (Khatun et al., 2025;
Carballo-Mendívil et al., 2025). Hasil: daftar siswa berisiko didominasi profil
sosial-ekonomi rendah — *actionable* untuk program bantuan.

## 7. Deployment tiered

- **Model utama `aspd_num`** untuk siswa yang memiliki skor numerasi.
- **Model fallback `tanpa_aspd`** untuk siswa tanpa skor asesmen.
- Siswa dengan variabel **inti** tidak lengkap diberi label *Data Tidak Lengkap* dan tidak
  diprediksi (menghindari bias imputasi). Variabel mutu sekolah yang kosong boleh diimputasi
  median (level sekolah).

---

## Referensi

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
https://doi.org/10.1023/A:1010933404324

Carballo-Mendívil, B., Arellano-González, A., Ríos-Vázquez, N. J., & Lizardi-Duarte, M. del P.
(2025). Predicting student dropout from day one: XGBoost-based early warning system using
pre-enrollment data. *Applied Sciences, 15*(16), 9202. https://doi.org/10.3390/app15169202

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent
selection bias in performance evaluation. *Journal of Machine Learning Research, 11*,
2079–2107.

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic
minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*,
321–357. https://doi.org/10.1613/jair.953

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings
of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*
(pp. 785–794). Association for Computing Machinery. https://doi.org/10.1145/2939672.2939785

Khatun, M. R., Mim, M. A., Tasin, M. M., & Hossain, M. M. (2025). A hybrid framework of
statistical, machine learning, and explainable AI methods for school dropout prediction.
*PLOS ONE, 20*(9), e0331917. https://doi.org/10.1371/journal.pone.0331917

King, G., & Zeng, L. (2001). Logistic regression in rare events data. *Political Analysis,
9*(2), 137–163. https://doi.org/10.1093/oxfordjournals.pan.a004868

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions.
In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 4765–4774). Curran
Associates.

van den Goorbergh, R., van Smeden, M., Timmerman, D., & Van Calster, B. (2022). The harm of
class imbalance corrections for risk prediction models: Illustration and simulation using
logistic regression. *Journal of the American Medical Informatics Association, 29*(9),
1525–1534. https://doi.org/10.1093/jamia/ocac093

Villar, A., & de Andrade, C. R. V. (2024). Supervised machine learning algorithms for
predicting student dropout and academic success: A comparative study. *Discover Artificial
Intelligence, 4*, 2. https://doi.org/10.1007/s44163-023-00079-z
