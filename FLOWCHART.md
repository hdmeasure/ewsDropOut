# Flowchart Pengerjaan Machine Learning (XGBoost)

Berikut adalah alur logika (flowchart) dari proses pelatihan model Machine Learning untuk deteksi dini risiko putus sekolah (Early Warning System) menggunakan algoritma XGBoost.

```mermaid
graph TD
    A[Mulai: Persiapan Dataset] --> B(Input Data)
    B --> C{Dataset Siswa DO & Non-DO}
    
    C -->|Memuat Data Kategorikal & Numerik| D[Tahap Preprocessing]
    D --> E[Mapping Ordinal: Pendidikan & Penghasilan Orang Tua]
    D --> F[Numeric Casting: Memastikan Nilai/Skor berbentuk Angka]
    D --> G[Membuang Baris dengan Data Kosong / Drop NA]
    
    E --> H[Tahap Seleksi Fitur]
    F --> H
    G --> H
    
    H --> I[Pelatihan Model Baseline]
    I --> J{Evaluasi Feature Importance}
    J -->|Importance < 0.02| K[Hapus Variabel Tidak Signifikan]
    J -->|Importance >= 0.02| L[Simpan Variabel Signifikan]
    
    K --> L
    L --> M[Tahap Hyperparameter Tuning]
    M --> N[Mencari Parameter Optimal dengan RandomizedSearchCV]
    N --> O[Penanganan Ketidakseimbangan Data: scale_pos_weight]
    
    O --> P[Tahap Pelatihan Akhir]
    P --> Q[Melatih Model XGBoost dengan Parameter Terbaik & Fitur Tersaring]
    
    Q --> R[Tahap Evaluasi & Prediksi]
    R --> S[Menghasilkan Metrik: Presisi, Recall, F1-Score]
    R --> T[Prediksi Probabilitas Risiko DO]
    
    S --> U[Selesai: Model Siap Diimplementasikan]
    T --> U
```

## Penjelasan Singkat
1. **Input Data**: Dataset yang memuat informasi demografi dan berbagai nilai/skor akademik.
2. **Preprocessing**: Memastikan tipe data sudah benar. Variabel kategorikal yang memiliki tingkatan (seperti tingkat pendidikan) diubah menjadi angka yang berurutan. Nilai/skor dipastikan bertipe numerik.
3. **Seleksi Fitur**: Melatih model awal secara kasar untuk melihat variabel apa saja yang sebenarnya berpengaruh. Variabel yang sumbangsihnya terlalu kecil (importance < 0.02) dieliminasi agar model lebih efisien.
4. **Tuning & Imbalance Handling**: Menggunakan teknik *RandomizedSearchCV* untuk mencari kombinasi *hyperparameter* terbaik dari XGBoost. Karena jumlah siswa DO biasanya jauh lebih sedikit (minoritas) dibandingkan siswa Non-DO, diterapkan pembobotan kelas (*scale_pos_weight*) agar model tidak bias.
5. **Pelatihan Akhir & Evaluasi**: Model final dilatih menggunakan parameter terbaik dan fitur yang sudah terseleksi, kemudian dievaluasi kinerjanya.
