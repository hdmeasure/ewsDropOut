# Early Warning System (EWS) Student Drop Out Prediction

Repository ini berisi kode sumber (*source code*) untuk melatih model Machine Learning berbasis **XGBoost** guna memprediksi risiko siswa putus sekolah atau Drop Out (DO/LTM) di tingkat SMP. Model ini memanfaatkan kombinasi variabel personal, demografi/ekonomi orang tua, hasil akademis ujian nasional (ASPD), serta survei lingkungan sekolah (Sulingjar).

## 🚀 Fitur Utama
1.  **Preprocessing & Encoding**: Mengonversi variabel ordinal Dapodik dan Sulingjar secara otomatis ke bentuk angka standar.
2.  **Feature Selection**: Menyaring variabel-variabel yang tidak signifikan secara otomatis (importance score < 0.02) menggunakan baseline model sebelum *tuning*.
3.  **Hyperparameter Tuning**: Melakukan pencarian parameter optimal dengan `RandomizedSearchCV` menggunakan Cross-Validation.
4.  **Threshold Optimization**: Mencari *threshold* probabilistik seimbang (F1-score maksimum) untuk menyesuaikan ketidakseimbangan kelas.

---

## 📂 Struktur Repositori
- `train.py`: Skrip utama untuk preprocessing, seleksi fitur, tuning hyperparameter, evaluasi model, dan serialisasi model.
- `CODEBOOK.md`: Kamus data yang menjelaskan setiap kolom prediktor, tipe data, serta aturan kodifikasinya.

---

## 🛠️ Cara Memulai

### 1. Prasyarat & Instalasi
Instal pustaka Python yang diperlukan terlebih dahulu:
```bash
pip install pandas numpy xgboost scikit-learn
```

### 2. Mempersiapkan Data
Pastikan Anda menaruh file data Anda di direktori yang sama dengan nama file:
- `dataset_do.csv` (data siswa yang teridentifikasi Drop Out/LTM)
- `dataset_nondo.csv` (data siswa aktif/kontrol)

Kolom prediktor dan kunci pencocokan (seperti `NISN` dan `NPSN`) harus mengikuti pedoman yang tertera di [CODEBOOK.md](CODEBOOK.md).

### 3. Menjalankan Pelatihan Model
Jalankan perintah berikut pada terminal Anda untuk melatih model:
```bash
python3 train.py
```
Skrip akan mencetak proses pembersihan data, variabel yang dibuang, hasil parameter terbaik, metrik evaluasi test set (Precision, Recall, F1, AUC), dan menyimpan model akhir sebagai `model_ews_do.pkl`.
