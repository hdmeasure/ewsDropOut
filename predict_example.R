# Contoh Penggunaan Model EWS Drop Out (.rds) di Bahasa R

library(xgboost)

# 1. Muat model EWS (.rds) yang telah dikonversi
model_path <- "model_ews_do.rds"

if (!file.exists(model_path)) {
  stop("File model 'model_ews_do.rds' tidak ditemukan. Silakan jalankan 'python3 convert_to_rds.py' terlebih dahulu.")
}

model_ews <- readRDS(model_path)

# 2. Siapkan data baru untuk diprediksi
# CATATAN PENTING: 
# - Data input harus memiliki jumlah kolom dan nama kolom yang sama persis saat pelatihan.
# - Urutan kolom juga harus disesuaikan dengan fitur signifikan yang digunakan model.
# - Skrip pelatihan (train.py) akan menyaring fitur tidak signifikan secara otomatis.
#   Pastikan Anda menyesuaikan kolom input di bawah ini sesuai dengan fitur final yang tersaring.

# Contoh data simulasi untuk 2 siswa:
data_baru <- matrix(c(
  75.5, 80.0, 7, 7, 3, 2, 2.5, 2.8, 2.4, 2.0, 3.0, 2.7, 2.9, 2.5, 2.8, 3.0,  # Siswa A
  50.0, 45.0, 2, 2, 1, 1, 1.2, 1.5, 1.0, 1.1, 1.5, 1.2, 1.3, 1.0, 1.1, 1.2   # Siswa B
), nrow = 2, byrow = TRUE)

# Berikan nama kolom (sesuaikan dengan nama kolom fitur model Anda)
colnames(data_baru) <- c(
  "score_read", "score_sci", "jenjang_pendidikan_ayah", "jenjang_pendidikan_ibu",
  "penghasilan_ayah", "penghasilan_ibu", "sulingjar_D.1", "sulingjar_D.10",
  "sulingjar_D.18", "sulingjar_D.2", "sulingjar_D.3", "sulingjar_D.4",
  "sulingjar_D.6", "sulingjar_D.8", "sulingjar_E.1", "sulingjar_E.5"
)

# 3. Melakukan prediksi probabilitas risiko DO (skala 0 s.d 1)
prediksi_prob <- predict(model_ews, newdata = data_baru)

# 4. Tampilkan hasil prediksi
for (i in 1:length(prediksi_prob)) {
  cat(sprintf("Siswa ke-%d: Risiko Drop Out = %.2f%%\n", i, prediksi_prob[i] * 100))
}
