# Codebook: Dataset Prediksi Siswa Drop Out (DO)

Dokumen ini menjelaskan struktur data, definisi variabel, dan aturan kodifikasi yang diterapkan pada dataset pemodelan Early Warning System (EWS) Drop Out:
1. **`dataset_do.csv`** (Kelompok Siswa Drop Out / Lulus Tidak Melanjutkan)
2. **`dataset_nondo.csv`** (Kelompok Siswa Aktif / Sukses Sekolah)

---

## 📌 Identitas Data & Kunci Pemadan
*   **`NISN`** (String): Nomor Induk Siswa Nasional (unik per siswa).
*   **`NPSN`** (String): Nomor Pokok Sekolah Nasional (asal sekolah siswa).

---

## 👥 Variabel Prediktor Siswa (Kodifikasi & Skala Ordinal/Numerik)

### 1. Karakteristik Personal Siswa
*   **`jenis_kelamin`** (Kategorikal - Biner):
    *   `1`: Laki-laki (`L`)
    *   `0`: Perempuan (`P`)
*   **`kebutuhan_khusus_siswa`** (Kategorikal - Biner):
    *   `0`: Tidak memiliki kebutuhan khusus
    *   `1`: Memiliki kebutuhan khusus (disabilitas/kesulitan belajar)

### 2. Status Ekonomi & Pendapatan Orang Tua
*   **`status_ekonomi`** (Kategorikal - Biner):
    *   `1`: Prasejahtera / Layak Bantuan Sosial (Rentan Miskin)
    *   `0`: Sejahtera / Tidak Layak Bantuan (Mampu)
*   **`penghasilan_ayah` / `penghasilan_ibu`** (Kategorikal - Ordinal):
    *   `0`: Tidak Berpenghasilan / Kurang dari Rp 500,000
    *   `1`: Rp 500,000 - Rp 999,999
    *   `2`: Rp 1,000,000 - Rp 1,999,999
    *   `3`: Rp 2,000,000 - Rp 4,999,999
    *   `4`: Rp 5,000,000 - Rp 20,000,000
    *   `5`: Lebih dari Rp 20,000,000

### 3. Profil Pendidikan Orang Tua
*   **`jenjang_pendidikan_ayah` / `jenjang_pendidikan_ibu`** (Kategorikal - Ordinal):
    *   `0`: Tidak Sekolah / PAUD / TK / Putus SD
    *   `1`: SD / Sederajat
    *   `2`: SMP / Sederajat
    *   `3`: SMA / Sederajat
    *   `4`: D1
    *   `5`: D2
    *   `6`: D3 / Diploma
    *   `7`: D4 / S1 / Sarjana
    *   `8`: S2
    *   `9`: S3

### 4. Aksesibilitas Fisik ke Sekolah
*   **`jarak_rumah_ke_sekolah`** (Kategorikal - Ordinal):
    *   `1`: Kurang dari 1 km
    *   `2`: Lebih dari 1 km
*   **`waktu_tempuh_ke_sekolah`** (Numerik): Waktu tempuh perjalanan ke sekolah dalam satuan menit.

---

## 📈 Variabel Akademis Siswa (Ujian Standar Nasional)
*   **`score_read`** (Numerik): Nilai Literasi Membaca (Skala 0 - 100)
*   **`score_num`** (Numerik): Nilai Numerasi / Matematika (Skala 0 - 100)
*   **`score_sci`** (Numerik): Nilai Sains / IPA (Skala 0 - 100)

---

## 🏫 Variabel Kualitas Sekolah (Survei Lingkungan Belajar - Sulingjar)
*   **`sulingjar_D.18`**: Kesiapsiagaan Bencana Sekolah
*   **`sulingjar_D.1`**: Kualitas Pembelajaran Kelas
*   **`sulingjar_D.2`**: Refleksi dan Perbaikan Pembelajaran Guru
*   **`sulingjar_D.3`**: Kepemimpinan Instruksional Kepala Sekolah
*   **`sulingjar_D.4`**: Indeks Iklim Keamanan Sekolah
*   **`sulingjar_D.6`**: Indeks Iklim Kesetaraan Gender di Sekolah
*   **`sulingjar_D.8`**: Indeks Iklim Kebinekaan Sekolah
*   **`sulingjar_D.10`**: Indeks Iklim Inklusivitas Sekolah
*   **`sulingjar_E.1`**: Indeks Partisipasi Warga Sekolah
*   **`sulingjar_E.5`**: Program dan Kebijakan Satuan Pendidikan

*(Setiap variabel Sulingjar dipetakan secara ordinal dari: `1`: Kurang, `2`: Sedang, `3`: Baik)*
