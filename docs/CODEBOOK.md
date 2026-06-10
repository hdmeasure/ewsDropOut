# Kamus Variabel (Codebook)

Definisi variabel yang dipakai pipeline. Nama kolom = nama teknis di kode.
Data mentah tidak disertakan; ini hanya spesifikasi struktur.

## Identitas (bukan prediktor)
| Kolom | Tipe | Keterangan |
|---|---|---|
| `nisn` | teks | ID siswa (disimpan teks, pertahankan leading zero) |
| `npsn` | teks | ID sekolah (dipakai untuk GroupKFold per sekolah) |

## Target
| Kolom | Nilai | Keterangan |
|---|---|---|
| `status_do` | 1 / 0 | 1 = DO (putus sekolah saat SMP); 0 = menyelesaikan SMP |

## Prediktor
| Kolom | Tipe | Skala / kode |
|---|---|---|
| `jk_bin` | biner | 1 = laki-laki, 0 = perempuan |
| `num` | numerik | skor numerasi asesmen standar (0–100) |
| `kode_pendidikan_ayah` / `_ibu` | ordinal | 0=tidak sekolah … 8=S3 |
| `kode_penghasilan_ayah` / `_ibu` | ordinal | 0=tidak berpenghasilan … 6=tertinggi |
| `sulingjar_*` | ordinal | indikator mutu sekolah: 1=Kurang, 2=Sedang, 3=Baik |

### Kode pendidikan orang tua
| Kode | Jenjang |
|---|---|
| 0 | Tidak sekolah |
| 1 | Putus SD |
| 2 | SD / sederajat |
| 3 | SMP / sederajat |
| 4 | SMA / sederajat |
| 5 | Diploma (D1–D3) |
| 6 | D4 / S1 |
| 7 | S2 |
| 8 | S3 |

### Kode penghasilan orang tua (per bulan, ordinal)
| Kode | Rentang |
|---|---|
| 0 | Tidak berpenghasilan |
| 1 | < 500 ribu |
| 2 | 500 rb – < 1 jt |
| 3 | 1 jt – < 2 jt |
| 4 | 2 jt – < 5 jt |
| 5 | 5 jt – 20 jt |
| 6 | > 20 jt |

### Indikator mutu sekolah (`sulingjar_*`)
Sumber: survei lingkungan belajar (level sekolah). Nilai teks `Kurang/Sedang/Baik`
dikonversi menjadi `1/2/3`. Indikator yang dipakai antara lain: kesiapsiagaan bencana,
kualitas pembelajaran, refleksi guru, kepemimpinan kepala sekolah, iklim keamanan,
iklim kesetaraan gender, iklim kebinekaan, iklim inklusivitas, partisipasi warga,
program satuan pendidikan.

## Fitur turunan asesmen (opsional, dipakai pada eksperimen)
| Kolom | Keterangan |
|---|---|
| `aspd_mean` | rerata skor asesmen |
| `aspd_std` | deviasi antar-skor (kerataan) |
| `lvl_read/num/sci` | level R/S/T per mata pelajaran (tercile acuan Non-DO) |

> Model final hanya memakai `num` dari kelompok asesmen (lihat METHODOLOGY §6).

## Output prediksi
| Kolom | Keterangan |
|---|---|
| `prob_do` | probabilitas DO (terkalibrasi + terkoreksi prior) |
| `risiko_do` | `BERISIKO` / `Tidak` / `Data Tidak Lengkap` |
| `alasan_risiko` | faktor pendorong utama (SHAP) per siswa berisiko |
| `model_dipakai` | `aspd_num` atau `tanpa_aspd` |
