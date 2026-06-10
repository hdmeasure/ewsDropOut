import pickle
import os
import subprocess

# Define filenames
pkl_file = 'model_ews_do.pkl'
json_file = 'model_temp.json'
rds_output = 'model_ews_do.rds'

def main():
    print(f"1. Memuat model pickle Python dari: {pkl_file}")
    if not os.path.exists(pkl_file):
        print(f"Error: File model '{pkl_file}' tidak ditemukan.")
        print("Pastikan Anda sudah menjalankan 'python3 train.py' terlebih dahulu untuk menghasilkan file model.")
        return

    with open(pkl_file, 'rb') as f:
        model_data = pickle.load(f)

    # Mengekstrak model XGBoost dari dictionary pkl
    if isinstance(model_data, dict) and 'model' in model_data:
        xgb_model = model_data['model']
        # Ekstrak threshold dan fitur untuk informasi tambahan
        threshold = model_data.get('balanced_threshold', 0.5)
        features = model_data.get('features', [])
        print(f"   Model memuat {len(features)} fitur dengan optimal threshold: {threshold:.4f}")
    else:
        xgb_model = model_data

    print(f"2. Mengekspor model XGBoost ke format JSON: {json_file}")
    # Simpan booster ke file JSON universal
    xgb_model.save_model(json_file)

    print("3. Membuat skrip R sementara untuk konversi...")
    r_script_content = f"""
library(xgboost)

# Memuat model dari JSON
model_r <- xgb.load("{json_file}")

# Menyimpan ke RDS
saveRDS(model_r, "{rds_output}")
message("Berhasil menyimpan model R ke: {rds_output}")
"""

    r_script_file = 'convert_temp.R'
    with open(r_script_file, 'w') as f:
        f.write(r_script_content)

    print("4. Menjalankan Rscript untuk membuat file .rds...")
    try:
        # Menjalankan perintah Rscript di sistem
        result = subprocess.run(['Rscript', r_script_file], capture_output=True, text=True, check=True)
        print(result.stdout)
        print(f"Konversi SELESAI! File disimpan sebagai: '{rds_output}'")
    except FileNotFoundError:
        print("Error: R atau Rscript tidak ditemukan di sistem ini.")
        print("Silakan jalankan perintah R secara manual untuk mengonversi model JSON:")
        print(f"  library(xgboost)")
        print(f"  model_r <- xgb.load('{json_file}')")
        print(f"  saveRDS(model_r, '{rds_output}')")
    except subprocess.CalledProcessError as e:
        print("Gagal menjalankan konversi melalui Rscript:")
        print(e.stderr)
    finally:
        # Bersihkan file perantara
        if os.path.exists(json_file):
            os.remove(json_file)
        if os.path.exists(r_script_file):
            os.remove(r_script_file)

if __name__ == '__main__':
    main()
