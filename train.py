import os
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 1. KONFIGURASI FILE & PARAMETER
# =====================================================================
# Nama file dataset input (disesuaikan dengan penamaan umum)
DO_DATASET_FILE = 'dataset_do.csv'
NONDO_DATASET_FILE = 'dataset_nondo.csv'
MODEL_OUTPUT_FILE = 'model_ews_do.pkl'

# Threshold minimum untuk signifikansi fitur (importance score di baseline model)
SIGNIFICANCE_THRESHOLD = 0.02

# =====================================================================
# 2. FUNGSI PREPROCESSING & KODIFIKASI
# =====================================================================
def map_education(val):
    """Memetakan jenjang pendidikan orang tua ke skala ordinal (0-9)"""
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if s in ['tidak sekolah', 'putus sd', '0', '0.0', '1', '1.0', '99', '99.0']: return 0
    if s in ['sd / sederajat', '2', '2.0']: return 1
    if s in ['smp / sederajat', '3', '3.0', '5', '5.0']: return 2
    if s in ['sma / sederajat', '4', '4.0', '6', '6.0']: return 3
    if s in ['d1', '20', '20.0']: return 4
    if s in ['d2', '21', '21.0']: return 5
    if s in ['d3', '22', '22.0']: return 6
    if s in ['d4', 's1', '23', '23.0', '30', '30.0']: return 7
    if s in ['s2', '35', '35.0']: return 8
    if s in ['s3', '40', '40.0']: return 9
    if s.isdigit() and int(s) <= 9: return int(s)
    return np.nan

def map_penghasilan(val):
    """Memetakan kategori penghasilan orang tua ke skala ordinal (0-5)"""
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if s in ['tidak berpenghasilan', 'kurang dari rp. 500,000', '1', '1.0', '99', '99.0']: return 0
    if s in ['rp. 500,000 - rp. 999,999', '2', '2.0']: return 1
    if s in ['rp. 1,000,000 - rp. 1,999,999', '3', '3.0']: return 2
    if s in ['rp. 2,000,000 - rp. 4,999,999', '4', '4.0']: return 3
    if s in ['rp. 5,000,000 - rp. 20,000,000', '5', '5.0']: return 4
    if s in ['lebih dari rp. 20,000,000', '6', '6.0']: return 5
    if s.isdigit() and int(s) <= 5: return int(s)
    return np.nan

def map_sulingjar(val):
    """Memetakan kategori capaian sulingjar ke skala ordinal (0-3)"""
    if pd.isna(val): return np.nan
    s = str(val).strip().title()
    if s == 'Kurang': return 1
    if s == 'Sedang': return 2
    if s == 'Baik': return 3
    if s == 'Capaian Tidak Tersedia': return 0
    if s.isdigit(): return int(s)
    return np.nan

def preprocess_data(df, features, sulingjar_cols):
    """Melakukan standardisasi tipe data dan kodifikasi ordinal"""
    df_copy = df.copy()
    
    # Preprocessing variabel sulingjar
    for col in sulingjar_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(map_sulingjar)
        else:
            df_copy[col] = np.nan
            
    # Preprocessing pendidikan orang tua
    for col in ['jenjang_pendidikan_ayah', 'jenjang_pendidikan_ibu']:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(map_education)
            
    # Preprocessing penghasilan orang tua
    for col in ['penghasilan_ayah', 'penghasilan_ibu']:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(map_penghasilan)
            
    # Pastikan tipe data numerik untuk semua prediktor
    for f in features:
        if f in df_copy.columns:
            df_copy[f] = pd.to_numeric(df_copy[f], errors='coerce')
            
    return df_copy

# =====================================================================
# 3. ALUR UTAMA PELATIHAN MODEL
# =====================================================================
def main():
    print("="*60)
    print("TRAINING & TUNING MODEL EWS DROP OUT (XGBOOST)")
    print("="*60)
    
    # 3.1 Load Dataset
    if not os.path.exists(DO_DATASET_FILE) or not os.path.exists(NONDO_DATASET_FILE):
        print(f"Error: Harap pastikan '{DO_DATASET_FILE}' dan '{NONDO_DATASET_FILE}' berada di direktori ini.")
        return
        
    print("Memuat dataset DO dan Non-DO...")
    df_do = pd.read_csv(DO_DATASET_FILE, low_memory=False)
    df_do['target_do_ltm'] = 1
    
    df_nondo = pd.read_csv(NONDO_DATASET_FILE, low_memory=False)
    df_nondo['target_do_ltm'] = 0
    
    # Selaraskan kolom irisan
    common_cols = list(set(df_do.columns).intersection(df_nondo.columns))
    df_combined = pd.concat([df_do[common_cols], df_nondo[common_cols]], ignore_index=True)
    
    # Tentukan variabel awal
    sulingjar_cols = sorted([c for c in df_combined.columns if 'sulingjar' in c])
    initial_features = [
        'score_read', 'score_num', 'score_sci', 
        'jenjang_pendidikan_ayah', 'jenjang_pendidikan_ibu', 
        'penghasilan_ayah', 'penghasilan_ibu'
    ] + sulingjar_cols
    
    # 3.2 Preprocessing & Pembersihan Baris Kosong
    print("Melakukan preprocessing dan kodifikasi data...")
    df_prep = preprocess_data(df_combined, initial_features, sulingjar_cols)
    df_clean = df_prep.dropna(subset=initial_features)
    
    X = df_clean[initial_features]
    y = df_clean['target_do_ltm']
    
    print(f"Ukuran data training bersih: {len(df_clean)} siswa (DO: {sum(y==1)}, Non-DO: {sum(y==0)})")
    
    # Split Train & Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3.3 Penanganan Ketidakseimbangan & Seleksi Fitur Baseline
    scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()
    
    print("\nMengevaluasi signifikansi variabel menggunakan baseline model...")
    baseline_xgb = xgb.XGBClassifier(
        scale_pos_weight=scale_pos,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    baseline_xgb.fit(X_train, y_train)
    
    # Filter fitur berdasarkan threshold signifikansi
    importances = baseline_xgb.feature_importances_
    significant_features = []
    for feat, imp in zip(initial_features, importances):
        if imp >= SIGNIFICANCE_THRESHOLD:
            significant_features.append(feat)
            print(f"  [SIGNIFIKAN] {feat:<25} : {imp:.4f}")
        else:
            print(f"  [TIDAK SIGNIFIKAN] {feat:<25} : {imp:.4f} (DIBUANG)")
            
    # Perbarui dataset dengan fitur tersaring
    X_train_sel = X_train[significant_features]
    X_test_sel = X_test[significant_features]
    
    # 3.4 Hyperparameter Tuning dengan Cross-Validation
    print("\nMelakukan tuning hyperparameter menggunakan RandomizedSearchCV...")
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    xgb_tuned = xgb.XGBClassifier(
        scale_pos_weight=scale_pos,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    random_search = RandomizedSearchCV(
        estimator=xgb_tuned,
        param_distributions=param_grid,
        n_iter=15,
        scoring='f1',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train_sel, y_train)
    best_model = random_search.best_estimator_
    
    print(f"\nHyperparameter terbaik ditemukan: {random_search.best_params_}")
    
    # 3.5 Evaluasi Performa Model & Pencarian Threshold Optimal
    y_prob_test = best_model.predict_proba(X_test_sel)[:, 1]
    
    # 1. Cari threshold seimbang (Statistik F1 Score Maksimum)
    best_t_f1 = 0.5
    max_f1 = 0
    best_p_f1 = 0
    best_r_f1 = 0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (y_prob_test >= t).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > max_f1:
            max_f1 = f1
            best_t_f1 = t
            best_p_f1 = precision_score(y_test, preds)
            best_r_f1 = recall_score(y_test, preds)
            
    print("\n=== PERFORMA MODEL DI TEST SET ===")
    print(f"ROC AUC Score                  : {roc_auc_score(y_test, y_prob_test):.4f}")
    print(f"Optimal Threshold (F1 Max)     : {best_t_f1:.4f}")
    print(f"Precision di Threshold F1 Max  : {best_p_f1:.4f}")
    print(f"Recall di Threshold F1 Max     : {best_r_f1:.4f}")
    print(f"F1-Score di Threshold F1 Max   : {max_f1:.4f}")
    
    # 3.6 Menyimpan Model Akhir
    medians = X_train_sel.median()
    print(f"\nMenyimpan model ke '{MODEL_OUTPUT_FILE}'...")
    with open(MODEL_OUTPUT_FILE, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'features': significant_features,
            'balanced_threshold': best_t_f1,
            'medians': medians
        }, f)
    print("Model berhasil disimpan.")
    
if __name__ == '__main__':
    main()
