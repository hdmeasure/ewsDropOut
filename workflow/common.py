#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitas bersama untuk seluruh workflow pemodelan DO.
Berisi: definisi kolom, loader, encoding, dan konstanta global.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

# === File sumber dataset model ===
DATASET_DO = os.path.join(BASE, 'datasetDO.xlsx')
DATASET_NONDO = os.path.join(BASE, 'datasetNonDO_SMP.xlsx')
DATASET_IMPL = os.path.join(BASE, 'dataset_implementasi_SMP.xlsx')

# === Folder output ===
OUT_DIR = os.path.join(HERE, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# === Skenario === (key -> nama sheet sumber)
# Skenario rekayasa fitur ASPD memakai sheet 'dengan_ASPD' (punya read/num/sci).
SCENARIOS = {
    'dengan_aspd': 'dengan_ASPD',   # ASPD mentah (read, num, sci)
    'tanpa_aspd': 'tanpa_ASPD',     # tanpa ASPD
    'aspd_mean': 'dengan_ASPD',     # rerata 3 skor
    'aspd_dev': 'dengan_ASPD',      # rerata + deviasi (kerataan skor)
    'aspd_pola': 'dengan_ASPD',     # pola level per mapel (R/S/T)
    'aspd_num': 'dengan_ASPD',      # hanya numerasi (num)
}
PRIMARY = ['dengan_aspd', 'tanpa_aspd']


def expand_scenarios(arg):
    if arg == 'both':
        return list(PRIMARY)
    if arg == 'all':
        return list(SCENARIOS)
    return [arg]

# === Prediktor kandidat ===
# tingkat_pendidikan DIBUANG (artefak sampling: DO di tk 7-9, NonDO kini di tk 10-11)
ASPD_COLS = ['read', 'num', 'sci']
SULINGJAR_COLS = ['sulingjar_D.18', 'sulingjar_D.1', 'sulingjar_D.2', 'sulingjar_D.3',
                  'sulingjar_D.4', 'sulingjar_D.6', 'sulingjar_D.8', 'sulingjar_D.10',
                  'sulingjar_E.1', 'sulingjar_E.5']
SES_COLS = ['kode_pendidikan_ayah', 'kode_pendidikan_ibu',
            'kode_penghasilan_ayah', 'kode_penghasilan_ibu']
BASE_COLS = ['jk_bin']  # jenis_kelamin terenkode
ASPD_RAW = ['read', 'num', 'sci']

# fitur ASPD hasil rekayasa per skenario
ENG_FEATURES = {
    'aspd_mean': ['aspd_mean'],
    'aspd_dev':  ['aspd_mean', 'aspd_std'],
    'aspd_pola': ['lvl_read', 'lvl_num', 'lvl_sci'],
}
# skenario yang memakai subset ASPD MENTAH (tanpa rekayasa)
RAW_SUBSET = {
    'aspd_num': ['num'],
}


def predictor_cols(scenario):
    """Daftar prediktor kandidat untuk skenario tertentu."""
    if scenario == 'dengan_aspd':
        return list(BASE_COLS) + list(ASPD_COLS) + list(SES_COLS) + list(SULINGJAR_COLS)
    if scenario in RAW_SUBSET:
        return list(BASE_COLS) + list(RAW_SUBSET[scenario]) + list(SES_COLS) + list(SULINGJAR_COLS)
    if scenario in ENG_FEATURES:
        return list(BASE_COLS) + list(ENG_FEATURES[scenario]) + list(SES_COLS) + list(SULINGJAR_COLS)
    return list(BASE_COLS) + list(SES_COLS) + list(SULINGJAR_COLS)


def needs_aspd_raw(scenario):
    """True bila skenario perlu read/num/sci (mentah atau utk rekayasa)."""
    return scenario == 'dengan_aspd' or scenario in ENG_FEATURES or scenario in RAW_SUBSET


def compute_aspd_cutpoints(df):
    """Tercile (q33,q66) per mapel dari kelompok NonDO (acuan = lulusan 'normal')."""
    import pandas as pd
    ref = df[df['status_do'] == 0] if 'status_do' in df.columns else df
    cut = {}
    for c in ASPD_RAW:
        v = pd.to_numeric(ref[c], errors='coerce').dropna()
        if len(v):
            cut[c] = [float(v.quantile(1/3)), float(v.quantile(2/3))]
        else:
            cut[c] = [0.0, 0.0]
    return cut


def add_aspd_features(df, cutpoints=None):
    """Tambah aspd_mean, aspd_std, aspd_range, lvl_*, pola_aspd dari read/num/sci.
    cutpoints: dict {mapel:[q33,q66]} acuan level. Jika None, dihitung dari NonDO df."""
    import numpy as np, pandas as pd
    df = df.copy()
    cols = []
    for c in ASPD_RAW:
        cols.append(pd.to_numeric(df[c], errors='coerce').values if c in df.columns
                    else np.full(len(df), np.nan))
    arr = np.vstack(cols).T  # n x 3
    with np.errstate(invalid='ignore'):
        df['aspd_mean'] = np.nanmean(arr, axis=1)
        df['aspd_std'] = np.nanstd(arr, axis=1)            # deviasi/kerataan skor
        df['aspd_range'] = np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1)

    if cutpoints is None:
        cutpoints = compute_aspd_cutpoints(df)

    abbr = {1: 'R', 2: 'S', 3: 'T'}

    def lvl(x, cp):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return 1 if x < cp[0] else (2 if x < cp[1] else 3)

    name = {'read': 'lvl_read', 'num': 'lvl_num', 'sci': 'lvl_sci'}
    for c in ASPD_RAW:
        cp = cutpoints[c]
        df[name[c]] = [lvl(x, cp) for x in pd.to_numeric(df.get(c), errors='coerce')]

    pola = []
    for r in df[['lvl_read', 'lvl_num', 'lvl_sci']].values:
        if any(isinstance(v, float) and np.isnan(v) for v in r):
            pola.append('')
        else:
            pola.append('-'.join(abbr[int(v)] for v in r))
    df['pola_aspd'] = pola
    return df, cutpoints

# === Kolom identitas (bukan prediktor) ===
ID_COLS = ['nisn', 'npsn']

RANDOM_STATE = 42


def encode_frame(df):
    """Encoding konsisten untuk semua tahap.
    - jenis_kelamin -> jk_bin (L=1, P=0)
    - kebutuhan_khusus_ya_tidak -> kk_bin (Ya=1, Tidak=0)  [opsional, tidak dipakai default]
    Ordinal pendidikan/penghasilan/sulingjar sudah numerik.
    """
    import numpy as np
    df = df.copy()
    if 'jenis_kelamin' in df.columns:
        df['jk_bin'] = df['jenis_kelamin'].astype(str).str.strip().str.upper().map(
            {'L': 1, 'P': 0})
    if 'kebutuhan_khusus_ya_tidak' in df.columns:
        df['kk_bin'] = df['kebutuhan_khusus_ya_tidak'].astype(str).str.strip().str.lower().map(
            {'ya': 1, 'tidak': 0})
    # paksa kolom numerik
    for c in ASPD_COLS + SES_COLS + SULINGJAR_COLS:
        if c in df.columns:
            df[c] = __import__('pandas').to_numeric(df[c], errors='coerce')
    return df


def load_training(scenario):
    """Gabung DO (SMP-only, label 1) + NonDO (semua, label 0) untuk satu skenario.
    Mengembalikan DataFrame berisi: prediktor + status_do + nisn + npsn.
    """
    import pandas as pd
    sheet = SCENARIOS[scenario]

    do = pd.read_excel(DATASET_DO, sheet_name=sheet, dtype={'nisn': str})
    nd = pd.read_excel(DATASET_NONDO, sheet_name=sheet, dtype={'nisn': str})

    # DO: hanya SMP (jenjang == 'SMP')
    do = do[do['jenjang'].astype(str).str.upper() == 'SMP'].copy()
    do['status_do'] = 1

    # NonDO: gunakan SELURUH baris (mereka lulusan SMP, kini kelas 11)
    nd = nd.copy()
    nd['status_do'] = 0

    df = pd.concat([do, nd], ignore_index=True, sort=False)
    df = encode_frame(df)

    # Kolom dasar yang selalu disertakan. Fitur ASPD rekayasa dibuat di m0
    # (butuh read/num/sci mentah), jadi sertakan ASPD_RAW bila skenario perlu.
    keep = list(BASE_COLS) + list(SES_COLS) + list(SULINGJAR_COLS) + ['status_do', 'nisn']
    if needs_aspd_raw(scenario):
        keep += ASPD_RAW
    if 'npsn' in df.columns:
        keep.append('npsn')
    keep = [c for c in dict.fromkeys(keep) if c in df.columns]
    return df[keep]


def metrics_block(y_true, y_prob, threshold=0.5):
    """Hitung metrik utama untuk data imbalanced."""
    import numpy as np
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  precision_score, recall_score, f1_score,
                                  brier_score_loss, confusion_matrix)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    out = {}
    try: out['roc_auc'] = roc_auc_score(y_true, y_prob)
    except Exception: out['roc_auc'] = float('nan')
    try: out['pr_auc'] = average_precision_score(y_true, y_prob)
    except Exception: out['pr_auc'] = float('nan')
    out['recall_do'] = recall_score(y_true, y_pred, zero_division=0)
    out['precision_do'] = precision_score(y_true, y_pred, zero_division=0)
    out['f1_do'] = f1_score(y_true, y_pred, zero_division=0)
    try: out['brier'] = brier_score_loss(y_true, y_prob)
    except Exception: out['brier'] = float('nan')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
    return out
