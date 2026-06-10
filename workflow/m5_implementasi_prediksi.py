#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAHAP 9 — Implementasi: skor seluruh siswa SMP (dataset implementasi) dengan
model final, beri label risiko, dan cek kerasionalan proporsi.

- Missing value diimputasi dengan median fitur dari data train (transparan via
  kolom 'n_fitur_kosong'); tidak ada na.omit.
- Probabilitas = terkalibrasi + terkoreksi prior (p_adj).
- Label risiko: dua mode threshold —
    'baserate' (default): threshold dihitung pada populasi implementasi sehingga
        fraksi ter-flag = pi_pop (proporsi rasional by construction, sesuai
        keputusan "threshold sesuai base rate populasi"). Robust thd distribution
        shift & imputasi.
    'model': pakai threshold bawaan model (dari train); bisa tidak match proporsi
        pada populasi baru.

Output:
  output/prediksi_implementasi_<scenario>.xlsx  (sheet 'prediksi' + 'ringkasan')

Jalankan:
    python3 m5_implementasi_prediksi.py --scenario tanpa_aspd
    python3 m5_implementasi_prediksi.py --scenario tanpa_aspd --threshold-mode model
"""
import os, argparse, warnings
import numpy as np
import pandas as pd

import common as C
from scorer import ModelBundle

warnings.filterwarnings('ignore')

ID_OUT = ['nama_siswa', 'nisn', 'nik', 'jenis_kelamin', 'npsn', 'nama_sekolah',
          'kecamatan', 'tingkat_pendidikan']


def run_scenario(scenario, threshold_mode, max_missing, pi_override):
    print(f'\n{"="*70}\nIMPLEMENTASI: {scenario}  (threshold-mode={threshold_mode}, '
          f'max_fitur_kosong={max_missing})\n{"="*70}')
    mdl_path = os.path.join(C.OUT_DIR, f'model_{scenario}.joblib')
    tr_path = os.path.join(C.OUT_DIR, f'split_{scenario}_train.csv')
    if not (os.path.exists(mdl_path) and os.path.exists(C.DATASET_IMPL)):
        print('[LEWAT] model atau dataset implementasi belum ada.')
        return

    bundle = ModelBundle.load(mdl_path)
    feats = bundle.features

    # median imputasi dari train
    med = {}
    if os.path.exists(tr_path):
        tr = pd.read_csv(tr_path)
        for f in feats:
            if f in tr.columns:
                med[f] = pd.to_numeric(tr[f], errors='coerce').median()
    for f in feats:
        med.setdefault(f, 0.0)

    df = pd.read_excel(C.DATASET_IMPL, dtype={'nisn': str, 'nik': str, 'npsn': str})
    df = C.encode_frame(df)

    # skenario rekayasa ASPD: bangun fitur turunan dari read/num/sci (pakai cutpoints m0)
    if any(str(f).startswith(('aspd_', 'lvl_')) for f in feats):
        import json
        cp_path = os.path.join(C.OUT_DIR, 'aspd_cutpoints.json')
        cutpoints = json.load(open(cp_path)) if os.path.exists(cp_path) else None
        df, _ = C.add_aspd_features(df, cutpoints)

    # pastikan semua fitur ada
    for f in feats:
        if f not in df.columns:
            df[f] = np.nan
    Xf = df[feats].apply(pd.to_numeric, errors='coerce')

    # pisahkan Sulingjar (boleh diimputasi) vs fitur INTI (tidak boleh)
    sul_feats = [f for f in feats if str(f).startswith('sulingjar')]
    core_feats = [f for f in feats if f not in sul_feats]

    n_miss_total = Xf.isna().sum(axis=1).values
    n_miss_core = (Xf[core_feats].isna().sum(axis=1).values
                   if core_feats else np.zeros(len(df), dtype=int))
    n_miss_sul = (Xf[sul_feats].isna().sum(axis=1).values
                  if sul_feats else np.zeros(len(df), dtype=int))

    # Layak diprediksi bila kekosongan FITUR INTI <= max_missing.
    # Sulingjar yang kosong tetap boleh (akan diimputasi median).
    eligible = n_miss_core <= max_missing

    pi = pi_override if pi_override is not None else bundle.pi_pop

    # imputasi median: mengisi Sulingjar yang kosong (+ sedikit inti yang ditoleransi)
    Ximp = Xf.fillna(value=med)
    p_adj_all = bundle.proba_adj(Ximp.values)

    # threshold dihitung HANYA pada baris eligible (proporsi rasional di antara
    # siswa yang benar-benar diprediksi)
    p_elig = p_adj_all[eligible]
    if threshold_mode == 'baserate' and pi > 0 and len(p_elig) > 0:
        thr = float(np.quantile(p_elig, 1 - pi))
    else:
        thr = bundle.threshold

    # susun kolom hasil
    prob_do = np.full(len(df), np.nan)
    prob_do[eligible] = np.round(p_adj_all[eligible], 4)

    flag = eligible & (p_adj_all >= thr)
    risiko = np.array(['Data Tidak Lengkap'] * len(df), dtype=object)
    risiko[eligible & (p_adj_all >= thr)] = 'BERISIKO'
    risiko[eligible & (p_adj_all < thr)] = 'Tidak'

    df['prob_do'] = prob_do
    df['risiko_do'] = risiko
    df['n_inti_kosong'] = n_miss_core
    df['n_sulingjar_kosong'] = n_miss_sul
    df['n_fitur_kosong'] = n_miss_total

    # kolom output
    out_cols = [c for c in ID_OUT if c in df.columns] + \
               feats + ['n_inti_kosong', 'n_sulingjar_kosong', 'n_fitur_kosong',
                        'prob_do', 'risiko_do']
    out_cols = list(dict.fromkeys([c for c in out_cols if c in df.columns]))
    res = df[out_cols].copy().sort_values('prob_do', ascending=False, na_position='last')

    # ---- ringkasan kerasionalan ----
    n = len(df)
    n_elig = int(eligible.sum())
    n_excl = n - n_elig
    n_flag = int(flag.sum())
    pct_elig = n_flag / n_elig if n_elig else 0
    pct_tot = n_flag / n if n else 0
    summary = pd.DataFrame({
        'metrik': ['scenario', 'family', 'threshold_mode', 'max_inti_kosong',
                   'threshold_dipakai', 'pi target (di antara diprediksi)',
                   'n_siswa_total', 'n_diprediksi', 'n_data_tidak_lengkap',
                   'n_berisiko', 'persen_berisiko_dr_diprediksi',
                   'persen_berisiko_dr_total'],
        'nilai': [scenario, bundle.family, threshold_mode, max_missing,
                  round(thr, 4), f'{pi:.3%}',
                  n, n_elig, n_excl,
                  n_flag, f'{pct_elig:.3%}', f'{pct_tot:.3%}']
    })

    out = os.path.join(C.OUT_DIR, f'prediksi_implementasi_{scenario}.xlsx')
    with pd.ExcelWriter(out, engine='openpyxl') as xw:
        res.to_excel(xw, sheet_name='prediksi', index=False)
        summary.to_excel(xw, sheet_name='ringkasan', index=False)
        # ringkasan per kecamatan jika ada
        if 'kecamatan' in df.columns:
            per_kec = df.groupby('kecamatan').agg(
                n=('risiko_do', 'size'),
                n_diprediksi=('risiko_do', lambda s: int((s != 'Data Tidak Lengkap').sum())),
                n_berisiko=('risiko_do', lambda s: int((s == 'BERISIKO').sum()))).reset_index()
            per_kec['persen_dr_diprediksi'] = (
                per_kec['n_berisiko'] / per_kec['n_diprediksi'].replace(0, np.nan) * 100).round(1)
            per_kec.to_excel(xw, sheet_name='per_kecamatan', index=False)

    print(summary.to_string(index=False))
    print(f'\n>> {n_excl} siswa ({n_excl/n:.1%}) diberi label "Data Tidak Lengkap" '
          f'(fitur INTI kosong > {max_missing}; Sulingjar kosong tetap diimputasi) '
          f'dan TIDAK diprediksi.')
    print(f'>> SANITY CHECK: {n_flag} dari {n_elig} siswa yang diprediksi ter-flag '
          f'berisiko ({pct_elig:.2%}); target {pi:.2%}.')
    print(f'>> Tersimpan: {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=list(C.SCENARIOS) + ['both'], default='tanpa_aspd',
                    help='default tanpa_aspd (cakupan implementasi lebih luas)')
    ap.add_argument('--threshold-mode', choices=['baserate', 'model'], default='baserate',
                    help='baserate=fraksi ter-flag=pi pada siswa yang diprediksi (default)')
    ap.add_argument('--max-missing', type=int, default=0,
                    help='maks. fitur INTI (non-Sulingjar) kosong agar tetap diprediksi '
                         '(default 0 = inti harus lengkap). Sulingjar yang kosong selalu '
                         'diimputasi. Selebihnya diberi label "Data Tidak Lengkap".')
    ap.add_argument('--pi', type=float, default=None,
                    help='override base rate utk proporsi flag (mis. 0.025). '
                         'Default = pi_pop dari model (m2).')
    args = ap.parse_args()
    scenarios = C.expand_scenarios(args.scenario)
    for sc in scenarios:
        run_scenario(sc, args.threshold_mode, args.max_missing, args.pi)


if __name__ == '__main__':
    main()
