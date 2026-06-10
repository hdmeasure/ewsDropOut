#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAHAP 9 (TIERED) — Implementasi gabungan dua model + alasan risiko per siswa.

Default: model UTAMA = aspd_num, FALLBACK = tanpa_aspd.

Logika pemilihan model per siswa:
  - Jika fitur INTI model UTAMA lengkap (mis. 'num' utk aspd_num) -> pakai model utama.
  - Jika tidak, tapi fitur INTI FALLBACK lengkap                  -> pakai fallback.
  - Jika keduanya tidak                                           -> "Data Tidak Lengkap".
  (Sulingjar yang kosong selalu boleh diimputasi median.)

Threshold dihitung PER GRUP model sehingga proporsi ter-flag = pi pada masing-masing
subpopulasi -> total ter-flag ≈ pi (proporsi rasional).

Alasan risiko: 3 faktor pendorong terbesar (SHAP positif) per siswa.

Output: output/prediksi_tiered.xlsx  (sheet 'prediksi', 'ringkasan', 'per_kecamatan')

Jalankan:
    python3 m6_implementasi_tiered.py --pi 0.025
    python3 m6_implementasi_tiered.py --primary aspd_num --fallback tanpa_aspd --pi 0.025
"""
import os, argparse, warnings
import numpy as np
import pandas as pd

import common as C
from scorer import ModelBundle

warnings.filterwarnings('ignore')

ID_OUT = ['nama_siswa', 'nisn', 'nik', 'jenis_kelamin', 'npsn', 'nama_sekolah',
          'kecamatan', 'tingkat_pendidikan']

# label manusiawi untuk alasan (TANPA kata arah; arah dihitung dari median)
LABELS = {
    'jk_bin': 'jenis kelamin',
    'read': 'nilai ASPD literasi membaca',
    'num': 'nilai ASPD numerasi',
    'sci': 'nilai ASPD sains',
    'kode_pendidikan_ayah': 'pendidikan ayah',
    'kode_pendidikan_ibu': 'pendidikan ibu',
    'kode_penghasilan_ayah': 'penghasilan ayah',
    'kode_penghasilan_ibu': 'penghasilan ibu',
}
SUL_LABEL = {
    'sulingjar_D.18': 'mutu sekolah: kesiapsiagaan bencana',
    'sulingjar_D.1': 'mutu sekolah: kualitas pembelajaran',
    'sulingjar_D.2': 'mutu sekolah: refleksi guru',
    'sulingjar_D.3': 'mutu sekolah: kepemimpinan kepsek',
    'sulingjar_D.4': 'mutu sekolah: iklim keamanan',
    'sulingjar_D.6': 'mutu sekolah: iklim kesetaraan gender',
    'sulingjar_D.8': 'mutu sekolah: iklim kebinekaan',
    'sulingjar_D.10': 'mutu sekolah: iklim inklusivitas',
    'sulingjar_E.1': 'mutu sekolah: partisipasi warga',
    'sulingjar_E.5': 'mutu sekolah: program satuan pendidikan',
}
LABELS.update(SUL_LABEL)


def label_of(feat):
    return LABELS.get(feat, feat)


def phrase(feat, val, med):
    """Frasa alasan dgn arah relatif median training."""
    if feat == 'jk_bin':
        return 'jenis kelamin laki-laki' if val >= 0.5 else 'jenis kelamin perempuan'
    ref = med.get(feat)
    arah = ''
    if ref is not None and not np.isnan(ref):
        if val < ref:
            arah = ' rendah'
        elif val > ref:
            arah = ' tinggi'
    try:
        vtxt = f'{val:g}'
    except Exception:
        vtxt = str(val)
    return f'{label_of(feat)}{arah} (={vtxt})'


def median_map(scenario, feats):
    med = {}
    p = os.path.join(C.OUT_DIR, f'split_{scenario}_train.csv')
    if os.path.exists(p):
        tr = pd.read_csv(p)
        for f in feats:
            if f in tr.columns:
                med[f] = pd.to_numeric(tr[f], errors='coerce').median()
    for f in feats:
        med.setdefault(f, 0.0)
    return med


def shap_pos_matrix(estimator, X, family):
    """Matriks SHAP kelas positif (n x fitur)."""
    import shap
    if family in ('xgb', 'rf'):
        expl = shap.TreeExplainer(estimator)
        sv = expl.shap_values(X)
        sv = np.array(sv)
        if sv.ndim == 3:
            if sv.shape[0] == 2:
                sv = sv[1]
            elif sv.shape[-1] == 2:
                sv = sv[:, :, 1]
            else:
                sv = sv[..., -1]
        return sv
    # logreg pipeline
    scaler = estimator.named_steps['scaler']
    clf = estimator.named_steps['clf']
    Xs = scaler.transform(X)
    expl = shap.LinearExplainer(clf, Xs)
    return np.array(expl.shap_values(Xs))


def reasons_for_group(bundle, Xv, med, top_k=3):
    """Kembalikan list string alasan untuk tiap baris di Xv (sudah diimputasi)."""
    feats = bundle.features
    try:
        sv = shap_pos_matrix(bundle.estimator, Xv, bundle.family)
    except Exception as e:
        print(f'   [SHAP alasan dilewati: {e}]')
        return ['' for _ in range(len(Xv))]

    out = []
    for i in range(sv.shape[0]):
        row = sv[i]
        order = np.argsort(row)[::-1]  # SHAP positif terbesar dulu
        parts = []
        for j in order[:top_k]:
            if row[j] <= 0:
                break
            parts.append(phrase(feats[j], Xv[i, j], med))
        out.append('; '.join(parts) if parts else 'tidak ada faktor risiko dominan')
    return out


def score_group(df, idx, bundle, med, pi, max_missing, scen_label):
    """Skor subset df.loc[idx] dgn satu model. Kembalikan dict hasil per index global."""
    feats = bundle.features
    sub = df.loc[idx]
    Xf = sub[feats].apply(pd.to_numeric, errors='coerce')
    Ximp = Xf.fillna(value=med)
    Xv = Ximp.values
    p_adj = bundle.proba_adj(Xv)

    if pi > 0 and len(p_adj) > 0:
        thr = float(np.quantile(p_adj, 1 - pi))
    else:
        thr = bundle.threshold
    flag = p_adj >= thr

    reasons = reasons_for_group(bundle, Xv, med)

    res = pd.DataFrame(index=sub.index)
    res['prob_do'] = np.round(p_adj, 4)
    res['risiko_do'] = np.where(flag, 'BERISIKO', 'Tidak')
    res['alasan_risiko'] = reasons
    res['model_dipakai'] = f'{bundle.family} ({scen_label})'
    return res, thr, int(flag.sum())


def core_missing(df, feats, max_missing):
    """Boolean: True jika fitur INTI (non-sulingjar) kosong melebihi toleransi."""
    core = [f for f in feats if not str(f).startswith('sulingjar')]
    Xf = df[core].apply(pd.to_numeric, errors='coerce')
    nmiss = Xf.isna().sum(axis=1).values
    return nmiss > max_missing, nmiss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--primary', default='aspd_num', help='model utama (default aspd_num)')
    ap.add_argument('--fallback', default='tanpa_aspd', help='model cadangan (default tanpa_aspd)')
    ap.add_argument('--pi', type=float, default=0.025, help='base rate target (default 0.025)')
    ap.add_argument('--max-missing', type=int, default=0,
                    help='maks. fitur inti kosong agar tetap diprediksi (default 0)')
    args = ap.parse_args()
    pi = args.pi
    max_missing = args.max_missing
    prim, fb = args.primary, args.fallback

    b_prim = ModelBundle.load(os.path.join(C.OUT_DIR, f'model_{prim}.joblib'))
    b_fb = ModelBundle.load(os.path.join(C.OUT_DIR, f'model_{fb}.joblib'))
    med_prim = median_map(prim, b_prim.features)
    med_fb = median_map(fb, b_fb.features)

    df = pd.read_excel(C.DATASET_IMPL, dtype={'nisn': str, 'nik': str, 'npsn': str})
    df = C.encode_frame(df)

    # bangun fitur ASPD rekayasa bila salah satu model memerlukannya
    allfeats = set(b_prim.features) | set(b_fb.features)
    if any(str(f).startswith(('aspd_', 'lvl_')) for f in allfeats):
        import json
        cp_path = os.path.join(C.OUT_DIR, 'aspd_cutpoints.json')
        cutpoints = json.load(open(cp_path)) if os.path.exists(cp_path) else None
        df, _ = C.add_aspd_features(df, cutpoints)

    for f in allfeats:
        if f not in df.columns:
            df[f] = np.nan
    df = df.reset_index(drop=True)
    n = len(df)

    miss_prim, _ = core_missing(df, b_prim.features, max_missing)
    miss_fb, _ = core_missing(df, b_fb.features, max_missing)

    use_prim = ~miss_prim
    use_fb = (~use_prim) & (~miss_fb)
    none_idx = df.index[~use_prim & ~use_fb]

    print(f'Total siswa: {n}')
    print(f'  -> {prim} (utama)   : {int(use_prim.sum())}')
    print(f'  -> {fb} (fallback) : {int(use_fb.sum())}')
    print(f'  -> Data Tidak Lengkap: {len(none_idx)}')

    parts = []
    thr_info = {}
    if use_prim.sum() > 0:
        r, thr, nf = score_group(df, df.index[use_prim], b_prim, med_prim, pi, max_missing, prim)
        parts.append(r); thr_info[prim] = thr
    if use_fb.sum() > 0:
        r, thr, nf = score_group(df, df.index[use_fb], b_fb, med_fb, pi, max_missing, fb)
        parts.append(r); thr_info[fb] = thr

    scored = pd.concat(parts) if parts else pd.DataFrame()
    # gabung kembali ke df
    df['prob_do'] = np.nan
    df['risiko_do'] = 'Data Tidak Lengkap'
    df['alasan_risiko'] = ''
    df['model_dipakai'] = 'Data Tidak Lengkap'
    if len(scored):
        df.loc[scored.index, ['prob_do', 'risiko_do', 'alasan_risiko', 'model_dipakai']] = \
            scored[['prob_do', 'risiko_do', 'alasan_risiko', 'model_dipakai']].values

    # kosongkan alasan utk yang tidak berisiko (fokus pada yang BERISIKO)
    df.loc[df['risiko_do'] != 'BERISIKO', 'alasan_risiko'] = \
        df.loc[df['risiko_do'] != 'BERISIKO', 'alasan_risiko'].where(
            df['risiko_do'] == 'Data Tidak Lengkap', '')

    out_cols = [c for c in ID_OUT if c in df.columns] + \
               ['model_dipakai', 'prob_do', 'risiko_do', 'alasan_risiko']
    out_cols = list(dict.fromkeys([c for c in out_cols if c in df.columns]))
    res = df[out_cols].copy().sort_values('prob_do', ascending=False, na_position='last')

    n_pred = int((df['risiko_do'] != 'Data Tidak Lengkap').sum())
    n_flag = int((df['risiko_do'] == 'BERISIKO').sum())
    summary = pd.DataFrame({
        'metrik': ['model_utama', 'model_fallback', 'pi target', 'max_inti_kosong',
                   'n_siswa_total', 'n_diprediksi',
                   f'n_pakai_{prim}', f'n_pakai_{fb}', 'n_data_tidak_lengkap',
                   'n_berisiko', 'persen_berisiko_dr_diprediksi', 'persen_berisiko_dr_total',
                   f'threshold_{prim}', f'threshold_{fb}'],
        'nilai': [prim, fb, f'{pi:.3%}', max_missing, n, n_pred,
                  int(use_prim.sum()), int(use_fb.sum()), len(none_idx),
                  n_flag,
                  f'{n_flag/n_pred:.3%}' if n_pred else '-',
                  f'{n_flag/n:.3%}',
                  round(thr_info.get(prim, float("nan")), 4),
                  round(thr_info.get(fb, float("nan")), 4)]
    })

    out = os.path.join(C.OUT_DIR, 'prediksi_tiered.xlsx')
    with pd.ExcelWriter(out, engine='openpyxl') as xw:
        res.to_excel(xw, sheet_name='prediksi', index=False)
        summary.to_excel(xw, sheet_name='ringkasan', index=False)
        if 'kecamatan' in df.columns:
            per_kec = df.groupby('kecamatan').agg(
                n=('risiko_do', 'size'),
                n_diprediksi=('risiko_do', lambda s: int((s != 'Data Tidak Lengkap').sum())),
                n_berisiko=('risiko_do', lambda s: int((s == 'BERISIKO').sum()))).reset_index()
            per_kec['persen_dr_diprediksi'] = (
                per_kec['n_berisiko'] / per_kec['n_diprediksi'].replace(0, np.nan) * 100).round(1)
            per_kec.to_excel(xw, sheet_name='per_kecamatan', index=False)

    print('\n' + summary.to_string(index=False))
    print(f'\n>> SANITY: {n_flag} dari {n_pred} siswa diprediksi ter-flag berisiko '
          f'({n_flag/n_pred:.2%}); target {pi:.2%}.')
    print(f'>> Tersimpan: {out}')


if __name__ == '__main__':
    main()
