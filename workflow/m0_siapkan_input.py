#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAHAP 0-1 — Siapkan input model.

- Gabung DO (SMP-only, label=1) + NonDO (semua, label=0).
- Buang tingkat_pendidikan (artefak sampling).
- Join NPSN dari file ekstrak (untuk GroupKFold).
- Encoding jenis_kelamin.
- Hitung prevalensi latih per skenario.

Output: output/model_input_<scenario>.csv  (nisn, npsn, prediktor, status_do)

Jalankan:
    python3 m0_siapkan_input.py
"""
import os
import pandas as pd
import common as C


def build_npsn_map():
    """nisn(normalisasi) -> npsn, dari file ekstrak DO & NonDO."""
    def nz(v):
        if pd.isna(v):
            return ''
        t = str(v).strip()
        if t.endswith('.0'):
            t = t[:-2]
        return t.lstrip('0')

    mp = {}
    for fname in ['DATA_SISWA_DO_SMP_EKSTRAK.xlsx', 'DATA_SISWA_NON_DO_EKSTRAK.xlsx']:
        path = os.path.join(C.BASE, fname)
        if not os.path.exists(path):
            print(f'  [LEWAT] {fname} tidak ada')
            continue
        df = pd.read_excel(path, dtype=str)
        cols = {c.lower(): c for c in df.columns}
        if 'nisn' not in cols or 'npsn' not in cols:
            continue
        for _, row in df[[cols['nisn'], cols['npsn']]].iterrows():
            k = nz(row[cols['nisn']])
            npsn = str(row[cols['npsn']]).strip() if not pd.isna(row[cols['npsn']]) else ''
            if k and npsn and k not in mp:
                mp[k] = npsn
    return mp


def nz(v):
    if pd.isna(v):
        return ''
    t = str(v).strip()
    if t.endswith('.0'):
        t = t[:-2]
    return t.lstrip('0')


def main():
    import json
    npsn_map = build_npsn_map()
    print(f'NPSN map: {len(npsn_map)} nisn->npsn')

    cutpoints = None
    summary = []
    for scenario in C.SCENARIOS:
        df = C.load_training(scenario)

        # join npsn jika belum ada
        if 'npsn' not in df.columns:
            df['npsn'] = df['nisn'].map(lambda x: npsn_map.get(nz(x), ''))
        else:
            mask = df['npsn'].isna() | (df['npsn'].astype(str).str.strip() == '')
            df.loc[mask, 'npsn'] = df.loc[mask, 'nisn'].map(lambda x: npsn_map.get(nz(x), ''))

        # rekayasa fitur ASPD bila perlu (cutpoints dari NonDO, dipakai konsisten)
        extra = []
        if scenario in C.ENG_FEATURES:
            df, cuts = C.add_aspd_features(df, cutpoints)
            if cutpoints is None:
                cutpoints = cuts
                with open(os.path.join(C.OUT_DIR, 'aspd_cutpoints.json'), 'w') as fh:
                    json.dump(cutpoints, fh, indent=2)
            extra = ['pola_aspd'] if scenario == 'aspd_pola' else []

        preds = C.predictor_cols(scenario)
        keep = ['nisn', 'npsn'] + preds + extra + ['status_do']
        keep = [c for c in dict.fromkeys(keep) if c in df.columns]
        df = df[keep]

        out = os.path.join(C.OUT_DIR, f'model_input_{scenario}.csv')
        df.to_csv(out, index=False)

        n = len(df)
        n_do = int((df['status_do'] == 1).sum())
        n_nd = int((df['status_do'] == 0).sum())
        prev = n_do / n if n else 0
        no_npsn = int((df['npsn'].astype(str).str.strip() == '').sum())
        print(f'\n[{scenario}] -> {out}')
        print(f'  n={n} | DO={n_do} | NonDO={n_nd} | prevalensi DO={prev:.3%}')
        print(f'  prediktor ({len(preds)}): {preds}')
        print(f'  baris tanpa NPSN: {no_npsn}')
        summary.append((scenario, n, n_do, n_nd, prev, no_npsn))

    print('\n=== RINGKASAN ===')
    print(f'{"skenario":<14}{"n":>8}{"DO":>7}{"NonDO":>8}{"prev":>9}{"no_npsn":>9}')
    for s, n, do, nd, prev, nn in summary:
        print(f'{s:<14}{n:>8}{do:>7}{nd:>8}{prev:>8.2%}{nn:>9}')
    print('\nCatatan: prevalensi latih ini masih enriched; koreksi prior dilakukan di m2.')


if __name__ == '__main__':
    main()
