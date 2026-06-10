#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contoh skoring di Python — dua jalur setara:
  (A) dari bundle .joblib (paling ringkas)
  (B) dari booster.json + spec.json (portabel, tanpa objek Python kustom)

Jalankan:
    python3 predict.py aspd_num contoh_input.csv
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def score_from_spec(name, df, models_dir):
    """Jalur portabel: hanya butuh xgboost + booster.json + spec.json."""
    import xgboost as xgb
    spec = json.load(open(os.path.join(models_dir, f'{name}_spec.json')))
    booster = xgb.Booster()
    booster.load_model(os.path.join(models_dir, f'{name}_booster.json'))

    feats = spec['features']
    X = df[feats].apply(pd.to_numeric, errors='coerce').values
    p_raw = booster.predict(xgb.DMatrix(X, feature_names=feats))

    cal = spec['calibration']
    if cal['method'] == 'sigmoid':
        p_cal = _sigmoid(cal['a'] * _logit(p_raw) + cal['b'])
    else:
        p_cal = np.interp(p_raw, cal['x'], cal['y'])

    offset = (np.log(spec['pi_pop'] / (1 - spec['pi_pop']))
              - np.log(spec['pi_train'] / (1 - spec['pi_train'])))
    p_adj = _sigmoid(_logit(p_cal) + offset)
    lab = np.where(p_adj >= spec['threshold'], 'BERISIKO', 'Tidak')
    return pd.DataFrame({'prob_do': np.round(p_adj, 4), 'risiko_do': lab})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name', nargs='?', default='aspd_num')
    ap.add_argument('input_csv', nargs='?', default=None)
    ap.add_argument('--models-dir', default=os.path.join(HERE, 'models'))
    args = ap.parse_args()

    if args.input_csv and os.path.exists(args.input_csv):
        df = pd.read_csv(args.input_csv)
    else:
        spec = json.load(open(os.path.join(args.models_dir, f'{args.name}_spec.json')))
        df = pd.DataFrame([{f: 0 for f in spec['features']}])
        print('Tidak ada input CSV; contoh 1 baris dummy.')

    out = score_from_spec(args.name, df, args.models_dir)
    print(out.head(20).to_string(index=False))
    n_flag = int((out['risiko_do'] == 'BERISIKO').sum())
    print(f'\nTotal: {len(out)} siswa, berisiko: {n_flag} ({n_flag/len(out):.2%})')


if __name__ == '__main__':
    main()
