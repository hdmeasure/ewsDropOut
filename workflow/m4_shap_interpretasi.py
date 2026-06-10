#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAHAP 8 — Interpretasi model dengan SHAP.

Output (per skenario, di output/):
  - shap_summary_<scenario>.png      (ringkasan global, arah & besaran)
  - shap_importance_<scenario>.csv   (ranking mean|SHAP|)

Memverifikasi arah pengaruh prediktor (logis/tidak).
Bekerja untuk family tree (rf/xgb) via TreeExplainer; logreg via koefisien.

Jalankan:
    python3 m4_shap_interpretasi.py
"""
import os, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import common as C
from scorer import ModelBundle

warnings.filterwarnings('ignore')


def run_scenario(scenario):
    print(f'\n{"="*70}\nSHAP: {scenario}\n{"="*70}')
    tr_path = os.path.join(C.OUT_DIR, f'split_{scenario}_train.csv')
    mdl_path = os.path.join(C.OUT_DIR, f'model_{scenario}.joblib')
    if not (os.path.exists(tr_path) and os.path.exists(mdl_path)):
        print('[LEWAT] split train / model belum ada.')
        return

    bundle = ModelBundle.load(mdl_path)
    feats = bundle.features
    df = pd.read_csv(tr_path, dtype={'nisn': str, 'npsn': str})
    df = df.dropna(subset=feats).reset_index(drop=True)
    X = df[feats].astype(float)
    # subsample untuk kecepatan jika besar
    if len(X) > 4000:
        X = X.sample(4000, random_state=C.RANDOM_STATE).reset_index(drop=True)

    est = bundle.estimator
    fam = bundle.family

    try:
        import shap
        if fam in ('rf', 'xgb'):
            model = est
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X.values)
            if isinstance(sv, list):
                sv = sv[-1]
        else:
            # logreg pipeline: jelaskan via linear explainer pada data terstandardisasi
            scaler = est.named_steps['scaler']
            clf = est.named_steps['clf']
            Xs = scaler.transform(X.values)
            explainer = shap.LinearExplainer(clf, Xs)
            sv = explainer.shap_values(Xs)

        mean_abs = np.abs(sv).mean(axis=0)
        imp = pd.DataFrame({'feature': feats, 'mean_abs_shap': mean_abs}) \
            .sort_values('mean_abs_shap', ascending=False)
        imp.to_csv(os.path.join(C.OUT_DIR, f'shap_importance_{scenario}.csv'), index=False)
        print(imp.to_string(index=False))

        plt.figure()
        shap.summary_plot(sv, X, show=False, max_display=len(feats))
        plt.title(f'SHAP summary — {scenario}')
        plt.tight_layout()
        plt.savefig(os.path.join(C.OUT_DIR, f'shap_summary_{scenario}.png'), dpi=120,
                    bbox_inches='tight')
        plt.close()
        print(f'>> SHAP tersimpan di {C.OUT_DIR}')

    except Exception as e:
        print(f'[SHAP gagal] {e} — fallback importance sederhana.')
        if fam in ('rf',):
            imp_vals = est.feature_importances_
        elif fam == 'xgb':
            imp_vals = est.feature_importances_
        else:
            imp_vals = np.abs(est.named_steps['clf'].coef_[0])
        imp = pd.DataFrame({'feature': feats, 'importance': imp_vals}) \
            .sort_values('importance', ascending=False)
        imp.to_csv(os.path.join(C.OUT_DIR, f'shap_importance_{scenario}.csv'), index=False)
        print(imp.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=list(C.SCENARIOS) + ['both', 'all'], default='both')
    args = ap.parse_args()
    scenarios = C.expand_scenarios(args.scenario)
    for sc in scenarios:
        run_scenario(sc)


if __name__ == '__main__':
    main()
