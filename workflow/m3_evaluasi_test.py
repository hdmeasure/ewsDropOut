#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAHAP 7 — Evaluasi pada hold-out test set (terkunci sejak m1).

Output (per skenario, di output/):
  - eval_test_<scenario>.txt          metrik lengkap
  - curve_roc_<scenario>.png
  - curve_pr_<scenario>.png
  - curve_calibration_<scenario>.png

Catatan: prevalensi test = enriched (bukan populasi). Diskriminasi (ROC/PR-AUC)
dan kalibrasi adalah fokus utama; cek proporsi rasional ada di m5 (implementasi).

Jalankan:
    python3 m3_evaluasi_test.py
"""
import os, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

import common as C
from scorer import ModelBundle

warnings.filterwarnings('ignore')


def run_scenario(scenario):
    print(f'\n{"="*70}\nEVALUASI TEST: {scenario}\n{"="*70}')
    te_path = os.path.join(C.OUT_DIR, f'split_{scenario}_test.csv')
    mdl_path = os.path.join(C.OUT_DIR, f'model_{scenario}.joblib')
    if not (os.path.exists(te_path) and os.path.exists(mdl_path)):
        print('[LEWAT] split test / model belum ada — jalankan m1 & m2 dulu.')
        return

    bundle = ModelBundle.load(mdl_path)
    df = pd.read_csv(te_path, dtype={'nisn': str, 'npsn': str})
    feats = bundle.features
    df = df.dropna(subset=feats + ['status_do']).reset_index(drop=True)
    X = df[feats].astype(float).values
    y = df['status_do'].astype(int).values

    p_raw = bundle.proba_raw(X)
    p_cal = bundle.proba_cal(X)
    p_adj = bundle.proba_adj(X)

    m = C.metrics_block(y, p_adj, threshold=bundle.threshold)
    from sklearn.metrics import brier_score_loss
    brier_cal = brier_score_loss(y, p_cal)

    flagged = float((p_adj >= bundle.threshold).mean())

    lines = [
        f'scenario        : {scenario}',
        f'family          : {bundle.family}',
        f'features        : {feats}',
        f'n_test          : {len(y)}  (DO={int(y.sum())}, prevalensi={y.mean():.3%})',
        f'threshold       : {bundle.threshold:.4f}',
        f'pi_train->pi_pop: {bundle.pi_train:.3%} -> {bundle.pi_pop:.3%}',
        '',
        '--- Diskriminasi (invariant thd kalibrasi/koreksi) ---',
        f'ROC-AUC         : {m["roc_auc"]:.4f}',
        f'PR-AUC          : {m["pr_auc"]:.4f}',
        '',
        f'--- Pada threshold {bundle.threshold:.4f} ---',
        f'recall (DO)     : {m["recall_do"]:.4f}',
        f'precision (DO)  : {m["precision_do"]:.4f}',
        f'F1 (DO)         : {m["f1_do"]:.4f}',
        f'flagged %       : {flagged:.3%}',
        f'confusion       : TN={m["tn"]} FP={m["fp"]} FN={m["fn"]} TP={m["tp"]}',
        '',
        '--- Kalibrasi ---',
        f'Brier (cal)     : {brier_cal:.4f}',
        f'Brier (adj)     : {m["brier"]:.4f}',
    ]
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(C.OUT_DIR, f'eval_test_{scenario}.txt'), 'w') as fh:
        fh.write(txt + '\n')

    # ---- kurva ROC ----
    fpr, tpr, _ = roc_curve(y, p_cal)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f'ROC (AUC={m["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC — {scenario}')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(C.OUT_DIR, f'curve_roc_{scenario}.png'), dpi=120)
    plt.close()

    # ---- kurva PR ----
    prec, rec, _ = precision_recall_curve(y, p_cal)
    plt.figure(figsize=(5, 5))
    plt.plot(rec, prec, label=f'PR (AP={m["pr_auc"]:.3f})')
    plt.axhline(y.mean(), ls='--', color='gray', label=f'baseline={y.mean():.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR — {scenario}')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(C.OUT_DIR, f'curve_pr_{scenario}.png'), dpi=120)
    plt.close()

    # ---- kurva kalibrasi ----
    try:
        frac_pos, mean_pred = calibration_curve(y, p_cal, n_bins=10, strategy='quantile')
        plt.figure(figsize=(5, 5))
        plt.plot(mean_pred, frac_pos, 'o-', label='model (cal)')
        plt.plot([0, 1], [0, 1], '--', color='gray', label='ideal')
        plt.xlabel('Prob prediksi'); plt.ylabel('Fraksi positif observasi')
        plt.title(f'Kalibrasi — {scenario}')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(C.OUT_DIR, f'curve_calibration_{scenario}.png'), dpi=120)
        plt.close()
    except Exception as e:
        print(f'[kalibrasi plot dilewati] {e}')

    print(f'>> Plot & metrik tersimpan di {C.OUT_DIR}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=list(C.SCENARIOS) + ['both', 'all'], default='both')
    args = ap.parse_args()
    scenarios = C.expand_scenarios(args.scenario)
    for sc in scenarios:
        run_scenario(sc)


if __name__ == '__main__':
    main()
