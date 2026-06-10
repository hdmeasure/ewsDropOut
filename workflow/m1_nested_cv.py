#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAHAP 3-4 — Nested cross-validation (GroupKFold per NPSN).

Untuk tiap family (logreg, rf, xgb):
  outer 5-fold GroupKFold -> estimasi performa tak-bias
  di tiap outer-train:
     (a) feature selection voting  (fs_voting.select_features)
     (b) hyperparameter tuning RandomizedSearchCV (inner GroupKFold, scoring=PR-AUC)
  prediksi outer-val -> metrik.

Output (per skenario, di output/):
  - split_<scenario>_train.csv / split_<scenario>_test.csv   (hold-out 20% sekolah)
  - cv_metrics_<scenario>.csv          (metrik tiap fold x family)
  - cv_summary_<scenario>.csv          (rata-rata per family)
  - feature_stability_<scenario>.csv   (frekuensi terpilih tiap fitur)
  - recommendation_<scenario>.txt      (family terbaik)

Jalankan:
    python3 m1_nested_cv.py                 # kedua skenario
    python3 m1_nested_cv.py --scenario tanpa_aspd --n-iter 25
"""
import os, argparse, warnings
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import (GroupShuffleSplit, GroupKFold,
                                      RandomizedSearchCV)
import common as C
import models as M
from fs_voting import select_features

warnings.filterwarnings('ignore')


def run_scenario(scenario, n_iter, n_outer, n_inner, seed):
    print(f'\n{"="*70}\nSKENARIO: {scenario}\n{"="*70}')
    path = os.path.join(C.OUT_DIR, f'model_input_{scenario}.csv')
    if not os.path.exists(path):
        print(f'[LEWAT] {path} belum ada — jalankan m0 dulu.')
        return
    df = pd.read_csv(path, dtype={'nisn': str, 'npsn': str})

    preds = [c for c in C.predictor_cols(scenario) if c in df.columns]
    # buang baris tanpa NPSN (perlu untuk GroupKFold) atau dengan prediktor NaN
    df = df.dropna(subset=preds + ['status_do']).copy()
    df = df[df['npsn'].astype(str).str.strip() != ''].copy()
    df = df.reset_index(drop=True)

    X = df[preds].astype(float)
    y = df['status_do'].astype(int)
    groups = df['npsn'].astype(str)
    print(f'n={len(df)} | DO={int(y.sum())} | prevalensi={y.mean():.3%} | '
          f'sekolah unik={groups.nunique()} | prediktor={len(preds)}')

    # ---- hold-out test 20% (per sekolah) ----
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    df.iloc[tr_idx].to_csv(os.path.join(C.OUT_DIR, f'split_{scenario}_train.csv'), index=False)
    df.iloc[te_idx].to_csv(os.path.join(C.OUT_DIR, f'split_{scenario}_test.csv'), index=False)
    print(f'Train={len(tr_idx)} | Test(hold-out)={len(te_idx)}')

    Xtr, ytr, gtr = X.iloc[tr_idx], y.iloc[tr_idx], groups.iloc[tr_idx]
    Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True); gtr = gtr.reset_index(drop=True)

    # ---- nested CV ----
    outer = GroupKFold(n_splits=n_outer)
    rows = []
    feat_freq = Counter()
    fold_count = 0

    for fold, (oi, ov) in enumerate(outer.split(Xtr, ytr, gtr), 1):
        Xo, yo, go = Xtr.iloc[oi], ytr.iloc[oi], gtr.iloc[oi]
        Xv, yv = Xtr.iloc[ov], ytr.iloc[ov]
        Xo = Xo.reset_index(drop=True); yo = yo.reset_index(drop=True); go = go.reset_index(drop=True)

        # (a) feature selection voting pada outer-train
        sel = select_features(Xo, yo, random_state=seed)
        for f in sel:
            feat_freq[f] += 1
        fold_count += 1
        print(f'\n[fold {fold}] fitur terpilih ({len(sel)}): {sel}')

        Xo_s = Xo[sel]
        Xv_s = Xv[sel]
        spw = M.pos_weight(yo)

        inner = GroupKFold(n_splits=n_inner)
        for fam in M.FAMILIES:
            est = M.make_estimator(fam, scale_pos_weight=spw, random_state=seed)
            search = RandomizedSearchCV(
                est, M.param_space(fam), n_iter=n_iter,
                scoring='average_precision', cv=inner, n_jobs=-1,
                random_state=seed, refit=True, error_score=0.0)
            search.fit(Xo_s.values, yo.values, groups=go.values)
            prob = search.best_estimator_.predict_proba(Xv_s.values)[:, 1]
            mt = C.metrics_block(yv.values, prob, threshold=0.5)
            mt.update({'scenario': scenario, 'fold': fold, 'family': fam,
                       'n_features': len(sel)})
            rows.append(mt)
            print(f'   {fam:7} PR-AUC={mt["pr_auc"]:.3f} ROC-AUC={mt["roc_auc"]:.3f} '
                  f'recall={mt["recall_do"]:.3f} F1={mt["f1_do"]:.3f}')

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(C.OUT_DIR, f'cv_metrics_{scenario}.csv'), index=False)

    summ = (metrics.groupby('family')[['pr_auc', 'roc_auc', 'recall_do',
                                       'precision_do', 'f1_do', 'brier']]
            .agg(['mean', 'std']))
    summ.columns = ['_'.join(c) for c in summ.columns]
    summ = summ.reset_index().sort_values('pr_auc_mean', ascending=False)
    summ.to_csv(os.path.join(C.OUT_DIR, f'cv_summary_{scenario}.csv'), index=False)

    stab = pd.DataFrame({'feature': list(C.predictor_cols(scenario))})
    stab = stab[stab['feature'].isin(preds)]
    stab['selected_in_folds'] = stab['feature'].map(lambda f: feat_freq.get(f, 0))
    stab['total_folds'] = fold_count
    stab['freq'] = stab['selected_in_folds'] / fold_count
    stab = stab.sort_values('freq', ascending=False)
    stab.to_csv(os.path.join(C.OUT_DIR, f'feature_stability_{scenario}.csv'), index=False)

    best = summ.iloc[0]['family']
    with open(os.path.join(C.OUT_DIR, f'recommendation_{scenario}.txt'), 'w') as fh:
        fh.write(f'BEST FAMILY (mean PR-AUC): {best}\n\n')
        fh.write('Ringkasan per family:\n')
        fh.write(summ.to_string(index=False))
        fh.write('\n\nStabilitas fitur (frekuensi terpilih):\n')
        fh.write(stab.to_string(index=False))

    print(f'\n=== RINGKASAN {scenario} ===')
    print(summ[['family', 'pr_auc_mean', 'roc_auc_mean', 'recall_do_mean', 'f1_do_mean']].to_string(index=False))
    print(f'\nFitur paling stabil:')
    print(stab[['feature', 'freq']].to_string(index=False))
    print(f'\n>> Rekomendasi family terbaik: {best}')
    print(f'   (file di {C.OUT_DIR})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=list(C.SCENARIOS) + ['both', 'all'], default='both')
    ap.add_argument('--n-iter', type=int, default=25, help='iterasi RandomizedSearch')
    ap.add_argument('--n-outer', type=int, default=5)
    ap.add_argument('--n-inner', type=int, default=3)
    ap.add_argument('--seed', type=int, default=C.RANDOM_STATE)
    args = ap.parse_args()

    scenarios = C.expand_scenarios(args.scenario)
    for sc in scenarios:
        run_scenario(sc, args.n_iter, args.n_outer, args.n_inner, args.seed)


if __name__ == '__main__':
    main()
