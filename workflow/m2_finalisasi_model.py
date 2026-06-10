#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAHAP 5-6 — Finalisasi model: refit + kalibrasi + koreksi prior + threshold.

Langkah:
  1. Muat split train.
  2. Tentukan family terbaik (dari m1, atau --family).
  3. Feature selection voting FINAL pada seluruh train.
  4. Tuning RandomizedSearchCV (GroupKFold) pada fitur terpilih.
  5. Split train -> fit (80%) / kalibrasi (20%) per sekolah.
  6. Refit estimator pada fit-part; fit kalibrator pada kalibrasi-part.
  7. Koreksi prior: pi_train -> pi_pop (--pi).
  8. Threshold: proporsi ter-flag pada train ≈ pi_pop.
  9. Simpan bundle: output/model_<scenario>.joblib  + ringkasan.

Jalankan:
    python3 m2_finalisasi_model.py --pi 0.03
    python3 m2_finalisasi_model.py --scenario tanpa_aspd --family xgb --pi 0.03
"""
import os, argparse, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV

import common as C
import models as M
from fs_voting import select_features
from scorer import ModelBundle, fit_calibrator

warnings.filterwarnings('ignore')


def best_family_from_m1(scenario):
    p = os.path.join(C.OUT_DIR, f'recommendation_{scenario}.txt')
    if os.path.exists(p):
        with open(p) as fh:
            first = fh.readline()
        if ':' in first:
            return first.split(':', 1)[1].strip()
    return 'xgb'


def run_scenario(scenario, family, pi_pop, n_iter, n_inner, seed):
    print(f'\n{"="*70}\nFINALISASI: {scenario}\n{"="*70}')
    tr_path = os.path.join(C.OUT_DIR, f'split_{scenario}_train.csv')
    if not os.path.exists(tr_path):
        print(f'[LEWAT] {tr_path} belum ada — jalankan m1 dulu.')
        return
    df = pd.read_csv(tr_path, dtype={'nisn': str, 'npsn': str})
    preds = [c for c in C.predictor_cols(scenario) if c in df.columns]
    df = df.dropna(subset=preds + ['status_do']).reset_index(drop=True)

    X = df[preds].astype(float)
    y = df['status_do'].astype(int)
    g = df['npsn'].astype(str)

    if family is None:
        family = best_family_from_m1(scenario)
    print(f'Family: {family} | n_train={len(df)} | prevalensi_train={y.mean():.3%}')

    # ---- 3. feature selection final ----
    sel, detail = select_features(X, y, random_state=seed, return_detail=True)
    detail.to_csv(os.path.join(C.OUT_DIR, f'fs_final_{scenario}.csv'), index=False)
    print(f'Fitur final ({len(sel)}): {sel}')
    Xs = X[sel]

    # ---- 4. tuning ----
    spw = M.pos_weight(y)
    est = M.make_estimator(family, scale_pos_weight=spw, random_state=seed)
    search = RandomizedSearchCV(est, M.param_space(family), n_iter=n_iter,
                                scoring='average_precision',
                                cv=GroupKFold(n_splits=n_inner), n_jobs=-1,
                                random_state=seed, refit=True, error_score=0.0)
    search.fit(Xs.values, y.values, groups=g.values)
    best_params = search.best_params_
    print(f'Best params: {best_params}')
    print(f'CV PR-AUC (tuning): {search.best_score_:.3f}')

    # ---- 5. split fit / kalibrasi (per sekolah) ----
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    fi, ci = next(gss.split(Xs, y, g))
    Xf, yf = Xs.iloc[fi], y.iloc[fi]
    Xc, yc = Xs.iloc[ci], y.iloc[ci]

    # ---- 6. refit estimator pada fit-part + kalibrasi pada calib-part ----
    final_est = M.make_estimator(family, scale_pos_weight=M.pos_weight(yf), random_state=seed)
    final_est.set_params(**best_params)
    final_est.fit(Xf.values, yf.values)

    p_cal_raw = final_est.predict_proba(Xc.values)[:, 1]
    n_pos_c = int(yc.sum())
    method = 'isotonic' if (len(yc) >= 1000 and n_pos_c >= 50) else 'sigmoid'
    calibrator, method = fit_calibrator(p_cal_raw, yc.values, method)
    print(f'Kalibrasi: {method} (n_calib={len(yc)}, pos={n_pos_c})')

    # pi_train = prevalensi calib-part (acuan kalibrator)
    pi_train = float(yc.mean())
    if pi_pop is None:
        pi_pop = pi_train
        print(f'[!] --pi tidak diberikan; pakai pi_pop=pi_train={pi_pop:.3%} (TANPA koreksi). '
              f'Set --pi dengan base rate riil untuk proporsi rasional.')

    # ---- 8. threshold: proporsi ter-flag ≈ pi_pop (pada train penuh) ----
    bundle = ModelBundle(family, sel, final_est, calibrator, method,
                         pi_train, pi_pop, threshold=0.5)
    p_adj_train = bundle.proba_adj(Xs.values)
    thr = float(np.quantile(p_adj_train, 1 - pi_pop))
    bundle.threshold = thr
    flagged = float((p_adj_train >= thr).mean())
    print(f'pi_train={pi_train:.3%} -> pi_pop={pi_pop:.3%} | threshold={thr:.4f} | '
          f'flagged_train={flagged:.3%}')

    # ---- 9. simpan ----
    out = os.path.join(C.OUT_DIR, f'model_{scenario}.joblib')
    bundle.save(out)
    with open(os.path.join(C.OUT_DIR, f'model_{scenario}_info.txt'), 'w') as fh:
        fh.write(f'scenario      : {scenario}\n')
        fh.write(f'family        : {family}\n')
        fh.write(f'features      : {sel}\n')
        fh.write(f'best_params   : {best_params}\n')
        fh.write(f'calibration   : {method}\n')
        fh.write(f'pi_train      : {pi_train:.5f}\n')
        fh.write(f'pi_pop        : {pi_pop:.5f}\n')
        fh.write(f'threshold     : {thr:.5f}\n')
        fh.write(f'flagged_train : {flagged:.5f}\n')
    print(f'>> Tersimpan: {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=list(C.SCENARIOS) + ['both', 'all'], default='both')
    ap.add_argument('--family', choices=M.FAMILIES, default=None,
                    help='paksa family; default = rekomendasi m1')
    ap.add_argument('--pi', type=float, default=None,
                    help='base rate DO populasi (mis. 0.03). Wajib untuk proporsi rasional.')
    ap.add_argument('--n-iter', type=int, default=40)
    ap.add_argument('--n-inner', type=int, default=4)
    ap.add_argument('--seed', type=int, default=C.RANDOM_STATE)
    args = ap.parse_args()

    scenarios = C.expand_scenarios(args.scenario)
    for sc in scenarios:
        run_scenario(sc, args.family, args.pi, args.n_iter, args.n_inner, args.seed)


if __name__ == '__main__':
    main()
