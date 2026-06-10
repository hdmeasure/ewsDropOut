#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export trained model bundles (joblib) into portable formats for the platform team:
  - <name>_booster.json : native XGBoost booster (loadable in Python & R)
  - <name>_spec.json    : feature order, calibration, prior correction, threshold
  - <name>.joblib       : original Python bundle
  - <name>.rds          : produced separately by make_rds.R (reads the two JSONs)

The spec.json is the single source of truth for the scoring math, so any runtime
(R, Python, Java, ...) can reproduce predictions without the Python objects.

Usage:
    python3 export_model.py --in-dir /path/to/joblib_bundles --out-dir models \
        --models aspd_num tanpa_aspd
"""
import os, sys, json, argparse, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
# scorer.ModelBundle lives in ../workflow
sys.path.insert(0, os.path.join(HERE, '..', 'workflow'))
from scorer import ModelBundle  # noqa: E402


def export_one(name, in_dir, out_dir):
    bundle = ModelBundle.load(os.path.join(in_dir, f'model_{name}.joblib'))
    if bundle.family != 'xgb':
        raise SystemExit(
            f'[{name}] family={bundle.family}. Hanya XGBoost yang didukung untuk '
            f'ekspor portabel. Latih ulang dengan --family xgb.')

    os.makedirs(out_dir, exist_ok=True)

    # 1) native booster
    booster_path = os.path.join(out_dir, f'{name}_booster.json')
    bundle.estimator.get_booster().save_model(booster_path)

    # 2) spec (kalibrasi + koreksi prior + threshold)
    if bundle.cal_method == 'sigmoid':
        a, b = bundle.calibrator
        cal = {'method': 'sigmoid', 'a': float(a), 'b': float(b)}
    else:  # isotonic
        ir = bundle.calibrator
        cal = {'method': 'isotonic',
               'x': [float(v) for v in ir.X_thresholds_],
               'y': [float(v) for v in ir.y_thresholds_]}

    spec = {
        'name': name,
        'family': bundle.family,
        'objective': 'binary:logistic',
        'features': list(bundle.features),      # URUTAN WAJIB dipertahankan
        'calibration': cal,
        'pi_train': float(bundle.pi_train),
        'pi_pop': float(bundle.pi_pop),
        'threshold': float(bundle.threshold),
        'scoring_steps': [
            'p_raw = booster.predict(X[features])',
            'p_cal = calibrate(p_raw)               # sigmoid: 1/(1+exp(-(a*logit(p)+b)))',
            'offset = log(pi_pop/(1-pi_pop)) - log(pi_train/(1-pi_train))',
            'p_adj = sigmoid(logit(p_cal) + offset)',
            'label = p_adj >= threshold',
        ],
        'notes': 'Threshold operasional sebaiknya dihitung ulang pada populasi '
                 'implementasi (kuantil 1-pi_pop) agar proporsi ter-flag = pi_pop.',
    }
    with open(os.path.join(out_dir, f'{name}_spec.json'), 'w') as fh:
        json.dump(spec, fh, indent=2)

    # 3) salinan joblib (Python)
    shutil.copy(os.path.join(in_dir, f'model_{name}.joblib'),
                os.path.join(out_dir, f'{name}.joblib'))

    print(f'[{name}] -> booster.json, spec.json, joblib  '
          f'(features={len(bundle.features)}, cal={bundle.cal_method})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True, help='folder berisi model_<name>.joblib')
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'models'))
    ap.add_argument('--models', nargs='+', default=['aspd_num', 'tanpa_aspd'])
    args = ap.parse_args()
    for name in args.models:
        export_one(name, args.in_dir, args.out_dir)
    print(f'\nSelesai. Artefak di: {args.out_dir}')
    print('Lanjutkan dengan: Rscript make_rds.R  (untuk menghasilkan .rds)')


if __name__ == '__main__':
    main()
