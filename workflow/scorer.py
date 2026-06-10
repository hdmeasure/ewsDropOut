#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelBundle — membungkus estimator + kalibrator + koreksi prior + threshold.
Dipakai m2 (simpan), m3 & m5 (muat & skor).

Alur skor:
  p_raw  = estimator.predict_proba(X)[:,1]
  p_cal  = kalibrasi(p_raw)                         (isotonic / sigmoid)
  p_adj  = koreksi_prior(p_cal, pi_train -> pi_pop) (geser log-odds)
  label  = (p_adj >= threshold)
"""
import numpy as np
import joblib

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class ModelBundle:
    def __init__(self, family, features, estimator, calibrator, cal_method,
                 pi_train, pi_pop, threshold):
        self.family = family
        self.features = list(features)
        self.estimator = estimator
        self.calibrator = calibrator      # objek dengan .predict (isotonic) / koef sigmoid
        self.cal_method = cal_method      # 'isotonic' | 'sigmoid'
        self.pi_train = float(pi_train)
        self.pi_pop = float(pi_pop)
        self.threshold = float(threshold)

    # ---- langkah skor ----
    def proba_raw(self, X):
        return self.estimator.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def proba_cal(self, X):
        p = self.proba_raw(X)
        if self.cal_method == 'isotonic':
            return np.clip(self.calibrator.predict(p), 0, 1)
        # sigmoid: calibrator = (a, b) untuk a*logit(p)+b
        a, b = self.calibrator
        return _sigmoid(a * _logit(p) + b)

    def proba_adj(self, X):
        p_cal = self.proba_cal(X)
        offset = (np.log(self.pi_pop / (1 - self.pi_pop))
                  - np.log(self.pi_train / (1 - self.pi_train)))
        return _sigmoid(_logit(p_cal) + offset)

    def predict_label(self, X):
        return (self.proba_adj(X) >= self.threshold).astype(int)

    # ---- IO ----
    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def fit_calibrator(p_raw, y, method):
    """Kembalikan (calibrator_obj, method)."""
    y = np.asarray(y).astype(int)
    if method == 'isotonic':
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        ir.fit(p_raw, y)
        return ir, 'isotonic'
    # sigmoid (Platt) pada logit(p_raw)
    from sklearn.linear_model import LogisticRegression
    z = _logit(p_raw).reshape(-1, 1)
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(z, y)
    a = float(lr.coef_[0, 0]); b = float(lr.intercept_[0])
    return (a, b), 'sigmoid'
