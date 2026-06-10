#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature selection HYBRID VOTING (dipakai bersama oleh m1 & m2).

Fitur dipertahankan bila muncul di >= 2 dari 3 metode:
  1. Signifikansi regresi logistik (p < 0.05)   [statsmodels; fallback f_classif]
  2. Random Forest importance > median
  3. SHAP |value| (XGBoost) > median

Semua dihitung HANYA pada data train yang diberikan (cegah kebocoran).
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def _sig_logit(X, y):
    """Set fitur dengan p<0.05 dari regresi logistik multivariat."""
    cols = list(X.columns)
    try:
        import statsmodels.api as sm
        Xc = sm.add_constant(X.astype(float), has_constant='add')
        # standardisasi untuk kestabilan numerik
        Xs = Xc.copy()
        for c in cols:
            sd = Xs[c].std()
            if sd > 0:
                Xs[c] = (Xs[c] - Xs[c].mean()) / sd
        model = sm.Logit(y.astype(int).values, Xs.values)
        res = model.fit(disp=0, maxiter=200, method='bfgs')
        pvals = res.pvalues[1:]  # buang const
        return {c for c, p in zip(cols, pvals) if np.isfinite(p) and p < 0.05}
    except Exception:
        from sklearn.feature_selection import f_classif
        F, p = f_classif(X.astype(float).values, y.astype(int).values)
        return {c for c, pp in zip(cols, p) if np.isfinite(pp) and pp < 0.05}


def _rf_importance(X, y, random_state=42):
    """Set fitur dengan RF importance di atas median."""
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                random_state=random_state, n_jobs=-1)
    rf.fit(X.astype(float).values, y.astype(int).values)
    imp = rf.feature_importances_
    med = np.median(imp)
    return {c for c, v in zip(X.columns, imp) if v > med}


def _shap_xgb(X, y, random_state=42):
    """Set fitur dengan mean|SHAP| di atas median (model XGBoost)."""
    try:
        from xgboost import XGBClassifier
        import shap
        pos = max(1, int((y == 0).sum())) / max(1, int((y == 1).sum()))
        xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                            subsample=0.9, colsample_bytree=0.9,
                            scale_pos_weight=pos, eval_metric='logloss',
                            random_state=random_state, n_jobs=-1,
                            tree_method='hist')
        xgb.fit(X.astype(float).values, y.astype(int).values)
        expl = shap.TreeExplainer(xgb)
        sv = expl.shap_values(X.astype(float).values)
        if isinstance(sv, list):
            sv = sv[-1]
        mean_abs = np.abs(sv).mean(axis=0)
        med = np.median(mean_abs)
        return {c for c, v in zip(X.columns, mean_abs) if v > med}
    except Exception:
        # fallback: gain importance
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                            eval_metric='logloss', random_state=random_state,
                            n_jobs=-1, tree_method='hist')
        xgb.fit(X.astype(float).values, y.astype(int).values)
        imp = xgb.feature_importances_
        med = np.median(imp)
        return {c for c, v in zip(X.columns, imp) if v > med}


def select_features(X, y, random_state=42, return_detail=False):
    """Voting >= 2 dari 3. Mengembalikan list fitur terpilih (urutan asli X.columns)."""
    y = pd.Series(np.asarray(y).astype(int))
    X = X.reset_index(drop=True)
    s1 = _sig_logit(X, y)
    s2 = _rf_importance(X, y, random_state)
    s3 = _shap_xgb(X, y, random_state)

    votes = {}
    for c in X.columns:
        v = (c in s1) + (c in s2) + (c in s3)
        votes[c] = v
    selected = [c for c in X.columns if votes[c] >= 2]
    # jaga minimal 2 fitur supaya model tetap bisa dilatih
    if len(selected) < 2:
        ranked = sorted(X.columns, key=lambda c: votes[c], reverse=True)
        selected = ranked[:max(2, len(selected))]

    if return_detail:
        detail = pd.DataFrame({
            'feature': list(X.columns),
            'sig_logit': [int(c in s1) for c in X.columns],
            'rf_imp': [int(c in s2) for c in X.columns],
            'shap_xgb': [int(c in s3) for c in X.columns],
            'votes': [votes[c] for c in X.columns],
            'selected': [int(c in selected) for c in X.columns],
        })
        return selected, detail
    return selected
