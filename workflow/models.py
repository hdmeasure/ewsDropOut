#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definisi family model + ruang hyperparameter (dipakai m1 & m2).
Imbalance ditangani via class_weight / scale_pos_weight (TANPA SMOTE).
"""
import numpy as np
from scipy.stats import randint, uniform, loguniform


def make_estimator(name, scale_pos_weight=1.0, random_state=42):
    if name == 'logreg':
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=2000,
                                       solver='liblinear', random_state=random_state)),
        ])
    if name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(class_weight='balanced_subsample',
                                      random_state=random_state, n_jobs=-1)
    if name == 'xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(eval_metric='logloss', random_state=random_state,
                             n_jobs=-1, tree_method='hist',
                             scale_pos_weight=scale_pos_weight)
    raise ValueError(name)


def param_space(name):
    if name == 'logreg':
        return {'clf__C': loguniform(1e-2, 1e2),
                'clf__penalty': ['l1', 'l2']}
    if name == 'rf':
        return {'n_estimators': randint(200, 600),
                'max_depth': randint(3, 14),
                'min_samples_leaf': randint(1, 20),
                'max_features': ['sqrt', 'log2', None]}
    if name == 'xgb':
        return {'n_estimators': randint(150, 500),
                'max_depth': randint(2, 6),
                'learning_rate': loguniform(1e-2, 3e-1),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'min_child_weight': randint(1, 10),
                'reg_lambda': loguniform(1e-1, 1e1)}
    raise ValueError(name)


FAMILIES = ['logreg', 'rf', 'xgb']


def pos_weight(y):
    y = np.asarray(y).astype(int)
    n_pos = max(1, int((y == 1).sum()))
    n_neg = int((y == 0).sum())
    return n_neg / n_pos
