"""
Clasificadores (RF, DA, SVM, KNN) y validación Leave-One-Subject-Out (LOSO).

Métrica del paper (Ec. 1): en cada iteración LOSO se calcula CA sobre los
segmentos del sujeto que se dejó fuera; el resultado es el PROMEDIO de las 24 CA.

Como cada sujeto es de una sola clase, la sensibilidad sólo está definida en los
pliegues de sujetos MCI y la especificidad en los de HC; por eso:
  sensibilidad = promedio de CA en sujetos MCI
  especificidad = promedio de CA en sujetos HC
y precisión/F-score se calculan con todas las predicciones juntas.
(SUPUESTO razonable: el paper dice que promedia, pero no detalla este caso.)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from . import config as C


class _KNN:
    """KNN con las 10 distancias de MATLAB (incluye seuclidean, mahalanobis y
    spearman, que necesitan estadísticos del conjunto de entrenamiento)."""

    def __init__(self, k: int, dist_idx: int):
        self.k = k
        self.dist = C.KNN_DISTANCES[dist_idx - 1]

    def _tf(self, X):
        return rankdata(X, axis=1) if self.dist == "spearman" else X

    def fit(self, X, y):
        d, kw = self.dist, {}
        if d == "cityblock":
            d = "manhattan"
        elif d == "chebychev":
            d = "chebyshev"
        elif d == "minkowski":
            kw = {"p": 2}                          # exponente por defecto de MATLAB
        elif d == "seuclidean":
            kw = {"metric_params": {"V": np.var(X, axis=0, ddof=1) + 1e-12}}
        elif d == "mahalanobis":
            cov = np.atleast_2d(np.cov(X, rowvar=False))
            kw = {"metric_params": {"VI": np.linalg.pinv(cov)}}
        elif d == "spearman":
            d = "correlation"
        Xt = self._tf(X)
        if d in ("correlation", "cosine") and Xt.shape[1] == 1:
            d = "euclidean"                        # con 1 variable no están definidas
        self.m = KNeighborsClassifier(n_neighbors=self.k, metric=d,
                                      algorithm="brute", **kw).fit(Xt, y)
        return self

    def predict(self, X):
        return self.m.predict(self._tf(X))


def make_classifier(name: str, params, seed: int = 0):
    p = [int(v) for v in params]
    if name == "KNN":
        return _KNN(p[0], p[1])
    if name == "SVM":
        kernel = C.SVM_KERNELS[p[0] - 1]
        return SVC(kernel=kernel, degree=3, gamma="scale", C=1.0)
    if name == "DA":
        if p[0] == 1:
            return LinearDiscriminantAnalysis()
        return QuadraticDiscriminantAnalysis(reg_param=C.QDA_REG)
    if name == "RF":
        return RandomForestClassifier(n_estimators=C.RF_N_TREES, max_depth=p[0],
                                      bootstrap=True, n_jobs=1, random_state=seed)
    raise ValueError(name)


def _zscore(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def loso(X, y, groups, clf_name, params, standardize=True, seed=0, full=False):
    """Promedio de CA (%) sobre los sujetos. Con full=True regresa todas las métricas."""
    accs, labels, y_all, p_all = [], [], [], []
    for g in np.unique(groups):
        te = groups == g
        Xtr, Xte = X[~te], X[te]
        if standardize:
            Xtr, Xte = _zscore(Xtr, Xte)
        clf = make_classifier(clf_name, params, seed).fit(Xtr, y[~te])
        pred = clf.predict(Xte)
        accs.append(100.0 * np.mean(pred == y[te]))
        labels.append(y[te][0])
        if full:
            y_all.append(y[te]); p_all.append(pred)
    accs, labels = np.array(accs), np.array(labels)
    ca = float(accs.mean())
    if not full:
        return ca
    yt, yp = np.concatenate(y_all), np.concatenate(p_all)
    tp = np.sum((yp == 1) & (yt == 1)); fp = np.sum((yp == 1) & (yt == 0))
    prec = 100.0 * tp / max(tp + fp, 1)
    sens = float(accs[labels == 1].mean())
    spec = float(accs[labels == 0].mean())
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    return {"acc": ca, "sens": sens, "spec": spec, "prec": float(prec), "f1": float(f1),
            "per_subject": accs.tolist()}


def kfold(X, y, clf_name, params, k=10, standardize=True, seed=0):
    """CV de 10 pliegues por segmentos (con fuga de sujeto) — para replicar la Tabla 3."""
    accs = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        Xtr, Xte = (_zscore(X[tr], X[te]) if standardize else (X[tr], X[te]))
        pred = make_classifier(clf_name, params, seed).fit(Xtr, y[tr]).predict(Xte)
        accs.append(100.0 * np.mean(pred == y[te]))
    return float(np.mean(accs))
