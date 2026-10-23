"""
Selección de electrodos a NIVEL PERSONA (una fila por persona = promedio de sus segmentos).

Modelo y fitness fijados de antemano:
  - LDA con contracción (Ledoit-Wolf), con estandarización.
  - Fitness interno: exactitud balanceada leave-one-out sobre las personas de entrenamiento.
  - Variables: potencia relativa (4 bandas) de los 15 electrodos no oculares (60 variables).

Métodos (los mismos de la fase 2 de desarrollo):
  M0  sin selección
  M1  NSGA-II, regla del paper (mayor exactitud interna)
  M4  búsqueda por zonas, exhaustiva (1 electrodo por zona, 180 combinaciones)
  M6  PSO binario, fitness = exactitud - 1 punto por electrodo
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .limpieza import CH15

NB = 4
ZONAS = {"frontal": ["F3", "Fz", "F4"], "central": ["C3", "Cz", "C4"],
         "temporal": ["T3", "T4", "T5", "T6"], "parieto_occipital": ["P3", "Pz", "P4", "O1", "O2"]}


def cols(mask):
    ch = np.flatnonzero(mask)
    return (ch[:, None] * NB + np.arange(NB)).ravel()


def model():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def balanced(pred, y):
    return 50 * (np.mean(pred[y == 1] == 1) + np.mean(pred[y == 0] == 0))


def inner_score(mask, X, y, cache):
    key = mask.tobytes()
    if key not in cache:
        c = cols(mask); pred = np.zeros(len(y), int)
        for k in range(len(y)):
            tr = np.arange(len(y)) != k
            pred[k] = model().fit(X[tr][:, c], y[tr]).predict(X[~tr][:, c])[0]
        cache[key] = balanced(pred, y)
    return cache[key]


class Problem:
    """Interfaz para mci_nsga2.nsga2.NSGA2."""
    def __init__(self, X, y):
        self.X, self.y, self.L, self.param_bounds, self.cache = X, y, len(CH15), [], {}

    def evaluate(self, chrom):
        m = chrom[:self.L]
        return int(m.sum()), inner_score(m, self.X, self.y, self.cache)


def select_M1(X, y, seed, pop=40, gen=25):
    from .nsga2 import NSGA2
    ga = NSGA2(Problem(X, y), pop_size=pop, n_gen=gen, seed=seed, n_jobs=1, verbose=False).run()
    best = max(ga.pareto_front(), key=lambda s: (s["acc"], -s["n_selected"]))
    return np.array(best["mask"], dtype=np.int64)


def select_M4(X, y):
    cache = {}
    idx = [[CH15.index(c) for c in z] for z in ZONAS.values()]
    best, best_s = None, -1
    for combo in itertools.product(*idx):
        m = np.zeros(len(CH15), np.int64); m[list(combo)] = 1
        s = inner_score(m, X, y, cache)
        if s > best_s:
            best, best_s = m, s
    return best


def select_M6(X, y, seed, n=20, iters=25, lam=1.0):
    rng = np.random.default_rng(seed); cache = {}
    L = len(CH15); v = rng.normal(0, 1, (n, L)); x = (rng.random((n, L)) < 0.3).astype(np.int64)

    def f(m):
        return -1e9 if m.sum() == 0 else inner_score(m, X, y, cache) - lam * m.sum()
    pb = x.copy(); pbf = np.array([f(m) for m in x]); gb = pb[np.argmax(pbf)].copy()
    for _ in range(iters):
        r1, r2 = rng.random((n, L)), rng.random((n, L))
        v = np.clip(0.7 * v + 1.5 * r1 * (pb - x) + 1.5 * r2 * (gb - x), -4, 4)
        x = (rng.random((n, L)) < 1 / (1 + np.exp(-v))).astype(np.int64)
        fx = np.array([f(m) for m in x])
        better = fx > pbf; pb[better], pbf[better] = x[better], fx[better]
        gb = pb[np.argmax(pbf)].copy()
    return gb


def fit_predict(mask, Xtr, ytr, Xte):
    c = cols(mask)
    return model().fit(Xtr[:, c], ytr).predict(Xte[:, c])
