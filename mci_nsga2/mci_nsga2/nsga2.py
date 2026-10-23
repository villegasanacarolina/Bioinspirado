"""
NSGA-II (Deb et al., 2002) para selección de canales / características.

Cromosoma (Figs. 5 y 6 del paper):
    [ b_1 ... b_L | p_1 (... p_2) ]
    b_i ∈ {0,1}: canal (L=19) o característica (L=114 DWT / 133 VMD) seleccionada
    p_j entero : hiperparámetro(s) del clasificador (K y distancia en KNN, kernel en SVM, ...)

Objetivos (Ecs. 6 y 7), ambos se MINIMIZAN internamente:
    f1 = número de canales (o características) seleccionados
    f2 = -CA promedio LOSO
Restricción No_ch >= 1: se maneja con REPARACIÓN (si un individuo no selecciona
nada, se le enciende un bit al azar), en vez de castigarlo.

Lo que el paper SÍ especifica: tamaño de población y generaciones
(200/50 canales; 1000/100 características). Los operadores no los especifica;
aquí se usan los estándar para codificación binaria:
    selección por torneo binario (rango, distancia de hacinamiento),
    cruza uniforme (pc = 0.9), mutación bit-flip (1/L) y reinicio aleatorio
    de los genes enteros (pm = 0.2).
"""
from __future__ import annotations

import time

import numpy as np
from joblib import Parallel, delayed

from .classify import loso


# ------------------------------------------------------------ utilidades NSGA
def fast_non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    """F (n, m) a minimizar -> lista de frentes (índices)."""
    n = len(F)
    le = (F[:, None, :] <= F[None, :, :]).all(-1)
    lt = (F[:, None, :] < F[None, :, :]).any(-1)
    dom = le & lt                          # dom[i, j]: i domina a j
    n_dom = dom.sum(0)                     # cuántos dominan a j
    fronts, current = [], np.where(n_dom == 0)[0]
    while len(current):
        fronts.append(current)
        n_dom = n_dom - dom[current].sum(0)
        n_dom[current] = -1
        for f in fronts:
            n_dom[f] = -1
        current = np.where(n_dom == 0)[0]
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        o = np.argsort(F[:, j], kind="stable")
        span = F[o[-1], j] - F[o[0], j]
        d[o[0]] = d[o[-1]] = np.inf
        if span > 0:
            d[o[1:-1]] += (F[o[2:], j] - F[o[:-2], j]) / span
    return d


def rank_and_crowd(F):
    rank = np.empty(len(F), dtype=int)
    crowd = np.empty(len(F))
    fronts = fast_non_dominated_sort(F)
    for r, fr in enumerate(fronts):
        rank[fr] = r
        crowd[fr] = crowding_distance(F[fr])
    return rank, crowd, fronts


# ------------------------------------------------------------ problema
class SelectionProblem:
    """mode='channels': cada bit enciende las n_bands columnas de un canal.
    mode='features': cada bit es una columna."""

    def __init__(self, X, y, groups, n_bands, clf_name, param_bounds,
                 mode="channels", standardize=True, seed=0):
        self.X, self.y, self.groups = X, y, groups
        self.n_bands, self.mode = n_bands, mode
        self.clf_name, self.param_bounds = clf_name, param_bounds
        self.standardize, self.seed = standardize, seed
        self.n_ch = X.shape[1] // n_bands
        self.L = self.n_ch if mode == "channels" else X.shape[1]

    def columns(self, mask):
        if self.mode == "channels":
            ch = np.flatnonzero(mask)
            return (ch[:, None] * self.n_bands + np.arange(self.n_bands)).ravel()
        return np.flatnonzero(mask)

    def evaluate(self, chrom):
        mask, params = chrom[: self.L], chrom[self.L:]
        cols = self.columns(mask)
        acc = loso(self.X[:, cols], self.y, self.groups, self.clf_name, params,
                   standardize=self.standardize, seed=self.seed)
        return int(mask.sum()), acc


def _eval_batch(problem, chroms):
    return [problem.evaluate(c) for c in chroms]


# ------------------------------------------------------------ algoritmo
class NSGA2:
    def __init__(self, problem: SelectionProblem, pop_size=200, n_gen=50,
                 pc=0.9, pm_bit=None, pm_int=0.2, seed=0, n_jobs=1, verbose=True):
        self.p = problem
        self.N, self.G = pop_size, n_gen
        self.pc, self.pm_int = pc, pm_int
        self.pm_bit = pm_bit if pm_bit is not None else 1.0 / problem.L
        self.rng = np.random.default_rng(seed)
        self.n_jobs, self.verbose = n_jobs, verbose
        self.lo = np.array([b[0] for b in problem.param_bounds], dtype=int)
        self.hi = np.array([b[1] for b in problem.param_bounds], dtype=int)
        self.cache: dict[bytes, tuple[int, float]] = {}
        self.history = []

    # --- operadores
    def _random_pop(self, n):
        L = self.p.L
        dens = self.rng.uniform(1.0 / L, 0.5, size=(n, 1))   # densidades variadas
        bits = (self.rng.random((n, L)) < dens).astype(np.int64)
        ints = self.rng.integers(self.lo, self.hi + 1, size=(n, len(self.lo)))
        return self._repair(np.hstack([bits, ints]))

    def _repair(self, P):
        L = self.p.L
        empty = P[:, :L].sum(1) == 0
        for i in np.flatnonzero(empty):
            P[i, self.rng.integers(L)] = 1
        P[:, L:] = np.clip(P[:, L:], self.lo, self.hi)
        return P

    def _tournament(self, rank, crowd, n):
        a = self.rng.integers(len(rank), size=n)
        b = self.rng.integers(len(rank), size=n)
        better_a = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] >= crowd[b]))
        return np.where(better_a, a, b)

    def _variation(self, P, rank, crowd):
        L = self.p.L
        n = self.N
        par = P[self._tournament(rank, crowd, n + n % 2)]
        kids = par.copy()
        for i in range(0, len(par), 2):
            if self.rng.random() < self.pc:
                swap = self.rng.random(par.shape[1]) < 0.5
                kids[i, swap], kids[i + 1, swap] = par[i + 1, swap], par[i, swap]
        flip = self.rng.random((len(kids), L)) < self.pm_bit
        kids[:, :L] = np.where(flip, 1 - kids[:, :L], kids[:, :L])
        if len(self.lo):
            reset = self.rng.random((len(kids), len(self.lo))) < self.pm_int
            new = self.rng.integers(self.lo, self.hi + 1, size=reset.shape)
            kids[:, L:] = np.where(reset, new, kids[:, L:])
        return self._repair(kids[:n])

    # --- evaluación con caché
    def _evaluate(self, P):
        keys = [c.tobytes() for c in P]
        todo = {}
        for k, c in zip(keys, P):
            if k not in self.cache and k not in todo:
                todo[k] = c
        if todo:
            items = list(todo.items())
            if self.n_jobs == 1:
                res = _eval_batch(self.p, [c for _, c in items])
            else:
                chunks = np.array_split(np.arange(len(items)), max(1, min(len(items), 4 * abs(self.n_jobs))))
                out = Parallel(n_jobs=self.n_jobs)(
                    delayed(_eval_batch)(self.p, [items[i][1] for i in ch]) for ch in chunks if len(ch))
                res = [r for o in out for r in o]
            for (k, _), r in zip(items, res):
                self.cache[k] = r
        vals = np.array([self.cache[k] for k in keys], dtype=float)
        return np.column_stack([vals[:, 0], -vals[:, 1]])     # minimizar ambos

    def run(self):
        t0 = time.time()
        P = self._random_pop(self.N)
        F = self._evaluate(P)
        rank, crowd, _ = rank_and_crowd(F)
        for g in range(1, self.G + 1):
            Q = self._variation(P, rank, crowd)
            FQ = self._evaluate(Q)
            R, FR = np.vstack([P, Q]), np.vstack([F, FQ])
            # quitar duplicados exactos para conservar diversidad
            _, uniq = np.unique(R, axis=0, return_index=True)
            uniq = np.sort(uniq)
            R, FR = R[uniq], FR[uniq]
            rank_r, crowd_r, fronts = rank_and_crowd(FR)
            keep = []
            for fr in fronts:
                if len(keep) + len(fr) <= self.N:
                    keep.extend(fr)
                else:
                    rest = fr[np.argsort(-crowd_r[fr])][: self.N - len(keep)]
                    keep.extend(rest)
                    break
            if len(keep) < self.N:                       # rellenar si hubo muchos duplicados
                extra = self._random_pop(self.N - len(keep))
                R = np.vstack([R, extra]); FR = np.vstack([FR, self._evaluate(extra)])
                keep.extend(range(len(R) - len(extra), len(R)))
            keep = np.array(keep)
            P, F = R[keep], FR[keep]
            rank, crowd, _ = rank_and_crowd(F)
            best = -F[:, 1].min()                        # F[:,1] = -CA
            self.history.append({"gen": g, "best_acc": float(best),
                                 "front_size": int((rank == 0).sum()),
                                 "evals": len(self.cache), "t": time.time() - t0})
            if self.verbose:
                print(f"  gen {g:3d}/{self.G}  mejor CA={best:6.2f}%  "
                      f"|frente|={(rank == 0).sum():3d}  evals={len(self.cache):6d}  "
                      f"{time.time() - t0:7.1f}s", flush=True)
        self.P, self.F, self.rank = P, F, rank
        return self

    def pareto_front(self):
        """Frente final (sin duplicados): lista de dicts ordenada por # seleccionados."""
        idx = np.flatnonzero(self.rank == 0)
        out, seen = [], set()
        L = self.p.L
        for i in idx[np.argsort(self.F[idx, 0])]:
            key = (int(self.F[i, 0]), round(-self.F[i, 1], 6))
            if key in seen:
                continue
            seen.add(key)
            out.append({"n_selected": int(self.F[i, 0]), "acc": float(-self.F[i, 1]),
                        "mask": self.P[i, :L].tolist(), "params": self.P[i, L:].tolist()})
        return out
