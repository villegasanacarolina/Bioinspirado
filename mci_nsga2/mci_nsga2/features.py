"""
Construcción de la matriz de características (Fig. 4).

Para no repetir la descomposición 10 veces, se calculan TODAS las medidas a la vez
y se guarda un tensor por método de descomposición:

    F[seg, canal, banda, medida]

Luego cada experimento toma una medida y aplana a (n_seg, ch * n_bandas),
con el orden canal-mayor: feature j = canal * n_bandas + banda
(igual que el cromosoma de la Fig. 6: primero las bandas de Fp1, luego Fp2...).
"""
from __future__ import annotations

import os

import numpy as np
from joblib import Parallel, delayed

from . import config as C
from .decomposition import DECOMPOSERS
from .measures import MEASURE_FUNCS


def segment_features(seg: np.ndarray, method: str) -> np.ndarray:
    """seg (ch, N) -> (ch, n_bandas, n_medidas)."""
    fn, names = DECOMPOSERS[method]
    bands = fn(seg)
    out = np.empty((seg.shape[0], len(names), len(C.MEASURES)))
    for b, sub in enumerate(bands):
        for m, mname in enumerate(C.MEASURES):
            out[:, b, m] = MEASURE_FUNCS[mname](sub)
    return out


def subject_features(segments: np.ndarray, method: str) -> np.ndarray:
    return np.stack([segment_features(s, method) for s in segments])


def build_feature_tensor(subjects, method: str, n_jobs: int = -1, verbose: int = 5):
    """Regresa dict con F (n_seg_total, ch, bandas, medidas), y, groups, sids."""
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(subject_features)(s.segments, method) for s in subjects)
    F = np.concatenate(res)
    y = np.concatenate([[s.label] * len(r) for s, r in zip(subjects, res)])
    groups = np.concatenate([[i] * len(r) for i, r in enumerate(res)])
    return {"F": F, "y": y, "groups": groups,
            "sids": np.array([s.sid for s in subjects]),
            "bands": np.array(DECOMPOSERS[method][1]),
            "channels": np.array(C.CHANNELS),
            "measures": np.array(C.MEASURES),
            "method": method}


def save_tensor(d: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **d)


def load_tensor(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["method"] = str(d["method"])
    return d


def feature_matrix(d: dict, measure: str):
    """Aplana para una medida: X (n_seg, ch*bandas)."""
    m = list(d["measures"]).index(measure)
    F = d["F"][..., m]
    n, ch, nb = F.shape
    X = F.reshape(n, ch * nb)
    names = [f"{c}:{b}" for c in d["channels"] for b in d["bands"]]
    return X, names, nb
