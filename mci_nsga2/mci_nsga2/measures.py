"""
Las 10 medidas del paper. Todas operan sobre el último eje de un arreglo (B, n)
y regresan (B,): un valor por sub-banda.

Las entropías siguen las definiciones de `wentropy` de MATLAB (que es lo que
usan los autores en sus trabajos previos, ref. 21 y 40 del paper).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import iqr as _iqr

from . import config as C

EPS = 1e-12


def std(x):
    return np.std(x, axis=-1, ddof=1)


def iqr(x):
    return _iqr(x, axis=-1)


def log_band_power(x):
    """LBP: log10 de la potencia media de la sub-banda."""
    return np.log10(np.mean(x ** 2, axis=-1) + EPS)


def teager_energy(x):
    """TeEng: log10( sum |x[n]^2 - x[n-1] x[n+1]| ) (operador de Teager-Kaiser)."""
    psi = x[..., 1:-1] ** 2 - x[..., :-2] * x[..., 2:]
    return np.log10(np.sum(np.abs(psi), axis=-1) + EPS)


def shannon_entropy(x):
    """ShEn (wentropy 'shannon'): -sum x^2 log(x^2)."""
    s2 = x ** 2
    return -np.sum(s2 * np.log(s2 + EPS), axis=-1)


def transformed_shannon_entropy(x):
    """TShEn: Shannon sobre la distribución de energía normalizada p_i = x_i^2/sum x^2.
    SUPUESTO: el paper no la define; esta es la definición 'normalizada' de wentropy."""
    s2 = x ** 2
    p = s2 / (s2.sum(axis=-1, keepdims=True) + EPS)
    return -np.sum(p * np.log2(p + EPS), axis=-1)


def sure_entropy(x, p=C.ENTROPY_THRESHOLD_UV):
    """SuEn (wentropy 'sure'): n - #{|x|<=p} + sum min(x^2, p^2)."""
    n = x.shape[-1]
    return n - np.sum(np.abs(x) <= p, axis=-1) + np.sum(np.minimum(x ** 2, p ** 2), axis=-1)


def threshold_entropy(x, p=C.ENTROPY_THRESHOLD_UV):
    """ThEn (wentropy 'threshold'): # de muestras con |x| > p."""
    return np.sum(np.abs(x) > p, axis=-1).astype(float)


def katz_fd(x):
    """KFD = log10(n) / (log10(n) + log10(d/L))."""
    dx = np.abs(np.diff(x, axis=-1))
    L = dx.sum(axis=-1)
    d = np.max(np.abs(x - x[..., :1]), axis=-1)
    n = x.shape[-1] - 1
    return np.log10(n) / (np.log10(n) + np.log10((d + EPS) / (L + EPS)))


def higuchi_fd(x, kmax=C.HFD_KMAX):
    """HFD: pendiente de log L(k) contra log(1/k)."""
    x = np.atleast_2d(x)
    N = x.shape[-1]
    lk = []
    ks = np.arange(1, kmax + 1)
    for k in ks:
        Lm = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            seg = x[..., idx]
            norm = (N - 1) / ((len(idx) - 1) * k)
            Lm.append(np.abs(np.diff(seg, axis=-1)).sum(axis=-1) * norm / k)
        lk.append(np.mean(Lm, axis=0))
    lk = np.log(np.stack(lk, axis=-1) + EPS)               # (B, kmax)
    xk = np.log(1.0 / ks)
    xk = xk - xk.mean()
    return (lk - lk.mean(axis=-1, keepdims=True)) @ xk / (xk @ xk)


MEASURE_FUNCS = {
    "STD": std, "IQR": iqr, "LBP": log_band_power, "TeEng": teager_energy,
    "ShEn": shannon_entropy, "TShEn": transformed_shannon_entropy,
    "SuEn": sure_entropy, "ThEn": threshold_entropy, "KFD": katz_fd, "HFD": higuchi_fd,
}
assert list(MEASURE_FUNCS) == C.MEASURES
