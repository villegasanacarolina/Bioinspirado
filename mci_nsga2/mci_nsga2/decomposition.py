"""
Descomposición de cada segmento en sub-bandas (Fig. 4 del paper).

DWT -> A5, D5, D4, D3, D2 (+ la señal original); D1 (64-128 Hz) se descarta.
VMD -> VMF1..VMF5 ordenadas de mayor a menor frecuencia central, residuo, original.

La VMD sigue el algoritmo de Dragomiretskiy & Zosso (2014), vectorizada para
descomponer los 19 canales de un segmento a la vez.
"""
from __future__ import annotations

import numpy as np
import pywt
from scipy.signal import find_peaks

from . import config as C


# --------------------------------------------------------------------- DWT
def dwt_bands(seg: np.ndarray, wavelet: str = C.WAVELET, level: int = C.DWT_LEVEL):
    """seg: (ch, N). Regresa lista de arreglos (ch, n_b) en el orden C.DWT_BANDS."""
    coeffs = pywt.wavedec(seg, wavelet, level=level, axis=-1)   # [A5, D5, D4, D3, D2, D1]
    A5, D5, D4, D3, D2, _D1 = coeffs
    return [D2, D3, D4, D5, A5, seg]


# --------------------------------------------------------------------- VMD
def _init_omega_peaks(x: np.ndarray, K: int) -> np.ndarray:
    """Frecuencias centrales iniciales (normalizadas, ciclos/muestra) a partir de
    los K picos más grandes del espectro, como la opción 'peaks' de MATLAB."""
    B, N = x.shape
    spec = np.abs(np.fft.rfft(x, axis=1))
    freqs = np.fft.rfftfreq(N)
    omega = np.zeros((B, K))
    for b in range(B):
        pk, _ = find_peaks(spec[b])
        pk = pk[np.argsort(spec[b, pk])[::-1][:K]]
        f0 = np.sort(freqs[pk])
        if len(f0) < K:                                   # rellena si faltan picos
            f0 = np.sort(np.concatenate([f0, np.linspace(0.01, 0.45, K - len(f0))]))
        omega[b] = f0
    return omega


def vmd(x: np.ndarray, K: int = C.VMD_K, alpha: float = C.VMD_ALPHA,
        tau: float = C.VMD_TAU, max_iter: int = C.VMD_MAX_ITER, tol: float = C.VMD_TOL):
    """x: (B, N) con N par.
    Regresa (modes (B, K, N), center_freqs (B, K) en ciclos/muestra),
    con los modos ordenados de mayor a menor frecuencia central."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    B, N = x.shape
    if N % 2:
        raise ValueError("N debe ser par")
    half = N // 2
    # extensión espejo (como en la implementación original)
    xm = np.concatenate([x[:, :half][:, ::-1], x, x[:, -half:][:, ::-1]], axis=1)
    T = xm.shape[1]
    freqs = np.arange(T) / T - 0.5
    f_hat = np.fft.fftshift(np.fft.fft(xm, axis=1), axes=1)
    f_hat_plus = f_hat.copy()
    f_hat_plus[:, : T // 2] = 0

    omega = _init_omega_peaks(x, K)
    # El semieje negativo de f_hat_plus es 0, y con u=0, lambda=0 iniciales se
    # queda en 0 para siempre: basta con iterar sobre las frecuencias >= 0.
    pos = slice(T // 2, T)
    fp = f_hat_plus[:, pos]
    fpos = freqs[pos]
    u_hat = np.zeros((B, K, fp.shape[1]), dtype=complex)
    lam = np.zeros_like(fp)
    active = np.arange(B)                       # filas (canales) que no han convergido

    for _ in range(max_iter):
        a = active
        ua, la, fa, oa = u_hat[a], lam[a], fp[a], omega[a]
        u_old = ua.copy()
        total = ua.sum(axis=1)
        for k in range(K):
            others = total - ua[:, k]
            ua[:, k] = (fa - others - la / 2) / (1 + alpha * (fpos[None, :] - oa[:, k:k + 1]) ** 2)
            total = others + ua[:, k]
            pw = np.abs(ua[:, k]) ** 2
            oa[:, k] = (pw @ fpos) / np.maximum(pw.sum(axis=1), 1e-300)
        la = la + tau * (total - fa)
        u_hat[a], lam[a], omega[a] = ua, la, oa
        diff = (np.abs(ua - u_old) ** 2).sum(axis=(1, 2))
        norm = (np.abs(u_old) ** 2).sum(axis=(1, 2)) + 1e-12
        active = a[diff / norm > tol]
        if active.size == 0:
            break

    # reconstrucción en el tiempo (espectro hermitiano)
    full = np.zeros((B, K, T), dtype=complex)
    full[:, :, pos] = u_hat
    idx = np.arange(1, T // 2 + 1)[::-1]
    full[:, :, idx] = np.conj(u_hat)
    full[:, :, 0] = np.conj(full[:, :, -1])
    u = np.real(np.fft.ifft(np.fft.ifftshift(full, axes=-1), axis=-1))
    u = u[:, :, T // 4 : 3 * T // 4]

    order = np.argsort(-omega, axis=1)                    # mayor frecuencia primero
    u = np.take_along_axis(u, order[:, :, None], axis=1)
    omega = np.take_along_axis(omega, order, axis=1)
    return u, omega


def vmd_bands(seg: np.ndarray):
    """seg: (ch, N) -> [VMF1..VMF5, Rest, Orig], cada uno (ch, N)."""
    modes, _ = vmd(seg)
    rest = seg - modes.sum(axis=1)
    return [modes[:, k] for k in range(modes.shape[1])] + [rest, seg]


DECOMPOSERS = {"DWT": (dwt_bands, C.DWT_BANDS), "VMD": (vmd_bands, C.VMD_BANDS)}
