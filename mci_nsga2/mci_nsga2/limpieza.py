"""
Limpieza de artefactos oculares con ICA y características de potencia relativa.

Pasos (idénticos para cualquier base):
  1. Preprocesado del paper: pasa-banda 0.5-64 Hz, referencia promedio, notch 50/100 Hz.
  2. ICA (15 componentes) ajustada sobre una copia filtrada a 1-40 Hz.
  3. Componentes oculares = el más correlacionado con el parpadeo (promedio Fp1/Fp2 < 4 Hz)
     y el más correlacionado con el movimiento horizontal (F8 - F7 < 4 Hz).
  4. Se quitan esos componentes, se eliminan Fp1, Fp2, F7 y F8 (los más contaminados:
     72%, 68%, 52% y 39% de su señal era ocular en EEG_1) y se recalcula la referencia
     promedio con los 15 electrodos restantes.
  5. Primeros 10 min, segmentos de 10 s -> log de potencia relativa en delta (1-4 Hz),
     theta (4-8), alfa (8-13) y beta (13-30), relativa a 1-30 Hz.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .data_loader import _normalize_ch, segment

OCULARES = ["Fp1", "Fp2", "F7", "F8"]
CH15 = [c for c in C.CHANNELS if c not in OCULARES]
BANDAS = {"delta": (1, 4), "theta": (4, 8), "alfa": (8, 13), "beta": (13, 30)}


def limpiar_ica(path: str, duration_s: float = C.DURATION_S):
    """Regresa (EEG limpio 15 canales en µV, fuentes oculares estandarizadas (2, n))."""
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(path, preload=True)
    ren, used = {}, set()
    for old in raw.ch_names:
        new = _normalize_ch(old)
        if new in C.CHANNELS and new not in used:
            ren[old] = new; used.add(new)
    raw.rename_channels(ren)
    faltan = [c for c in C.CHANNELS if c not in raw.ch_names]
    if faltan:
        raise ValueError(f"{path}: faltan canales {faltan}")
    raw.pick(C.CHANNELS); raw.set_channel_types({c: "eeg" for c in C.CHANNELS})
    if abs(raw.info["sfreq"] - C.FS) > 1e-6:
        raw.resample(C.FS)
    if np.median(np.abs(raw._data)) > 1e-3:          # EDF con µV declarados como V
        raw._data *= 1e-6
    raw.crop(0, min(raw.times[-1], duration_s + 5))
    raw.filter(*C.BANDPASS, fir_design="firwin")
    raw.set_eeg_reference("average", projection=False)
    raw.notch_filter([C.LINE_FREQ, 2 * C.LINE_FREQ], fir_design="firwin")

    fit = raw.copy().filter(1.0, 40.0)
    ica = mne.preprocessing.ICA(n_components=15, random_state=0, max_iter="auto")
    ica.fit(fit)
    i = {c: k for k, c in enumerate(C.CHANNELS)}
    low = raw.copy().filter(None, 4.0).get_data()
    veog = (low[i["Fp1"]] + low[i["Fp2"]]) / 2
    heog = low[i["F8"]] - low[i["F7"]]
    S = ica.get_sources(raw).get_data()
    Sl = mne.filter.filter_data(S, C.FS, None, 4.0)
    rv = [abs(np.corrcoef(c, veog)[0, 1]) for c in Sl]
    rh = [abs(np.corrcoef(c, heog)[0, 1]) for c in Sl]
    ocular = sorted({int(np.argmax(rv)), int(np.argmax(rh))})

    clean = ica.apply(raw.copy(), exclude=ocular).pick(CH15)
    clean.set_eeg_reference("average", projection=False)
    n = int(duration_s * C.FS)
    Xc = clean.get_data()[:, :n] * 1e6
    Xo = S[ocular][:, :n]
    Xo = Xo / Xo.std(axis=1, keepdims=True)
    return Xc, Xo


def potencia_relativa(X: np.ndarray) -> np.ndarray:
    """X (canales, n) -> (n_segmentos, canales*4): log de potencia relativa por banda."""
    from scipy.signal import welch
    segs = segment(X)
    f, P = welch(segs, fs=C.FS, nperseg=512, axis=-1)
    tot = P[..., (f >= 1) & (f < 30)].sum(-1)
    rp = np.stack([P[..., (f >= lo) & (f < hi)].sum(-1) / tot for lo, hi in BANDAS.values()], -1)
    return np.log(rp).reshape(len(segs), -1)
