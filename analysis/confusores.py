"""Exploratorio: ruido de red (50 Hz) y actividad lenta frontal (ocular) en la señal original, por persona.
Uso: python confusores.py  (escribe confusores.csv)"""
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # raíz del repositorio
EEG1_DIR = _os.environ.get("EEG1_DIR", "data/EEG_1")
EEG2_DIR = _os.environ.get("EEG2_DIR", "data/EEG_2")

import sys
import numpy as np, pandas as pd, mne
from scipy.signal import welch
sys.path.insert(0, "mci_nsga2")
from mci_nsga2.data_loader import find_subjects, _normalize_ch
from mci_nsga2 import config as C

mne.set_log_level("ERROR")
FRONT = ["Fp1", "Fp2", "F7", "F8"]
rows = []
for base, path, excl in (("EEG_1", EEG1_DIR, None), ("EEG_2", EEG2_DIR, None)):
    for s in find_subjects(path, None):
        raw = mne.io.read_raw_edf(s.path, preload=True)
        ren = {}
        for old in raw.ch_names:
            new = _normalize_ch(old)
            if new in C.CHANNELS and new not in ren.values():
                ren[old] = new
        raw.rename_channels(ren); raw.pick(C.CHANNELS)
        if np.median(np.abs(raw._data)) > 1e-3:
            raw._data *= 1e-6
        X = raw.get_data()[:, : int(600 * raw.info["sfreq"])] * 1e6
        f, P = welch(X, fs=raw.info["sfreq"], nperseg=1024, axis=-1)
        band = lambda lo, hi: P[:, (f >= lo) & (f < hi)].sum(-1)
        tot = band(1, 45)
        linea = np.log10(band(49, 51).mean() / tot.mean())
        idx = [C.CHANNELS.index(c) for c in FRONT]
        ocular = np.log10(band(0.5, 4)[idx].mean())
        post = [C.CHANNELS.index(c) for c in ["P3", "Pz", "P4", "O1", "O2"]]
        rows.append({"base": base, "sid": s.sid, "y": s.label, "linea50_log": linea, "frontal_lento_log": ocular,
                     "amplitud_rms": float(np.sqrt((X ** 2).mean()))})
        print(base, s.sid, f"{linea:.2f} {ocular:.2f}", flush=True)
pd.DataFrame(rows).to_csv("results/exploratory/confusores.csv", index=False)
