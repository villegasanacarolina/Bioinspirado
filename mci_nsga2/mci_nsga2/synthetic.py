"""
EEG sintético SOLO para probar que el pipeline corre de punta a punta.
No sustituye al dataset real.

Diseño: ruido 1/f + ritmo alfa por sujeto + ruido de línea a 50 Hz.
En los sujetos "MCI" el alfa se enlentece (~8 Hz) y sube theta, pero SÓLO en
los canales Cz, F8, T4 y Fp1; en el resto de canales no hay diferencia de grupo.
Así se puede comprobar que NSGA-II encuentra esos canales.
"""
from __future__ import annotations

import os

import numpy as np

from . import config as C

INFORMATIVE = ["Cz", "F8", "T4", "Fp1"]


def _pink(rng, n_ch, n):
    X = np.fft.rfft(rng.standard_normal((n_ch, n)), axis=1)
    f = np.fft.rfftfreq(n, 1 / C.FS)
    f[0] = f[1]
    return np.fft.irfft(X / np.sqrt(f), n=n, axis=1)


def make_subject(rng, is_mci, seconds):
    n = int(seconds * C.FS)
    t = np.arange(n) / C.FS
    x = 8 * _pink(rng, len(C.CHANNELS), n) / 10
    alpha_hc = rng.normal(10.0, 0.7)
    for i, ch in enumerate(C.CHANNELS):
        f_alpha = alpha_hc
        theta_amp = rng.uniform(3.0, 6.0)
        if is_mci and ch in INFORMATIVE:
            f_alpha = alpha_hc - rng.normal(0.9, 0.3)
            theta_amp += 2.5
        elif ch not in INFORMATIVE:
            # artefactos de amplitud/espectro variables entre sujetos (ruido "irrelevante")
            x[i] *= rng.uniform(0.5, 2.5)
            x[i] += rng.uniform(0, 8) * np.sin(2 * np.pi * rng.uniform(3, 30) * t)
        am = 1 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t + rng.uniform(0, 6.28))
        x[i] += 12 * am * np.sin(2 * np.pi * f_alpha * t + rng.uniform(0, 6.28))
        x[i] += theta_amp * np.sin(2 * np.pi * rng.normal(6, 0.3) * t + rng.uniform(0, 6.28))
        x[i] += 2.0 * np.sin(2 * np.pi * 50 * t)                       # línea
        x[i] += rng.normal(0, 1.5, n)
    return x


def make_dataset(out_dir, n_mci=11, n_hc=13, minutes=10.2, seed=0):
    import pyedflib
    rng = np.random.default_rng(seed)
    for grp, n_sub, is_mci in (("MCI", n_mci, True), ("Normal", n_hc, False)):
        d = os.path.join(out_dir, grp)
        os.makedirs(d, exist_ok=True)
        for k in range(1, n_sub + 1):
            x = make_subject(rng, is_mci, minutes * 60)
            path = os.path.join(d, f"{grp.lower()}{k:02d}.edf")
            w = pyedflib.EdfWriter(path, len(C.CHANNELS), file_type=pyedflib.FILETYPE_EDFPLUS)
            hdr = [{"label": f"EEG {ch}-REF", "dimension": "uV", "sample_frequency": C.FS,
                    "physical_max": 500, "physical_min": -500,
                    "digital_max": 32767, "digital_min": -32768} for ch in C.CHANNELS]
            w.setSignalHeaders(hdr)
            w.writeSamples(list(np.clip(x, -499, 499)))
            w.close()
    print(f"EDF sintéticos en {out_dir}: {n_mci} MCI + {n_hc} HC, {minutes} min c/u")
