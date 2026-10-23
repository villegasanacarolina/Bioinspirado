r"""
Segunda verificación de duplicados entre EEG_1 y EEG_2, robusta a recortes.

Prueba A (decisiva) - BÚSQUEDA DE FRAGMENTO:
  De cada registro de EEG_2 se toman 3 fragmentos de 30 s (al 25%, 50% y 75% del
  registro) del canal Cz y se buscan en CUALQUIER posición de cada registro de
  EEG_1 (correlación cruzada normalizada). Si es la misma grabación, aunque esté
  recortada o tenga otra duración, el fragmento aparece con correlación ~1.
Prueba B (complementaria) - HUELLA ESPECTRAL:
  Espectro promedio de los 19 canales (1-40 Hz). Sirve como referencia de qué
  tan parecidas son personas distintas; por sí sola NO es decisiva.

Uso:  python huella_espectral.py RUTA/EEG_1.zip RUTA\EEG_2.zip
Salida: duplicados_EEG1_EEG2.csv
"""
import os
import sys
import zipfile
import tempfile

import numpy as np
import pandas as pd
import mne
from scipy.signal import welch, fftconvolve, decimate

mne.set_log_level("ERROR")
CH = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
      "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
FS_LOW = 64          # para la búsqueda se baja la frecuencia de muestreo a 64 Hz
FRAG_S = 30
UMBRAL = 0.95        # correlación de fragmento a partir de la cual se considera la misma grabación


def leer(path):
    raw = mne.io.read_raw_edf(path, preload=True)
    ren = {}
    for c in raw.ch_names:
        n = c.upper().replace("EEG", "").strip().split("-")[0].strip()
        for t in CH:
            if t.upper() == n and t not in ren.values():
                ren[c] = t
    raw.rename_channels(ren)
    raw.pick([c for c in CH if c in raw.ch_names])
    x = raw.get_data()
    fs = raw.info["sfreq"]
    # huella espectral
    f, P = welch(x, fs=fs, nperseg=1024, axis=-1)
    sel = (f >= 1) & (f <= 40)
    spec = np.log10(P[:, sel] + 1e-30).ravel()
    # señal Cz a 64 Hz para la búsqueda de fragmentos
    cz = x[raw.ch_names.index("Cz")]
    cz = decimate(cz, int(round(fs / FS_LOW)), zero_phase=True)
    return spec, cz


def leer_zip(zpath, tmp):
    out = {}
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if n.lower().endswith(".edf") and not n.startswith("__MACOSX"):
                p = z.extract(n, tmp)
                out[os.path.basename(n)] = leer(p)
                os.remove(p)
                print("   leído", os.path.basename(zpath), os.path.basename(n), flush=True)
    return out


def max_corr_fragmento(frag, señal):
    """Correlación de Pearson máxima entre 'frag' y cualquier ventana de 'señal'."""
    L = len(frag)
    if len(señal) < L:
        return 0.0
    f = (frag - frag.mean()) / (frag.std() * L + 1e-30)
    num = fftconvolve(señal, f[::-1], mode="valid")
    c1 = np.concatenate([[0], np.cumsum(señal)])
    c2 = np.concatenate([[0], np.cumsum(señal ** 2)])
    s = c1[L:] - c1[:-L]; s2 = c2[L:] - c2[:-L]
    sd = np.sqrt(np.maximum(s2 / L - (s / L) ** 2, 1e-30))
    return float(np.max(num / sd))


def orden(k):
    b = os.path.splitext(k)[0]
    return (0, int(b)) if b.isdigit() else (1, b)


def main(z1, z2):
    with tempfile.TemporaryDirectory() as tmp:
        d1 = leer_zip(z1, tmp)
        d2 = leer_zip(z2, tmp)
    k1, k2 = sorted(d1, key=orden), sorted(d2, key=orden)

    # B) huella espectral
    A = np.array([d1[k][0] for k in k1]); B = np.array([d2[k][0] for k in k2])
    A = (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)
    B = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True)
    R = B @ A.T / A.shape[1]

    # A) búsqueda de fragmentos
    L = FRAG_S * FS_LOW
    rows = []
    for i, k in enumerate(k2):
        cz2 = d2[k][1]
        frags = [cz2[int(p * len(cz2)): int(p * len(cz2)) + L] for p in (0.25, 0.5, 0.75)]
        frags = [fr for fr in frags if len(fr) == L]
        mejor, mejor_k = 0.0, None
        for kk in k1:
            c = np.median([max_corr_fragmento(fr, d1[kk][1]) for fr in frags])
            if c > mejor:
                mejor, mejor_k = c, kk
        j = int(np.argmax(R[i]))
        rows.append({"archivo_EEG_2": k,
                     "fragmento_mas_parecido_EEG_1": mejor_k, "corr_fragmento": round(mejor, 4),
                     "espectro_mas_parecido_EEG_1": k1[j], "corr_espectro": round(float(R[i, j]), 4),
                     "misma_grabacion": mejor > UMBRAL})
        print(f"   {k}: fragmento {mejor:.3f} ({mejor_k})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("duplicados_EEG1_EEG2.csv", index=False)

    R11 = A @ A.T / A.shape[1]
    otros = R11[~np.eye(len(A), dtype=bool)]
    print("\n===== RESUMEN =====")
    print(f"A) Búsqueda de fragmento: correlación máxima EEG_2 vs EEG_1 = {df.corr_fragmento.max():.4f} "
          f"(umbral de 'misma grabación' = {UMBRAL})")
    print(f"   Registros de EEG_2 que son la misma grabación que uno de EEG_1: {int(df.misma_grabacion.sum())}")
    if df.misma_grabacion.any():
        print(df[df.misma_grabacion][["archivo_EEG_2", "fragmento_mas_parecido_EEG_1", "corr_fragmento"]]
              .to_string(index=False))
    print(f"B) Huella espectral (referencia): entre personas distintas de EEG_1 máx = {otros.max():.3f}, "
          f"promedio = {otros.mean():.3f}; EEG_2 vs EEG_1 máx = {df.corr_espectro.max():.3f}")
    print("\nDetalle en duplicados_EEG1_EEG2.csv")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Uso: python huella_espectral.py EEG_1.zip EEG_2.zip")
    main(sys.argv[1], sys.argv[2])
