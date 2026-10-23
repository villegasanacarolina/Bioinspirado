"""
Inventario y comparación de EEG_1.zip (27 sujetos) y EEG_2.zip (34 pacientes)
de la base de Isfahan (https://misp.mui.ac.ir/en/eeg-data).

Qué responde:
  1. ¿Qué trae EEG_2? (archivos, canales, frecuencia, duración, etiquetas)
  2. ¿Hay sujetos repetidos entre EEG_1 y EEG_2? Se compara de dos formas:
     - hash MD5 (archivos idénticos byte por byte)
     - "huella" de la señal (el mismo registro aunque esté renombrado o recortado)

Uso en Google Colab:
  1. Sube EEG_1.zip y EEG_2.zip a tu Google Drive (p. ej. en una carpeta "EEG").
  2. En una celda:
        !pip install -q mne openpyxl
        from google.colab import drive; drive.mount('/content/drive')
        !python inventario_eeg.py /content/drive/MyDrive/EEG/EEG_1.zip /content/drive/MyDrive/EEG/EEG_2.zip
  3. Descarga los dos CSV que genera (pesan unos KB) para revisar los resultados.

Uso en tu compu: python inventario_eeg.py ruta/EEG_1.zip ruta/EEG_2.zip
"""
import hashlib
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import mne

mne.set_log_level("ERROR")
SEG_S = 60  # segundos usados para la huella


def extract(zpath, out):
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if not n.startswith("__MACOSX") and not n.endswith("/")]
        z.extractall(out, members=names)
    return names


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(folder, tag):
    rows, prints = [], {}
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            p = os.path.join(root, f)
            row = {"zip": tag, "archivo": os.path.relpath(p, folder),
                   "tamano_MB": round(os.path.getsize(p) / 2**20, 2), "md5": md5(p)}
            if f.lower().endswith(".edf"):
                try:
                    r = mne.io.read_raw_edf(p, preload=False)
                    row.update({
                        "n_canales": len(r.ch_names),
                        "canales": " ".join(r.ch_names),
                        "fs_Hz": r.info["sfreq"],
                        "duracion_min": round(r.times[-1] / 60, 2),
                        "fecha_registro": str(r.info.get("meas_date")),
                        "anotaciones": len(r.annotations),
                    })
                    fs = int(r.info["sfreq"])
                    x = r.get_data(start=0, stop=min(r.n_times, SEG_S * fs))
                    prints[(tag, row["archivo"])] = x
                except Exception as e:  # archivo dañado o formato distinto
                    row["error"] = str(e)[:120]
            elif f.lower().endswith((".xlsx", ".xls", ".csv", ".txt")):
                row["nota"] = "posible archivo de etiquetas"
            rows.append(row)
    return pd.DataFrame(rows), prints


def best_match(prints):
    """Para cada registro de EEG_2, busca el de EEG_1 más parecido.
    Compara el primer canal de los primeros 60 s, probando desfases de hasta 10 s
    por si un archivo está recortado al inicio."""
    a = {k: v for k, v in prints.items() if k[0] == "EEG_1"}
    b = {k: v for k, v in prints.items() if k[0] == "EEG_2"}
    out = []
    for kb, xb in b.items():
        best = (None, 0.0)
        sb = xb[0, : 20 * 256]
        sb = (sb - sb.mean()) / (sb.std() + 1e-12)
        for ka, xa in a.items():
            if xa.shape[0] != xb.shape[0]:
                continue
            ref = xa[0]
            for lag in range(0, min(10 * 256, len(ref) - len(sb)), 64):
                w = ref[lag: lag + len(sb)]
                c = float(np.dot(sb, (w - w.mean()) / (w.std() + 1e-12)) / len(sb))
                if c > best[1]:
                    best = (ka[1], c)
        out.append({"archivo_EEG_2": kb[1], "parecido_en_EEG_1": best[0],
                    "correlacion": round(best[1], 4),
                    "mismo_registro": best[1] > 0.99})
    return pd.DataFrame(out)


def main(z1, z2):
    base = "inventario_tmp"
    for tag, z in (("EEG_1", z1), ("EEG_2", z2)):
        print(f"Descomprimiendo {z} ...")
        extract(z, os.path.join(base, tag))
    d1, p1 = describe(os.path.join(base, "EEG_1"), "EEG_1")
    d2, p2 = describe(os.path.join(base, "EEG_2"), "EEG_2")
    inv = pd.concat([d1, d2], ignore_index=True)
    inv.to_csv("inventario_EEG.csv", index=False)

    same_md5 = set(d1["md5"]) & set(d2["md5"])
    match = best_match({**p1, **p2})
    match.to_csv("coincidencias_EEG1_EEG2.csv", index=False)

    print("\n===== RESUMEN =====")
    for tag, d in (("EEG_1", d1), ("EEG_2", d2)):
        edf = d[d["archivo"].str.lower().str.endswith(".edf")]
        print(f"{tag}: {len(d)} archivos, {len(edf)} EDF")
        if len(edf):
            print(f"   canales: {sorted(int(v) for v in edf['n_canales'].dropna().unique())} | "
                  f"fs: {sorted(float(v) for v in edf['fs_Hz'].dropna().unique())} Hz | "
                  f"duración: {edf['duracion_min'].min()}–{edf['duracion_min'].max()} min")
        otros = d[~d["archivo"].str.lower().str.endswith(".edf")]["archivo"].tolist()
        print(f"   otros archivos: {otros}")
    print(f"Archivos idénticos (MD5) entre EEG_1 y EEG_2: {len(same_md5)}")
    print(f"Registros de EEG_2 que son el mismo registro que uno de EEG_1 (corr > 0.99): "
          f"{int(match['mismo_registro'].sum())}")
    print("\nCSV generados: inventario_EEG.csv y coincidencias_EEG1_EEG2.csv")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Uso: python inventario_eeg.py EEG_1.zip EEG_2.zip")
    main(sys.argv[1], sys.argv[2])
