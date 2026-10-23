"""
Lectura de los EDF del dataset de Isfahan (misp.mui.ac.ir), preprocesado y
segmentación en ventanas de 10 s.

Pasos del paper (sección "Subjects and data pre-processing"):
  1. Pasa-banda 0.5-64 Hz
  2. Re-referencia al promedio común (CAR)
  3. Quitar ruido de línea (Cleanline)  -> aquí: notch 50/100 Hz
  4. ICA para EOG/EMG/ECG              -> aquí: opcional y automático (--ica)
  5. Inspección visual                  -> NO reproducible; ver noise_report()
  6. Excluir 3 HC ruidosos (no dicen cuáles) -> parámetro `exclude`
  7. Primeros 10 min, segmentos de 10 s sin traslape
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field

import numpy as np

from . import config as C


@dataclass
class Subject:
    sid: str
    label: int                 # 1 = MCI, 0 = HC
    path: str
    segments: np.ndarray = field(default=None, repr=False)   # (n_seg, n_ch, N) en µV


def _normalize_ch(name: str) -> str:
    n = name.upper()
    n = re.sub(r"^EEG\s*", "", n)
    n = re.split(r"[-_ ]", n.strip())[0]           # quita -REF, -LE, -A1 ...
    if n in C.CHANNEL_ALIASES:
        return C.CHANNEL_ALIASES[n]
    for ch in C.CHANNELS:
        if ch.upper() == n:
            return ch
    return name


def _label_from_path(path: str) -> int | None:
    p = path.lower()
    if "mci" in p:
        return 1
    if any(k in p for k in ("normal", "healthy", "control", "hc")):
        return 0
    return None


def _labels_from_states_xlsx(data_dir: str) -> dict:
    """La base de Isfahan (EEG_1.zip) trae 'states.xlsx' con columnas
    'file number' y 'status' (MCI / NORMAL). Si existe, se usa automáticamente."""
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower() == "states.xlsx":
                import openpyxl
                ws = openpyxl.load_workbook(os.path.join(root, f), data_only=True).worksheets[0]
                out = {}
                for row in list(ws.iter_rows(values_only=True))[1:]:
                    if row[0] is None or row[1] is None:
                        continue
                    status = str(row[1]).strip().upper()
                    out[f"{int(row[0])}.edf"] = 1 if status == "MCI" else 0
                print(f"Etiquetas leídas de {f}: {sum(out.values())} MCI, "
                      f"{len(out) - sum(out.values())} HC")
                return out
    return {}


def find_subjects(data_dir: str, labels_csv: str | None = None) -> list[Subject]:
    """Busca todos los .edf. La etiqueta sale del nombre de la carpeta/archivo
    ('MCI' vs 'Normal') o de un CSV con columnas file,label (1=MCI, 0=HC)."""
    explicit = {}
    if labels_csv:
        with open(labels_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                explicit[os.path.basename(row["file"])] = int(row["label"])
    else:
        explicit = _labels_from_states_xlsx(data_dir)
    subjects = []
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if not f.lower().endswith(".edf"):
                continue
            path = os.path.join(root, f)
            label = explicit.get(f, _label_from_path(os.path.relpath(path, data_dir)))
            if label is None:
                raise ValueError(f"No sé si {path} es MCI o HC. Usa --labels-csv.")
            grp = "MCI" if label == 1 else "HC"
            sid = f"{grp}_{os.path.splitext(f)[0]}"
            subjects.append(Subject(sid=sid, label=label, path=path))
    if not subjects:
        raise FileNotFoundError(f"No encontré archivos .edf en {data_dir}")
    subjects.sort(key=lambda s: (-s.label, s.sid))
    return subjects


def load_and_preprocess(path: str, duration_s: float = C.DURATION_S,
                        use_ica: bool = False, verbose: bool = False) -> np.ndarray:
    """Regresa la señal preprocesada (19, n_samples) en microvolts."""
    import mne
    mne.set_log_level("ERROR" if not verbose else "INFO")

    raw = mne.io.read_raw_edf(path, preload=True)
    mapping = {ch: _normalize_ch(ch) for ch in raw.ch_names}
    # si dos canales colapsan al mismo nombre, MNE se queja: renombramos solo los útiles
    used, ren = set(), {}
    for old, new in mapping.items():
        if new in C.CHANNELS and new not in used:
            ren[old] = new
            used.add(new)
    raw.rename_channels(ren)
    missing = [ch for ch in C.CHANNELS if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"{path}: faltan canales {missing}. Canales: {raw.ch_names}")
    raw.pick(C.CHANNELS)
    raw.set_channel_types({ch: "eeg" for ch in C.CHANNELS})

    if abs(raw.info["sfreq"] - C.FS) > 1e-6:
        raw.resample(C.FS)

    # nos quedamos con los primeros 10 min (+ un margen para los bordes del filtro)
    tmax = min(raw.times[-1], duration_s + 5)
    raw.crop(0, tmax)

    raw.filter(*C.BANDPASS, fir_design="firwin")
    raw.set_eeg_reference("average", projection=False)
    raw.notch_filter([C.LINE_FREQ, 2 * C.LINE_FREQ], fir_design="firwin")

    if use_ica:
        raw.set_montage("standard_1020", on_missing="ignore")
        ica = mne.preprocessing.ICA(n_components=15, random_state=0, max_iter="auto")
        ica.fit(raw.copy().filter(1.0, None))
        bads = []
        for eog_ch in ("Fp1", "Fp2"):
            idx, _ = ica.find_bads_eog(raw, ch_name=eog_ch)
            bads += idx
        try:
            idx, _ = ica.find_bads_muscle(raw)
            bads += idx
        except Exception:
            pass
        ica.exclude = sorted(set(bads))
        ica.apply(raw)

    data = raw.get_data()
    # MNE devuelve volts. Pero si el EDF trae vacío el campo de unidad física
    # (pasa en el dataset de Isfahan), MNE asume V aunque los valores estén en µV.
    # Un EEG real nunca tiene amplitud típica > 1 mV, así que lo usamos para detectarlo.
    if np.median(np.abs(data)) > 1e-3:
        if verbose:
            print("  unidades del EDF ya en µV (campo de dimensión vacío)")
    else:
        data = data * 1e6                            # V -> µV
    n_keep = int(duration_s * C.FS)
    if data.shape[1] < n_keep:
        print(f"  aviso: {os.path.basename(path)} dura {data.shape[1]/C.FS:.0f}s (<{duration_s}s)")
    return data[:, :n_keep]


def segment(data: np.ndarray, seg_s: float = C.SEGMENT_S) -> np.ndarray:
    """(ch, n) -> (n_seg, ch, N) con ventanas sin traslape."""
    N = int(seg_s * C.FS)
    n_seg = data.shape[1] // N
    return data[:, : n_seg * N].reshape(data.shape[0], n_seg, N).transpose(1, 0, 2)


def noise_report(subjects: list[Subject]) -> list[tuple[str, float, float]]:
    """Sustituto objetivo de la 'inspección visual': para cada sujeto calcula
    % de segmentos con amplitud pico > 100 µV y potencia relativa > 30 Hz.
    Sirve para decidir qué 3 HC excluir (el paper no dice cuáles fueron)."""
    from scipy.signal import welch
    rows = []
    for s in subjects:
        seg = s.segments
        peak = np.abs(seg).max(axis=2)                      # (n_seg, ch)
        pct_bad = 100 * np.mean(peak.max(axis=1) > 100)
        f, P = welch(seg, fs=C.FS, nperseg=C.FS * 2, axis=-1)
        hf = P[..., f > 30].sum(-1) / P[..., (f >= 0.5)].sum(-1)
        rows.append((s.sid, pct_bad, 100 * float(hf.mean())))
    return rows
