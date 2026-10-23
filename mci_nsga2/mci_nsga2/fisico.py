"""
Análisis secundario preregistrado: MONTAJE FÍSICO.

Motivación: los métodos M0 y M4 calculan sus características después de una
referencia promedio y de una ICA ajustadas con todos los electrodos. Por eso la
señal de "un" electrodo contiene información de los demás, y un dispositivo que
solo tuviera esos electrodos no podría calcularla. Aquí se repiten M0 y M4
cambiando ÚNICAMENTE el montaje:

  igual que limpieza.py           primeros 10 min, segmentos de 10 s, mismos 15 electrodos
                                  (sin Fp1, Fp2, F7, F8), mismas 4 bandas, mismo Welch,
                                  log de potencia relativa a 1-30 Hz, promedio por persona,
                                  mismo modelo (LDA con contracción) y misma búsqueda por zonas
  distinto (montaje físico)       sin referencia promedio (se conserva la referencia original
                                  del registro) y sin ICA: solo operaciones de un canal
                                  (pasa-banda 0.5-64 Hz y notch 50/100 Hz)

Uso (desde la carpeta mci_nsga2):
  python -m mci_nsga2.fisico extraer --data-dir RUTA/EEG_2 --nombre EEG_2
  python -m mci_nsga2.fisico extraer --data-dir RUTA/EEG_1 --nombre EEG_1 --igual-que externo/EEG_1.npz
  python -m mci_nsga2.fisico probar  --train externo/EEG_1_fisico.npz --test externo/EEG_2_fisico.npz

'--igual-que' toma exactamente las mismas personas que el archivo de 'externo extraer'
(en EEG_1 se excluyen los 3 sanos más ruidosos; en EEG_2 no se excluye a nadie).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import config as C
from .data_loader import _normalize_ch, find_subjects
from .limpieza import CH15, potencia_relativa


def leer_fisico(path: str, duration_s: float = C.DURATION_S) -> np.ndarray:
    """EEG de los 15 electrodos en µV con procesamiento canal por canal (sin CAR ni ICA)."""
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
    if np.median(np.abs(raw._data)) > 1e-3:          # EDF con µV declarados como V (igual que limpieza.py)
        raw._data *= 1e-6
    raw.crop(0, min(raw.times[-1], duration_s + 5))
    raw.filter(*C.BANDPASS, fir_design="firwin")       # filtro por canal
    raw.notch_filter([C.LINE_FREQ, 2 * C.LINE_FREQ], fir_design="firwin")   # filtro por canal
    raw.pick(CH15)                                      # sin referencia promedio, sin ICA
    n = int(duration_s * C.FS)
    return raw.get_data()[:, :n] * 1e6


def cmd_extraer(a):
    subjects = find_subjects(a.data_dir, a.labels_csv)
    if a.igual_que:
        keep = {_persona(x, a.igual_que) for x in np.load(a.igual_que)["sids"]}
        subjects = [s for s in subjects if s.sid in keep]
        print(f"Mismas personas que {a.igual_que}: {len(subjects)}")
    print(f"{a.nombre}: {len(subjects)} personas ({sum(s.label for s in subjects)} MCI, "
          f"{sum(1 - s.label for s in subjects)} sanos) | montaje físico")
    rel, y, groups, sids = [], [], [], []
    t0 = time.time()
    for k, s in enumerate(subjects):
        R = potencia_relativa(leer_fisico(s.path))
        rel.append(R); y += [s.label] * len(R); groups += [k] * len(R); sids.append(f"{a.nombre}_{s.sid}")
        print(f"  {s.sid:12s} {len(R)} segmentos  ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"{a.nombre}_fisico.npz")
    np.savez_compressed(path, relpow=np.concatenate(rel), y=np.array(y), groups=np.array(groups), sids=np.array(sids))
    print(f"-> {path}")


MARGEN = -10.0      # margen de no inferioridad preregistrado (puntos de exactitud balanceada)
N_BOOT = 10000


def _persona(sid, path):
    """'EEG_1_MCI_11' -> 'MCI_11' (quita el prefijo con el nombre de la base, tomado del archivo)."""
    base = os.path.splitext(os.path.basename(path))[0].removesuffix("_fisico") + "_"
    return str(sid).removeprefix(base)


def _cargar(path):
    from .externo import _person_means
    d = dict(np.load(path))
    subj = np.unique(d["groups"])
    Y = np.array([d["y"][d["groups"] == s][0] for s in subj])
    sids = [_persona(x, path) for x in d["sids"]]
    return _person_means(d["relpow"], d["groups"]), Y, sids


def _ajustar(mask, Xtr, Ytr, Xte):
    from . import persona as P
    c = P.cols(mask)
    m = P.model().fit(Xtr[:, c], Ytr)
    return m.predict(Xte[:, c]), m.decision_function(Xte[:, c])


def _boot_diferencia(pa, pb, y, seed=0):
    """Bootstrap de personas estratificado por clase: exactitud balanceada de A menos la de B (pareado)."""
    from .persona import balanced
    rng = np.random.default_rng(seed)
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        d[b] = balanced(pa[idx], y[idx]) - balanced(pb[idx], y[idx])
    return d


def cmd_probar(a):
    from sklearn.metrics import roc_auc_score
    from . import persona as P
    from .externo import _report

    Xtr, Ytr, _ = _cargar(a.train); Xte, Yte, sid_te = _cargar(a.test)
    print(f"Montaje físico | entrenamiento: {len(Ytr)} personas | prueba: {len(Yte)} personas")
    full = np.ones(len(CH15), np.int64)
    metodos = {"M0": full, "M4": P.select_M4(Xtr, Ytr)}
    ica = None
    if a.ica_train and a.ica_test:            # versión ICA de Jessi, recalculada sin cambios (M0 y M4 son deterministas)
        Itr, IYtr, _ = _cargar(a.ica_train); Ite, IYte, isid_te = _cargar(a.ica_test)
        assert np.array_equal(IYtr, Ytr) and np.array_equal(IYte, Yte), "las personas no coinciden"
        assert isid_te == sid_te, "las personas de prueba no están en el mismo orden"
        ica = {"M0": full, "M4": P.select_M4(Itr, IYtr)}
    rows, comparaciones = [], []
    for k, mask in metodos.items():
        pred, score = _ajustar(mask, Xtr, Ytr, Xte)
        auc = roc_auc_score(Yte, score)
        r = _report(f"{k}-físico", pred, Yte, f"AUC {auc:.3f} | {[CH15[i] for i in np.flatnonzero(mask)]}")
        r.update(auc=auc, electrodos=[CH15[i] for i in np.flatnonzero(mask)], pred=pred.tolist())
        rows.append(r)
        if ica is not None:
            ipred, iscore = _ajustar(ica[k], Itr, IYtr, Ite)
            d_obs = P.balanced(pred, Yte) - P.balanced(ipred, Yte)
            d = _boot_diferencia(pred, ipred, Yte)
            p = (1 + (d <= MARGEN).sum()) / (1 + N_BOOT)     # H0: diferencia <= margen
            comparaciones.append({"metodo": k, "bal_fisico": P.balanced(pred, Yte), "bal_ica": P.balanced(ipred, Yte),
                                  "auc_fisico": auc, "auc_ica": roc_auc_score(Yte, iscore),
                                  "diferencia": d_obs, "ic90_bajo": float(np.percentile(d, 5)),
                                  "ic90_alto": float(np.percentile(d, 95)), "p_no_inferioridad": p,
                                  "electrodos_ica": [CH15[i] for i in np.flatnonzero(ica[k])]})
    if comparaciones:                         # Holm sobre las 2 comparaciones (M0, M4)
        orden = sorted(range(len(comparaciones)), key=lambda i: comparaciones[i]["p_no_inferioridad"])
        acum = 0.0
        for rango, i in enumerate(orden):
            acum = max(acum, min(1.0, (len(orden) - rango) * comparaciones[i]["p_no_inferioridad"]))
            comparaciones[i]["p_holm"] = acum
        print(f"\nH4: físico no inferior a ICA (margen {MARGEN:+.0f} puntos de exactitud balanceada)")
        for c in comparaciones:
            print(f"  {c['metodo']}: físico {c['bal_fisico']:.1f}% vs ICA {c['bal_ica']:.1f}% | diferencia "
                  f"{c['diferencia']:+.1f} (IC90% {c['ic90_bajo']:+.1f} a {c['ic90_alto']:+.1f}) | "
                  f"p={c['p_no_inferioridad']:.3f}, Holm {c['p_holm']:.3f} -> "
                  f"{'NO INFERIOR' if c['p_holm'] < 0.05 else 'no se demuestra no inferioridad'}")
    os.makedirs(a.results_dir, exist_ok=True)
    out = os.path.join(a.results_dir, "prueba_externa_fisico.json")
    with open(out, "w") as fh:
        json.dump({"margen": MARGEN, "personas_prueba": sid_te, "y_prueba": Yte.tolist(),
                   "resultados": rows, "no_inferioridad": comparaciones}, fh, indent=1)
    print(f"-> {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="mci_nsga2.fisico")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extraer")
    e.add_argument("--data-dir", required=True); e.add_argument("--nombre", required=True)
    e.add_argument("--labels-csv"); e.add_argument("--out-dir", default="externo")
    e.add_argument("--igual-que", help="npz de 'externo extraer' cuyas personas se conservan")
    t = sub.add_parser("probar")
    t.add_argument("--train", required=True); t.add_argument("--test", required=True)
    t.add_argument("--ica-train", help="npz de 'externo extraer' (entrenamiento) para la comparación pareada H4")
    t.add_argument("--ica-test", help="npz de 'externo extraer' (prueba) para la comparación pareada H4")
    t.add_argument("--results-dir", default="results")
    a = p.parse_args(argv)
    {"extraer": cmd_extraer, "probar": cmd_probar}[a.cmd](a)


if __name__ == "__main__":
    main()
