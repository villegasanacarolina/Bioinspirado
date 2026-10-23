"""
Prueba externa: desarrollar con EEG_1 y probar UNA sola vez con EEG_2.

  python -m mci_nsga2.externo extraer --data-dir RUTA\\EEG_1 --nombre EEG_1 --exclude-noisiest-hc 3
  python -m mci_nsga2.externo extraer --data-dir RUTA\\EEG_2 --nombre EEG_2
  python -m mci_nsga2.externo probar  --train externo\\EEG_1.npz --test externo\\EEG_2.npz

'extraer' guarda, por persona y segmento de 10 s:
  - características del paper (DWT + 10 medidas) con el preprocesado del paper, 19 canales
  - potencia relativa por banda con EEG limpio por ICA, 15 canales (método mejorado)
En EEG_1 se excluyen los 3 sanos más ruidosos para replicar al paper (24 personas).
En EEG_2 NO se excluye a nadie: el grupo de prueba no se depura después de verlo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from . import config as C


def cmd_extraer(a):
    from .data_loader import find_subjects, load_and_preprocess, noise_report, segment
    from .features import segment_features
    from .limpieza import limpiar_ica, potencia_relativa

    subjects = find_subjects(a.data_dir, a.labels_csv)
    print(f"{a.nombre}: {len(subjects)} personas "
          f"({sum(s.label for s in subjects)} MCI, {sum(1 - s.label for s in subjects)} sanos)")
    for s in subjects:
        s.segments = segment(load_and_preprocess(s.path))
    if a.exclude_noisiest_hc:
        rep = sorted([r for r in noise_report(subjects) if r[0].startswith("HC_")],
                     key=lambda r: (r[1], r[2]), reverse=True)
        excl = {r[0] for r in rep[: a.exclude_noisiest_hc]}
        print("Excluidos (más ruidosos):", sorted(excl))
        subjects = [s for s in subjects if s.sid not in excl]

    paper, rel, y, groups, sids = [], [], [], [], []
    t0 = time.time()
    for k, s in enumerate(subjects):
        F = np.stack([segment_features(seg, "DWT") for seg in s.segments])   # (60, 19, 6, 10)
        Xc, _ = limpiar_ica(s.path)
        R = potencia_relativa(Xc)                                             # (60, 60)
        n = min(len(F), len(R))
        paper.append(F[:n]); rel.append(R[:n])
        y += [s.label] * n; groups += [k] * n; sids.append(f"{a.nombre}_{s.sid}")
        print(f"  {s.sid:12s} {n} segmentos  ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"{a.nombre}.npz")
    np.savez_compressed(path, paper=np.concatenate(paper), relpow=np.concatenate(rel),
                        y=np.array(y), groups=np.array(groups), sids=np.array(sids))
    print(f"-> {path}")


def _person_means(R, groups):
    subj = np.unique(groups)
    return np.array([R[groups == s].mean(0) for s in subj])


def _report(name, pred, y, extra=""):
    from scipy.stats import binomtest
    from .persona import balanced
    k = int((pred == y).sum()); n = len(y)
    ci = binomtest(k, n).proportion_ci(method="exact")
    bal = balanced(pred, y)
    rng = np.random.default_rng(0)                 # p: etiquetas de prueba revueltas, predicciones fijas
    null = np.array([balanced(pred, rng.permutation(y)) for _ in range(10000)])
    p = (1 + (null >= bal).sum()) / (1 + len(null))
    sens = 100 * np.mean(pred[y == 1] == 1); spec = 100 * np.mean(pred[y == 0] == 0)
    print(f"{name:34s} personas bien {k:2d}/{n}  exactitud {100*k/n:5.1f}% (IC95% {100*ci.low:4.1f}-{100*ci.high:4.1f})"
          f"  balanceada {bal:5.1f}% (p={p:.3f})  sens {sens:5.1f}  espec {spec:5.1f}  {extra}", flush=True)
    return {"metodo": name, "bien": k, "n": n, "exactitud": 100 * k / n, "ic_bajo": 100 * ci.low,
            "ic_alto": 100 * ci.high, "balanceada": bal, "p": p, "sens": sens, "espec": spec, "extra": extra,
            "pred": [int(v) for v in pred], "y": [int(v) for v in y]}   # añadido: predicción por persona (no cambia ningún cálculo)


def cmd_probar(a):
    from .classify import make_classifier, _zscore
    from .limpieza import CH15
    from .nsga2 import NSGA2, SelectionProblem
    from . import persona as P

    tr = dict(np.load(a.train)); te = dict(np.load(a.test))
    ytr_s, gtr = tr["y"], tr["groups"]; yte_s, gte = te["y"], te["groups"]
    Ytr = np.array([ytr_s[gtr == s][0] for s in np.unique(gtr)])
    Yte = np.array([yte_s[gte == s][0] for s in np.unique(gte)])
    print(f"Entrenamiento: {len(Ytr)} personas ({Ytr.sum()} MCI) | Prueba: {len(Yte)} personas ({Yte.sum()} MCI)")
    print("Los métodos se fijaron ANTES de ver la base de prueba. La prueba se usa una sola vez.\n")
    rows = []

    # ---- R: réplica del paper (DWT-ThEn-LDA, segmentos, 19 canales, NSGA-II 200x50, regla de mayor CA)
    m = list(C.MEASURES).index("ThEn")
    Xtr = tr["paper"][..., m].reshape(len(ytr_s), -1); Xte = te["paper"][..., m].reshape(len(yte_s), -1)
    prob = SelectionProblem(Xtr, ytr_s, gtr, 6, "DA", C.PARAM_BOUNDS["DA"], mode="channels", seed=0)
    ga = NSGA2(prob, pop_size=a.pop, n_gen=a.gen, seed=0, n_jobs=a.n_jobs, verbose=False).run()
    best = max(ga.pareto_front(), key=lambda s: (s["acc"], -s["n_selected"]))
    cc = prob.columns(np.array(best["mask"]))
    Xa, Xb = _zscore(Xtr[:, cc], Xte[:, cc])
    seg_pred = make_classifier("DA", best["params"]).fit(Xa, ytr_s).predict(Xb)
    pred = np.array([int(seg_pred[gte == s].mean() > 0.5) for s in np.unique(gte)])
    chans = [C.CHANNELS[i] for i in np.flatnonzero(best["mask"])]
    print(f"[Réplica del paper] CA interna que reportaría el paper (LOSO en entrenamiento): {best['acc']:.1f}% "
          f"con {chans}")
    rows.append(_report("R  réplica del paper", pred, Yte,
                        f"CA interna {best['acc']:.1f}% | CA por segmento en prueba {100*np.mean(seg_pred == yte_s):.1f}% | {chans}"))

    # ---- métodos a nivel persona sobre EEG limpio (potencia relativa)
    Xp_tr = _person_means(tr["relpow"], gtr); Xp_te = _person_means(te["relpow"], gte)
    full = np.ones(len(CH15), np.int64)
    rows.append(_report("M0 sin selección (15 electrodos)", P.fit_predict(full, Xp_tr, Ytr, Xp_te), Yte))
    m4 = P.select_M4(Xp_tr, Ytr)
    rows.append(_report("M4 zonas exhaustivo", P.fit_predict(m4, Xp_tr, Ytr, Xp_te), Yte,
                        str([CH15[i] for i in np.flatnonzero(m4)])))
    for name, sel in (("M1 NSGA-II regla del paper", P.select_M1), ("M6 PSO binario parsimonioso", P.select_M6)):
        for seed in range(a.seeds):
            mk = sel(Xp_tr, Ytr, seed)
            rows.append(_report(f"{name} (semilla {seed})", P.fit_predict(mk, Xp_tr, Ytr, Xp_te), Yte,
                                str([CH15[i] for i in np.flatnonzero(mk)])))

    # ---- diagnóstico: ¿se distinguen las dos bases solo por su EEG? (efecto de base)
    Xb_all = np.vstack([Xp_tr, Xp_te]); yb = np.r_[np.zeros(len(Xp_tr), int), np.ones(len(Xp_te), int)]
    pb = np.array([P.fit_predict(np.ones(len(CH15), np.int64), Xb_all[np.arange(len(yb)) != k],
                                 yb[np.arange(len(yb)) != k], Xb_all[[k]])[0] for k in range(len(yb))])
    print(f"\n[Efecto de base] Exactitud balanceada para distinguir EEG de entrenamiento vs prueba: "
          f"{P.balanced(pb, yb):.1f}% (50% = indistinguibles)")
    os.makedirs(a.results_dir, exist_ok=True)
    out = os.path.join(a.results_dir, "prueba_externa.json")
    with open(out, "w") as fh:
        json.dump({"resultados": rows, "efecto_base": P.balanced(pb, yb)}, fh, indent=1)
    print(f"-> {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="mci_nsga2.externo")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extraer")
    e.add_argument("--data-dir", required=True)
    e.add_argument("--nombre", required=True)
    e.add_argument("--labels-csv")
    e.add_argument("--out-dir", default="externo")
    e.add_argument("--exclude-noisiest-hc", type=int, default=0)
    e.set_defaults(fn=cmd_extraer)
    t = sub.add_parser("probar")
    t.add_argument("--train", required=True)
    t.add_argument("--test", required=True)
    t.add_argument("--results-dir", default="results")
    t.add_argument("--pop", type=int, default=200, help="población NSGA-II de la réplica (paper: 200)")
    t.add_argument("--gen", type=int, default=50, help="generaciones NSGA-II de la réplica (paper: 50)")
    t.add_argument("--seeds", type=int, default=5, help="semillas para M1 y M6")
    t.add_argument("--n-jobs", type=int, default=-1)
    t.set_defaults(fn=cmd_probar)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
