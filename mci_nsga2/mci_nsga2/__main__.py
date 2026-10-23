"""
Línea de comandos.  Uso:  python -m mci_nsga2 <comando> [opciones]

  extract  Lee EDF -> preprocesa -> segmenta -> DWT y VMD -> 10 medidas (se guarda en .npz)
  exp1     Experimento I: todas las características, LOSO (y 10-fold opcional)  ~ Tablas 2 y 3
  exp2     Experimento II: NSGA-II para seleccionar canales                       ~ Tablas 4,5,8,9
  exp3     Experimento III: NSGA-II para seleccionar características              ~ Tablas 6,7,10,11,13
  synth    Genera EDF sintéticos para probar el pipeline
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time

import numpy as np

from . import config as C
from . import paper_reference as REF


# ------------------------------------------------------------------ extract
def cmd_extract(a):
    from .data_loader import find_subjects, load_and_preprocess, noise_report, segment
    from .features import build_feature_tensor, save_tensor

    subjects = find_subjects(a.data_dir, a.labels_csv)
    print(f"{len(subjects)} sujetos: {sum(s.label for s in subjects)} MCI, "
          f"{sum(1 - s.label for s in subjects)} HC")
    for s in subjects:
        t0 = time.time()
        s.segments = segment(load_and_preprocess(s.path, a.duration, a.ica))
        print(f"  {s.sid:25s} {s.segments.shape[0]:3d} segmentos  ({time.time() - t0:.1f}s)")

    # reporte de ruido (sustituto de la inspección visual)
    os.makedirs(a.out_dir, exist_ok=True)
    rep = noise_report(subjects)
    with open(os.path.join(a.out_dir, "noise_report.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["sid", "pct_segmentos_>100uV", "pct_potencia_>30Hz"])
        w.writerows(rep)

    excl = set()
    for pat in a.exclude or []:
        excl |= {s.sid for s in subjects if pat in s.sid}
    if a.exclude_noisiest_hc:
        hc = [r for r in rep if r[0].startswith("HC_")]
        hc.sort(key=lambda r: (r[1], r[2]), reverse=True)
        excl |= {r[0] for r in hc[: a.exclude_noisiest_hc]}
    if excl:
        print("Excluidos:", sorted(excl))
    subjects = [s for s in subjects if s.sid not in excl]
    print(f"Quedan {len(subjects)} sujetos (el paper usa 24: 11 MCI + 13 HC)")

    for method in a.methods:
        t0 = time.time()
        print(f"Extrayendo características {method} ...")
        d = build_feature_tensor(subjects, method, n_jobs=a.n_jobs, verbose=0)
        path = os.path.join(a.out_dir, f"{method}.npz")
        save_tensor(d, path)
        print(f"  {path}: F{d['F'].shape}  ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ helpers
def _load(a, method):
    from .features import load_tensor
    return load_tensor(os.path.join(a.features_dir, f"{method}.npz"))


def _expand(values, universe):
    return universe if (not values or values == ["all"]) else values


def _param_grid(clf):
    return list(itertools.product(*[range(lo, hi + 1) for lo, hi in C.PARAM_BOUNDS[clf]]))


def _fmt(v):
    return "  -  " if v is None else f"{v:6.2f}"


# ------------------------------------------------------------------ exp1
def cmd_exp1(a):
    from joblib import Parallel, delayed
    from .classify import kfold, loso
    from .features import feature_matrix

    rows = []
    for method in _expand(a.methods, ["VMD", "DWT"]):
        d = _load(a, method)
        for measure in _expand(a.measures, C.MEASURES):
            X, _, _ = feature_matrix(d, measure)
            y, g = d["y"], d["groups"]
            for clf in _expand(a.classifiers, C.CLASSIFIERS):
                params = C.DEFAULT_PARAMS[clf]
                acc = loso(X, y, g, clf, params, standardize=not a.no_standardize)
                row = {"method": method, "measure": measure, "clf": clf,
                       "params": params, "loso": acc,
                       "paper_loso": REF.EXP1_LOSO.get((method, measure, clf))}
                if a.tune:                       # mejor hiperparámetro por LOSO
                    grid = _param_grid(clf)
                    res = Parallel(n_jobs=a.n_jobs)(delayed(loso)(
                        X, y, g, clf, p, not a.no_standardize) for p in grid)
                    i = int(np.argmax(res))
                    row["loso_tuned"], row["params_tuned"] = res[i], list(grid[i])
                if a.kfold:
                    row["kfold10"] = kfold(X, y, clf, params, standardize=not a.no_standardize)
                    row["paper_kfold10"] = REF.EXP1_KFOLD.get((method, measure, clf))
                rows.append(row)
                extra = f"  tuned={row['loso_tuned']:6.2f}{row['params_tuned']}" if a.tune else ""
                extra += f"  10fold={row['kfold10']:6.2f}" if a.kfold else ""
                print(f"{method}-{measure:5s}-{clf:3s}  LOSO={acc:6.2f}  "
                      f"paper={_fmt(row['paper_loso'])}{extra}", flush=True)
    _save(a, "exp1", rows)
    _table_exp1(rows)


def _table_exp1(rows):
    """Imprime en formato Tabla 2 (filas = medida, columnas = clasificador)."""
    for method in sorted({r["method"] for r in rows}):
        print(f"\n== {method}: CA LOSO (%) con todos los canales ==")
        clfs = [c for c in C.CLASSIFIERS if any(r["clf"] == c for r in rows)]
        print("Medida  " + "".join(f"{c:>8s}" for c in clfs))
        for m in C.MEASURES:
            rr = {r["clf"]: r for r in rows if r["method"] == method and r["measure"] == m}
            if rr:
                print(f"{m:7s} " + "".join(f"{rr[c]['loso']:8.2f}" if c in rr else "      - "
                                           for c in clfs))


# ------------------------------------------------------------------ exp2 / exp3
def _run_nsga(a, mode):
    from .classify import loso
    from .features import feature_matrix
    from .nsga2 import NSGA2, SelectionProblem

    default = C.NSGA_CHANNELS if mode == "channels" else C.NSGA_FEATURES
    pop = a.pop or default["pop_size"]
    gens = a.gen or default["n_gen"]
    ref_table = REF.EXP2 if mode == "channels" else REF.EXP3
    all_rows = []
    for method in _expand(a.methods, ["VMD", "DWT"]):
        d = _load(a, method)
        for measure in _expand(a.measures, C.MEASURES):
            X, names, nb = feature_matrix(d, measure)
            y, g = d["y"], d["groups"]
            for clf in _expand(a.classifiers, C.CLASSIFIERS):
                tag = f"{method}-{measure}-{clf}"
                print(f"\n### {tag}  ({mode}, pob={pop}, gen={gens}, semilla={a.seed})")
                prob = SelectionProblem(X, y, g, nb, clf, C.PARAM_BOUNDS[clf], mode=mode,
                                        standardize=not a.no_standardize, seed=a.seed)
                ga = NSGA2(prob, pop_size=pop, n_gen=gens, seed=a.seed, n_jobs=a.n_jobs).run()
                front = ga.pareto_front()

                # métricas completas (Tabla 13) para la solución de mayor CA
                best = max(front, key=lambda s: (s["acc"], -s["n_selected"]))
                cols = prob.columns(np.array(best["mask"]))
                full = loso(X[:, cols], y, g, clf, best["params"],
                            standardize=not a.no_standardize, full=True)
                for s in front:
                    m = np.array(s["mask"])
                    if mode == "channels":
                        s["selected"] = [C.CHANNELS[i] for i in np.flatnonzero(m)]
                    else:
                        s["selected"] = [names[i] for i in np.flatnonzero(m)]
                        s["n_channels"] = len({n.split(":")[0] for n in s["selected"]})
                print(f"  Frente de Pareto {tag}:")
                for s in front:
                    nch = f" ({s['n_channels']} canales)" if mode == "features" else ""
                    print(f"    {s['n_selected']:3d} sel{nch}: CA={s['acc']:6.2f}%  "
                          f"params={s['params']}  {s['selected'] if s['n_selected'] <= 20 else ''}")
                print(f"  Mejor: CA={full['acc']:.2f} sens={full['sens']:.2f} "
                      f"spec={full['spec']:.2f} prec={full['prec']:.2f} F={full['f1']:.2f}")
                ref = ref_table.get((method, measure, clf))
                if ref is not None:
                    print(f"  Paper: {ref}")
                all_rows.append({"method": method, "measure": measure, "clf": clf,
                                 "mode": mode, "pop": pop, "gen": gens, "seed": a.seed,
                                 "front": front, "best_full_metrics": full,
                                 "history": ga.history, "paper": ref if ref is None else
                                 (ref if isinstance(ref, tuple) else {str(k): v for k, v in ref.items()})})
                _save(a, f"exp{'2' if mode == 'channels' else '3'}_{tag}", all_rows[-1])
    return all_rows


def cmd_exp2(a):
    _run_nsga(a, "channels")


def cmd_exp3(a):
    _run_nsga(a, "features")


def _save(a, name, obj):
    os.makedirs(a.results_dir, exist_ok=True)
    path = os.path.join(a.results_dir, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"  -> {path}")


# ------------------------------------------------------------------ synth
def cmd_synth(a):
    from .synthetic import make_dataset
    make_dataset(a.out_dir, n_mci=a.n_mci, n_hc=a.n_hc, minutes=a.minutes, seed=a.seed)


# ------------------------------------------------------------------ parser
def main(argv=None):
    p = argparse.ArgumentParser(prog="mci_nsga2", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--data-dir", required=True)
    e.add_argument("--out-dir", default="features")
    e.add_argument("--labels-csv")
    e.add_argument("--methods", nargs="+", default=["DWT", "VMD"])
    e.add_argument("--duration", type=float, default=C.DURATION_S)
    e.add_argument("--ica", action="store_true", help="ICA automática (MNE) para artefactos")
    e.add_argument("--exclude", nargs="*", help="sujetos a excluir (subcadena del id)")
    e.add_argument("--exclude-noisiest-hc", type=int, default=0,
                   help="excluye los N HC más ruidosos (el paper excluyó 3)")
    e.add_argument("--n-jobs", type=int, default=-1)
    e.set_defaults(fn=cmd_extract)

    def common(sp):
        sp.add_argument("--features-dir", default="features")
        sp.add_argument("--results-dir", default="results")
        sp.add_argument("--methods", nargs="*")
        sp.add_argument("--measures", nargs="*")
        sp.add_argument("--classifiers", nargs="*")
        sp.add_argument("--no-standardize", action="store_true")
        sp.add_argument("--n-jobs", type=int, default=-1)
        sp.add_argument("--seed", type=int, default=0)

    x1 = sub.add_parser("exp1"); common(x1)
    x1.add_argument("--tune", action="store_true", help="busca el mejor hiperparámetro con LOSO")
    x1.add_argument("--kfold", action="store_true", help="también 10-fold (Tabla 3)")
    x1.set_defaults(fn=cmd_exp1)

    for name, fn in (("exp2", cmd_exp2), ("exp3", cmd_exp3)):
        x = sub.add_parser(name); common(x)
        x.add_argument("--pop", type=int, help="tamaño de población (paper: 200 / 1000)")
        x.add_argument("--gen", type=int, help="generaciones (paper: 50 / 100)")
        x.set_defaults(fn=fn)

    s = sub.add_parser("synth")
    s.add_argument("--out-dir", default="synthetic_data")
    s.add_argument("--n-mci", type=int, default=11)
    s.add_argument("--n-hc", type=int, default=13)
    s.add_argument("--minutes", type=float, default=10.2)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_synth)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
