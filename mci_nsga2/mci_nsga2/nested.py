"""
LOSO anidado: estimación honesta de la exactitud después de seleccionar canales
o características con NSGA-II.

En el paper, NSGA-II usa como fitness la CA LOSO de los 24 sujetos y luego reporta
esa misma CA. El sujeto de prueba de cada pliegue influyó en qué canales se
eligieron, así que la cifra es optimista (sesgo de selección).

Aquí, para cada sujeto s (pliegue externo):
  1. NSGA-II corre SOLO con los otros 23 sujetos (fitness = LOSO interno de 23 pliegues)
  2. del frente de Pareto se toma la solución de mayor CA interna
  3. se entrena con los 23 sujetos y se evalúa en s, que nunca se vio
La CA anidada es el promedio de esos 24 resultados.

Uso:
  python -m mci_nsga2.nested --features-dir F --method DWT --measure ThEn --clf KNN \
         --mode channels --pop 60 --gen 20
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from joblib import Parallel, delayed

from . import config as C
from .classify import _zscore, make_classifier
from .features import feature_matrix, load_tensor
from .nsga2 import NSGA2, SelectionProblem


def outer_fold(X, y, groups, nb, clf, mode, pop, gen, seed, g_out):
    tr = groups != g_out
    prob = SelectionProblem(X[tr], y[tr], groups[tr], nb, clf, C.PARAM_BOUNDS[clf],
                            mode=mode, seed=seed)
    ga = NSGA2(prob, pop_size=pop, n_gen=gen, seed=seed, n_jobs=1, verbose=False).run()
    front = ga.pareto_front()
    best = max(front, key=lambda s: (s["acc"], -s["n_selected"]))
    cols = prob.columns(np.array(best["mask"]))
    Xtr, Xte = _zscore(X[tr][:, cols], X[~tr][:, cols])
    pred = make_classifier(clf, best["params"], seed).fit(Xtr, y[tr]).predict(Xte)
    return {"subject": int(g_out), "label": int(y[~tr][0]),
            "outer_acc": float(100 * np.mean(pred == y[~tr])),
            "inner_acc": best["acc"], "n_selected": best["n_selected"],
            "mask": best["mask"], "params": best["params"]}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", required=True)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--method", required=True)
    p.add_argument("--measure", required=True)
    p.add_argument("--clf", required=True)
    p.add_argument("--mode", choices=["channels", "features"], default="channels")
    p.add_argument("--pop", type=int, default=60)
    p.add_argument("--gen", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=-1)
    a = p.parse_args(argv)

    d = load_tensor(os.path.join(a.features_dir, f"{a.method}.npz"))
    X, names, nb = feature_matrix(d, a.measure)
    y, g = d["y"], d["groups"]
    t0 = time.time()
    # Cada pliegue se guarda en cuanto termina (un renglón JSON); si la corrida se
    # interrumpe, al relanzarla se saltan los pliegues ya hechos.
    os.makedirs(a.results_dir, exist_ok=True)
    ckpt = os.path.join(a.results_dir,
                        f"nested_{a.mode}_{a.method}-{a.measure}-{a.clf}_p{a.pop}g{a.gen}s{a.seed}.jsonl")
    done = {}
    if os.path.exists(ckpt):
        for line in open(ckpt):
            r = json.loads(line); done[r["subject"]] = r
    todo = [s for s in np.unique(g) if int(s) not in done]
    print(f"Pliegues ya hechos: {len(done)}; por hacer: {len(todo)}", flush=True)
    for r in Parallel(n_jobs=a.n_jobs, return_as="generator_unordered")(
            delayed(outer_fold)(X, y, g, nb, a.clf, a.mode, a.pop, a.gen, a.seed, s) for s in todo):
        done[r["subject"]] = r
        with open(ckpt, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(f"  pliegue {r['subject']:2d}: interna {r['inner_acc']:6.2f}%  externa {r['outer_acc']:6.2f}%"
              f"  ({len(done)}/24, {(time.time() - t0) / 60:.1f} min)", flush=True)
    res = [done[int(s)] for s in np.unique(g)]
    outer = np.array([r["outer_acc"] for r in res])
    inner = np.array([r["inner_acc"] for r in res])
    lab = np.array([r["label"] for r in res])
    for r in res:
        r["sid"] = str(d["sids"][r["subject"]])
        m = np.array(r["mask"])
        r["selected"] = ([C.CHANNELS[i] for i in np.flatnonzero(m)] if a.mode == "channels"
                         else [names[i] for i in np.flatnonzero(m)])
    summary = {"combo": f"{a.method}-{a.measure}-{a.clf}", "mode": a.mode,
               "pop": a.pop, "gen": a.gen,
               "nested_acc": float(outer.mean()),
               "nested_sens": float(outer[lab == 1].mean()),
               "nested_spec": float(outer[lab == 0].mean()),
               "mean_inner_acc": float(inner.mean()),
               "minutes": (time.time() - t0) / 60, "folds": res}
    print(f"\n{summary['combo']} ({a.mode}): CA interna (optimista) = {inner.mean():.2f}%  |  "
          f"CA LOSO anidada (honesta) = {outer.mean():.2f}%  "
          f"(sens {summary['nested_sens']:.1f}, spec {summary['nested_spec']:.1f})")
    os.makedirs(a.results_dir, exist_ok=True)
    out = os.path.join(a.results_dir, f"nested_{a.mode}_{summary['combo']}.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
