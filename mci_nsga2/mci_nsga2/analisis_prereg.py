"""
Análisis confirmatorio preregistrado (H1-H4) a partir de los resultados de la prueba externa.

No vuelve a entrenar nada: lee las predicciones por persona que guardan
  results/prueba_externa.json          (python -m mci_nsga2.externo probar ...)
  results/prueba_externa_fisico.json   (python -m mci_nsga2.fisico probar ... --ica-train ... --ica-test ...)
y aplica exactamente las reglas de decisión del preregistro en OSF (Preregistration_MCI_EEG.pdf).

Uso:  python -m mci_nsga2.analisis_prereg [--results-dir results]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re

import numpy as np

from .persona import balanced

MARGEN = -10.0      # no inferioridad, puntos de exactitud balanceada
ALFA = 0.05
N_BOOT = 10000


def _boot(preds, y, seed=0):
    """Bootstrap de personas estratificado por clase. preds: dict nombre -> predicciones. Devuelve dict de muestras."""
    rng = np.random.default_rng(seed)
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = {k: np.empty(N_BOOT) for k in preds}
    for b in range(N_BOOT):
        idx = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        for k, p in preds.items():
            out[k][b] = balanced(p[idx], y[idx])
    return out


def _holm(ps):
    orden = np.argsort(ps); adj = np.empty(len(ps)); acum = 0.0
    for r, i in enumerate(orden):
        acum = max(acum, min(1.0, (len(ps) - r) * ps[i])); adj[i] = acum
    return adj


def _electrodos(extra):
    m = re.search(r"\[.*\]", extra or "")
    return ast.literal_eval(m.group(0)) if m else None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mci_nsga2.analisis_prereg")
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args(argv)
    ext = json.load(open(os.path.join(a.results_dir, "prueba_externa.json")))
    rows = ext["resultados"]
    if "pred" not in rows[0]:
        raise SystemExit("prueba_externa.json no tiene predicciones por persona: usa la versión de externo.py de este paquete")
    y = np.array(rows[0]["y"])

    # ---- predicción de cada método (M1 y M6: voto de mayoría de sus semillas, número impar)
    metodos, n_elec = {}, {}
    for r in rows:
        nombre = r["metodo"].split()[0]                      # R, M0, M4, M1, M6
        metodos.setdefault(nombre, []).append(np.array(r["pred"]))
        el = _electrodos(r["extra"])
        n_elec.setdefault(nombre, []).append(15 if el is None else len(el))     # M0 no imprime lista: 15
    pred = {k: (np.mean(v, 0) > 0.5).astype(int) if len(v) > 1 else v[0] for k, v in metodos.items()}
    ca_interna = float(re.search(r"CA interna ([\d.]+)%", rows[0]["extra"]).group(1))

    # ---- primario: cada método frente al azar (permutación de etiquetas, Holm sobre los 5)
    rng = np.random.default_rng(0)
    perms = [rng.permutation(y) for _ in range(N_BOOT)]
    nombres = list(pred)
    ps = []
    for k in nombres:
        obs = balanced(pred[k], y)
        null = np.array([balanced(pred[k], yp) for yp in perms])
        ps.append((1 + (null >= obs).sum()) / (1 + N_BOOT))
    holm = _holm(np.array(ps))
    B = _boot(pred, y)
    print(f"Prueba externa: {len(y)} personas ({y.sum()} MCI, {len(y) - y.sum()} sanas)\n")
    print(f"{'método':6s} {'electrodos':>10s} {'balanceada':>10s} {'IC95% bootstrap':>17s} {'p':>7s} {'p Holm':>7s}")
    tabla = []
    for k, p_, h in zip(nombres, ps, holm):
        lo, hi = np.percentile(B[k], [2.5, 97.5])
        ne = float(np.median(n_elec[k]))
        tabla.append({"metodo": k, "electrodos_mediana": ne, "balanceada": balanced(pred[k], y),
                      "ic95": [lo, hi], "p": p_, "p_holm": h})
        print(f"{k:6s} {ne:10.0f} {balanced(pred[k], y):9.1f}% {lo:7.1f} a {hi:5.1f}% {p_:7.3f} {h:7.3f}")

    def no_inf(a_, b_):
        d = B[a_] - B[b_]
        return float(balanced(pred[a_], y) - balanced(pred[b_], y)), float(np.percentile(d, 5)), \
            float((1 + (d <= MARGEN).sum()) / (1 + N_BOOT))

    dec = {}
    # H1: la CA interna de R queda por encima del IC95% de su exactitud balanceada externa
    hi_r = np.percentile(B["R"], 97.5)
    dec["H1"] = {"ca_interna": ca_interna, "balanceada_externa": balanced(pred["R"], y), "ic95_alto": hi_r,
                 "confirmada": bool(ca_interna > hi_r)}
    # H2: M0 no inferior a R
    d, lo, p = no_inf("M0", "R")
    dec["H2"] = {"diferencia": d, "ic90_bajo": lo, "p": p, "confirmada": bool(p < ALFA)}
    # H3: M4 y M6 no inferiores a M1 (Holm sobre 2) y con menos electrodos
    h3 = {k: no_inf(k, "M1") for k in ("M4", "M6")}
    adj = _holm(np.array([h3["M4"][2], h3["M6"][2]]))
    dec["H3"] = {k: {"diferencia": v[0], "ic90_bajo": v[1], "p": v[2], "p_holm": float(q),
                     "menos_electrodos": bool(np.median(n_elec[k]) < np.median(n_elec["M1"])),
                     "confirmada": bool(q < ALFA and np.median(n_elec[k]) < np.median(n_elec["M1"]))}
                 for (k, v), q in zip(h3.items(), adj)}
    # H4: montaje físico (lo calcula fisico.py)
    fp = os.path.join(a.results_dir, "prueba_externa_fisico.json")
    if os.path.exists(fp):
        fis = json.load(open(fp))
        dec["H4"] = {c["metodo"]: {"diferencia": c["diferencia"], "ic90_bajo": c["ic90_bajo"],
                                   "p_holm": c["p_holm"], "confirmada": bool(c["p_holm"] < ALFA)}
                     for c in fis.get("no_inferioridad", [])}

    print(f"\nH1  CA interna de R {ca_interna:.1f}% vs IC95% externo hasta {hi_r:.1f}% -> "
          f"{'CONFIRMADA' if dec['H1']['confirmada'] else 'no confirmada'}")
    print(f"H2  M0 - R = {dec['H2']['diferencia']:+.1f} (límite inferior IC90% {dec['H2']['ic90_bajo']:+.1f}; "
          f"margen {MARGEN:+.0f}) p={dec['H2']['p']:.3f} -> {'CONFIRMADA' if dec['H2']['confirmada'] else 'no confirmada'}")
    for k, v in dec["H3"].items():
        print(f"H3  {k} - M1 = {v['diferencia']:+.1f} (lím. inf. {v['ic90_bajo']:+.1f}) p Holm={v['p_holm']:.3f}, "
              f"menos electrodos: {v['menos_electrodos']} -> {'CONFIRMADA' if v['confirmada'] else 'no confirmada'}")
    for k, v in dec.get("H4", {}).items():
        print(f"H4  {k} físico - ICA = {v['diferencia']:+.1f} (lím. inf. {v['ic90_bajo']:+.1f}) p Holm={v['p_holm']:.3f}"
              f" -> {'CONFIRMADA' if v['confirmada'] else 'no confirmada'}")
    if "efecto_base" in ext:
        print(f"\nDescriptivo: efecto de base (distinguir EEG_1 de EEG_2) {ext['efecto_base']:.1f}% (50% = indistinguibles)")
    out = os.path.join(a.results_dir, "prereg_resultados.json")
    json.dump({"metodos": tabla, "hipotesis": dec}, open(out, "w"), indent=1, default=float)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
