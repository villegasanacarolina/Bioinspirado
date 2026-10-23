"""Reúne los resultados confirmatorios y exploratorios en resumen.json (misma semilla y bootstrap que los scripts registrados)."""
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # raíz del repositorio
EEG1_DIR = _os.environ.get("EEG1_DIR", "data/EEG_1")
EEG2_DIR = _os.environ.get("EEG2_DIR", "data/EEG_2")

import sys, json, re, numpy as np
sys.path.insert(0, "mci_nsga2")
from mci_nsga2.analisis_prereg import _boot
from mci_nsga2.fisico import _boot_diferencia
from mci_nsga2.persona import balanced

R = "results/confirmatory/"
ext = json.load(open(R + "prueba_externa.json"))
fis = json.load(open(R + "prueba_externa_fisico.json"))
pre = json.load(open(R + "prereg_resultados.json"))
rows = ext["resultados"]; y = np.array(rows[0]["y"])
met = {}
for r in rows:
    met.setdefault(r["metodo"].split()[0], []).append(np.array(r["pred"]))
pred = {k: (np.mean(v, 0) > 0.5).astype(int) if len(v) > 1 else v[0] for k, v in met.items()}
B = _boot(pred, y)
def dif(a, b):
    d = B[a] - B[b]
    return {"dif": float(balanced(pred[a], y) - balanced(pred[b], y)), "lo90": float(np.percentile(d, 5)), "hi90": float(np.percentile(d, 95))}
S = {"n": len(y), "n_mci": int(y.sum()), "metodos": {}, "dif": {}, "semillas": []}
for t in pre["metodos"]:
    k = t["metodo"]; p = pred[k]
    S["metodos"][k] = {**t, "sens": float(100 * np.mean(p[y == 1] == 1)), "espec": float(100 * np.mean(p[y == 0] == 0)),
                       "bien": int((p == y).sum())}
S["dif"]["M0-R"] = dif("M0", "R"); S["dif"]["M4-M1"] = dif("M4", "M1"); S["dif"]["M6-M1"] = dif("M6", "M1")
yf = np.array(fis["y_prueba"])
fpred = {r["metodo"].split("-")[0]: np.array(r["pred"]) for r in fis["resultados"]}
for c in fis["no_inferioridad"]:
    k = c["metodo"]
    S["dif"][f"{k}fis-{k}"] = {"dif": c["diferencia"], "lo90": c["ic90_bajo"], "hi90": c["ic90_alto"], "p_holm": c["p_holm"]}
    S["metodos"][f"{k}-fis"] = {"balanceada": c["bal_fisico"], "auc": c["auc_fisico"], "auc_ica": c["auc_ica"],
                               "electrodos": next(r["electrodos"] for r in fis["resultados"] if r["metodo"].startswith(k)),
                               "electrodos_ica": c["electrodos_ica"]}
for r in rows:
    S["semillas"].append({"metodo": r["metodo"], "balanceada": r["balanceada"], "p": r["p"], "extra": r["extra"]})
S["hipotesis"] = pre["hipotesis"]; S["efecto_base"] = ext.get("efecto_base")
S["R_ca_interna"] = pre["hipotesis"]["H1"]["ca_interna"]
m = re.search(r"\[(.*)\]", rows[0]["extra"]); S["R_canales"] = eval("[" + m.group(1) + "]") if m else None
json.dump(S, open("results/exploratory/resumen.json", "w"), indent=1, default=float)
print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("balanceada", "ic95", "p", "p_holm", "electrodos_mediana", "sens", "espec", "bien", "auc")}
                  for k, v in S["metodos"].items()}, indent=1, default=float))
print(json.dumps(S["dif"], indent=1)); print("efecto base", S["efecto_base"], "| R interna", S["R_ca_interna"], S["R_canales"])
