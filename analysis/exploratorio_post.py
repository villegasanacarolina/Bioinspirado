"""Análisis exploratorios posteriores a la prueba confirmatoria (declarados en el preregistro)."""
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # raíz del repositorio
EEG1_DIR = _os.environ.get("EEG1_DIR", "data/EEG_1")
EEG2_DIR = _os.environ.get("EEG2_DIR", "data/EEG_2")

import sys, json, numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
sys.path.insert(0, "mci_nsga2")
from mci_nsga2 import persona as P
from mci_nsga2.fisico import _cargar
from mci_nsga2.limpieza import CH15

E = "mci_nsga2/externo/"
R = "results/confirmatory/"
out = {}

def evalua(Xtr, Ytr, Xte, Yte, mask):
    c = P.cols(mask); m = P.model().fit(Xtr[:, c], Ytr)
    pred = m.predict(Xte[:, c]); sc = m.decision_function(Xte[:, c])
    return P.balanced(pred, Yte), roc_auc_score(Yte, sc), pred, sc

# (i) entrenar con las 27 personas de EEG_1
for tag, tr27, te in (("ICA", "mci_nsga2/externo/EEG_1_27.npz", E + "EEG_2.npz"),
                      ("fisico", "mci_nsga2/externo/EEG_1_27_fisico.npz", E + "EEG_2_fisico.npz")):
    Xtr, Ytr, _ = _cargar(tr27); Xte, Yte, _ = _cargar(te)
    full = np.ones(15, np.int64); m4 = P.select_M4(Xtr, Ytr)
    for k, mk in (("M0", full), ("M4", m4)):
        b, a, _, _ = evalua(Xtr, Ytr, Xte, Yte, mk)
        out[f"27|{tag}|{k}"] = {"balanceada": b, "auc": a, "electrodos": [CH15[i] for i in np.flatnonzero(mk)]}
        print(f"27 personas {tag:6s} {k}: balanceada {b:5.1f}  AUC {a:.3f}  {[CH15[i] for i in np.flatnonzero(mk)]}", flush=True)

# (ii) AUC externo de M0/M4 entrenados con 24 (ICA y físico) + relación con el ruido de red en EEG_2
conf = pd.read_csv("results/exploratory/confusores.csv")
c2 = conf[conf.base == "EEG_2"].set_index("sid")
for tag, tr, te in (("ICA", E + "EEG_1.npz", E + "EEG_2.npz"), ("fisico", E + "EEG_1_fisico.npz", E + "EEG_2_fisico.npz")):
    Xtr, Ytr, _ = _cargar(tr); Xte, Yte, sids = _cargar(te)
    for k, mk in (("M0", np.ones(15, np.int64)), ("M4", P.select_M4(Xtr, Ytr))):
        b, a, pred, sc = evalua(Xtr, Ytr, Xte, Yte, mk)
        ln = c2.loc[sids, "linea50_log"].values; fr = c2.loc[sids, "frontal_lento_log"].values
        r1 = spearmanr(sc, ln); r2 = spearmanr(sc, fr)
        out[f"24|{tag}|{k}"] = {"balanceada": b, "auc": a, "rho_linea50": r1.statistic, "p_linea50": r1.pvalue,
                                "rho_frontal": r2.statistic, "p_frontal": r2.pvalue, "score": sc.tolist(), "y": Yte.tolist(), "sids": sids}
        print(f"24 personas {tag:6s} {k}: balanceada {b:5.1f}  AUC {a:.3f}  rho(score, 50Hz) {r1.statistic:+.2f} (p {r1.pvalue:.3f})"
              f"  rho(score, frontal lento) {r2.statistic:+.2f} (p {r2.pvalue:.3f})", flush=True)

# (iii) estabilidad de electrodos por semilla (M1, M6) y efecto de base
ext = json.load(open(R + "prueba_externa.json"))
for metodo in ("M1", "M6"):
    sel = [tuple(eval(r["extra"][r["extra"].index("["):])) for r in ext["resultados"] if r["metodo"].startswith(metodo)]
    cnt = Counter(c for s in sel for c in s)
    out[f"estabilidad|{metodo}"] = {"selecciones": [list(s) for s in sel], "frecuencia": dict(cnt)}
    print(metodo, "selecciones por semilla:", sel, "| frecuencia:", dict(cnt.most_common()))
out["efecto_base"] = ext.get("efecto_base")
json.dump(out, open("results/exploratory/exploratorio_post.json", "w"), indent=1, default=float)

# (iv) POST HOC (no preregistrado): estandarizar cada base con su propia media y desviación (sin etiquetas)
def zbase(X): return (X - X.mean(0)) / X.std(0)
for tag, tr, te in (("ICA", E + "EEG_1.npz", E + "EEG_2.npz"), ("fisico", E + "EEG_1_fisico.npz", E + "EEG_2_fisico.npz")):
    Xtr, Ytr, _ = _cargar(tr); Xte, Yte, _ = _cargar(te)
    Xtr, Xte = zbase(Xtr), zbase(Xte)
    for k, mk in (("M0", np.ones(15, np.int64)), ("M4", P.select_M4(Xtr, Ytr))):
        b, a, pred, sc = evalua(Xtr, Ytr, Xte, Yte, mk)
        rng = np.random.default_rng(0)
        null = np.array([roc_auc_score(rng.permutation(Yte), sc) for _ in range(10000)])
        pa = (1 + (null >= a).sum()) / 10001
        out[f"posthoc_z|{tag}|{k}"] = {"balanceada": b, "auc": a, "p_auc": pa, "electrodos": [CH15[i] for i in np.flatnonzero(mk)]}
        print(f"POST HOC z por base {tag:6s} {k}: balanceada {b:5.1f}  AUC {a:.3f} (p {pa:.3f})  {[CH15[i] for i in np.flatnonzero(mk)]}", flush=True)
json.dump(out, open("results/exploratory/exploratorio_post.json", "w"), indent=1, default=float)
