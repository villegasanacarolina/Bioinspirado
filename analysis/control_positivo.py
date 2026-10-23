import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # raíz del repositorio
EEG1_DIR = _os.environ.get("EEG1_DIR", "data/EEG_1")
EEG2_DIR = _os.environ.get("EEG2_DIR", "data/EEG_2")
import sys, json, numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
sys.path.insert(0,"mci_nsga2")
from mci_nsga2.limpieza import CH15
POST=[CH15.index(c) for c in ["P3","Pz","P4","O1","O2"]]
out={}
for f in ["EEG_1","EEG_1_fisico","EEG_2","EEG_2_fisico"]:
    d=np.load(f"mci_nsga2/externo/{f}.npz"); g=d["groups"]
    S=np.unique(g); X=np.array([d["relpow"][g==s].mean(0) for s in S]); y=np.array([d["y"][g==s][0] for s in S])
    for bi,b in enumerate(["delta","theta","alpha","beta"]):
        v=X[:,[c*4+bi for c in POST]].mean(1)
        auc=roc_auc_score(y,-v) # AUC de "menor valor => MCI"
        p=mannwhitneyu(v[y==1],v[y==0]).pvalue
        out[f"{f}|{b}"]={"mci_mediana":float(np.median(v[y==1])),"hc_mediana":float(np.median(v[y==0])),"auc_menor_es_mci":float(auc),"p":float(p),"vals":v.tolist(),"y":y.tolist()}
        if b in("alpha","theta"): print(f"{f:14s} {b:6s} MCI {np.median(v[y==1]):.3f} HC {np.median(v[y==0]):.3f}  AUC(menor=MCI) {auc:.2f} p {p:.3f}")
json.dump(out,open("control_positivo.json","w"))
