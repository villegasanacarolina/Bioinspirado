import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # raíz del repositorio
EEG1_DIR = _os.environ.get("EEG1_DIR", "data/EEG_1")
EEG2_DIR = _os.environ.get("EEG2_DIR", "data/EEG_2")
import sys, json, numpy as np, itertools
sys.path.insert(0,"mci_nsga2")
from mci_nsga2 import persona as P
from mci_nsga2.fisico import _cargar
res={}
for tag,f in (("ICA","EEG_1.npz"),("fisico","EEG_1_fisico.npz")):
    X,Y,_=_cargar(f"mci_nsga2/externo/{f}")
    i1=np.flatnonzero(Y==1); i0=np.flatnonzero(Y==0); c=P.cols(np.ones(15,np.int64)); win=[]
    for a,b in itertools.product(i1,i0):
        tr=np.ones(len(Y),bool); tr[[a,b]]=False
        m=P.model().fit(X[tr][:,c],Y[tr]); s=m.decision_function(X[[a,b]][:,c])
        win.append(1.0 if s[0]>s[1] else 0.5 if s[0]==s[1] else 0.0)
    res[tag]=float(np.mean(win)); print(tag,"M0 LPO-AUC",round(np.mean(win),3),"pares",len(win))
json.dump(res,open("lpo_auc.json","w"))
