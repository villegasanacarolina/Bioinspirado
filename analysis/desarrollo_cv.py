import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # raíz del repositorio
EEG1_DIR = _os.environ.get("EEG1_DIR", "data/EEG_1")
EEG2_DIR = _os.environ.get("EEG2_DIR", "data/EEG_2")
import sys, json, numpy as np
sys.path.insert(0,"mci_nsga2")
from mci_nsga2 import config as C
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
d=np.load("mci_nsga2/externo/EEG_1.npz"); y=d["y"]; g=d["groups"]
res={}
for meas in ["ThEn","TeEng","STD"]:
    m=list(C.MEASURES).index(meas); X=d["paper"][...,m].reshape(len(y),-1)
    for name,clf in (("LDA",lambda: make_pipeline(StandardScaler(),LDA())),("KNN",lambda: make_pipeline(StandardScaler(),KNN(5)))):
        # LOSO por persona: promedio de exactitud por sujeto (como el paper) y exactitud balanceada por persona (voto)
        accs=[]; pred_p=[]; yp=[]
        for s in np.unique(g):
            tr=g!=s; p=clf().fit(X[tr],y[tr]).predict(X[~tr]); accs.append((p==y[~tr]).mean()); pred_p.append(int(p.mean()>0.5)); yp.append(y[~tr][0])
        pred_p=np.array(pred_p); yp=np.array(yp)
        bal=50*((pred_p[yp==1]==1).mean()+(pred_p[yp==0]==0).mean())
        kf=[]
        for tr,te in StratifiedKFold(10,shuffle=True,random_state=0).split(X,y):
            kf.append((clf().fit(X[tr],y[tr]).predict(X[te])==y[te]).mean())
        res[f"DWT-{meas}-{name}"]={"loso_ca":100*np.mean(accs),"loso_bal_persona":bal,"kfold10_segmentos":100*np.mean(kf)}
        print(f"DWT-{meas}-{name}: LOSO CA {100*np.mean(accs):5.1f}  balanceada persona {bal:5.1f}  10-fold segmentos {100*np.mean(kf):5.1f}",flush=True)
json.dump(res,open("desarrollo_cv.json","w"),indent=1)
