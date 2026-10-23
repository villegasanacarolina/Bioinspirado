"""Figuras del artículo en inglés y español. Uso: python figs.py"""
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # raíz del repositorio
EEG1_DIR = _os.environ.get("EEG1_DIR", "data/EEG_1")
EEG2_DIR = _os.environ.get("EEG2_DIR", "data/EEG_2")

import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "mci_nsga2")
from mci_nsga2.analisis_prereg import _boot

S = json.load(open("results/exploratory/resumen.json"))
DEV = json.load(open("results/exploratory/desarrollo_cv.json"))
EXP = json.load(open("results/exploratory/exploratorio_post.json"))
CONF = pd.read_csv("results/exploratory/confusores.csv")
FIS = json.load(open("results/confirmatory/prueba_externa_fisico.json"))

BLUE, ORANGE, GREY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False})

T = {
 "en": dict(chance="chance (50%)", thr="needed to beat chance (62%)", bal="Person-level balanced accuracy on EEG_2 (%)",
            internal="internal LOSO accuracy\n(what the paper\nwould report)", ext="external test, 95% CI",
            elec="electrodes", dev_title="", loso="Person-wise LOSO (segment accuracy averaged per person)",
            kf="Record-wise 10-fold (segments of a person mixed)", paper="Reported by Aljalal et al. (Exp. I)",
            acc="Accuracy (%)", margin="non-inferiority margin (−10)", diff="Difference in balanced accuracy (points), 90% CI",
            noninf="non-inferior", inconc="not shown", trivial="met trivially (both predict MCI for all)", allmci="all 34 predicted MCI",
            delta="Delta relative power (%)\n(1–4 Hz of 1–30 Hz, mean of 15 electrodes)", line="Mains interference\nlog10 P(49–51 Hz)/P(1–45 Hz)",
            front="Slow frontal power\nlog10 µV² (0.5–4 Hz, Fp1/Fp2/F7/F8)", mci="MCI", hc="Control",
            sel_paper="Most frequent in\nAljalal et al.", sel_R="R (replication)", sel_M4="M4 (ICA)", sel_M4f="M4-physical",
            sel_M1="M1, 5 seeds", sel_M6="M6, 5 seeds",
            labels={"R": "R  replication", "M0": "M0  ICA, no selection", "M4": "M4  one per zone", "M1": "M1  NSGA-II",
                    "M6": "M6  binary PSO", "M0-fis": "M0  physical montage", "M4-fis": "M4  physical montage"},
            dlabels={"M0-R": "H2  M0 − R", "M4-M1": "H3  M4 − M1", "M6-M1": "H3  M6 − M1",
                     "M0fis-M0": "H4  M0 physical − ICA", "M4fis-M4": "H4  M4 physical − ICA"}),
 "es": dict(chance="azar (50%)", thr="necesario para superar el azar (62%)", bal="Exactitud balanceada por persona en EEG_2 (%)",
            internal="exactitud LOSO\ninterna (la que\nreportaría el paper)", ext="prueba externa, IC 95%",
            elec="electrodos", dev_title="", loso="LOSO por persona (exactitud de segmentos promediada por persona)",
            kf="10-fold por registro (segmentos de una persona mezclados)", paper="Reportado por Aljalal et al. (Exp. I)",
            acc="Exactitud (%)", margin="margen de no inferioridad (−10)", diff="Diferencia en exactitud balanceada (puntos), IC 90%",
            noninf="no inferior", inconc="no demostrado", trivial="se cumple trivialmente (ambos predicen MCI para todos)", allmci="las 34 predichas MCI",
            delta="Potencia delta relativa (%)\n(1–4 Hz de 1–30 Hz, media de 15 electrodos)", line="Interferencia de red\nlog10 P(49–51 Hz)/P(1–45 Hz)",
            front="Potencia lenta frontal\nlog10 µV² (0.5–4 Hz, Fp1/Fp2/F7/F8)", mci="MCI", hc="Control",
            sel_paper="Más frecuentes en\nAljalal et al.", sel_R="R (réplica)", sel_M4="M4 (ICA)", sel_M4f="M4 físico",
            sel_M1="M1, 5 semillas", sel_M6="M6, 5 semillas",
            labels={"R": "R  réplica", "M0": "M0  ICA, sin selección", "M4": "M4  uno por zona", "M1": "M1  NSGA-II",
                    "M6": "M6  PSO binario", "M0-fis": "M0  montaje físico", "M4-fis": "M4  montaje físico"},
            dlabels={"M0-R": "H2  M0 − R", "M4-M1": "H3  M4 − M1", "M6-M1": "H3  M6 − M1",
                     "M0fis-M0": "H4  M0 físico − ICA", "M4fis-M4": "H4  M4 físico − ICA"}),
}

# IC 95 % para las versiones físicas (mismo bootstrap estratificado)
yf = np.array(FIS["y_prueba"])
fp = {r["metodo"].split("-")[0] + "-fis": np.array(r["pred"]) for r in FIS["resultados"]}
Bf = _boot(fp, yf)
for k in fp:
    S["metodos"][k]["ic95"] = list(np.percentile(Bf[k], [2.5, 97.5]))
    S["metodos"][k]["electrodos_mediana"] = len(S["metodos"][k]["electrodos"])

def fig_dev(L, path):
    combos = list(DEV)
    paper = {"DWT-STD-LDA": 89.72, "DWT-TeEng-LDA": 80.35, "DWT-ThEn-KNN": 53.75, "DWT-STD-KNN": 65.56}
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    yy = np.arange(len(combos))[::-1]
    for i, c in zip(yy, combos):
        a, b = DEV[c]["loso_ca"], DEV[c]["kfold10_segmentos"]
        ax.plot([a, b], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter(a, i, s=36, color=BLUE, zorder=3, label=L["loso"] if i == yy[0] else None)
        ax.scatter(b, i, s=36, color=GREY, zorder=3, label=L["kf"] if i == yy[0] else None)
        if c in paper:
            ax.scatter(paper[c], i, s=44, marker="D", facecolor="white", edgecolor=ORANGE, lw=1.4, zorder=4,
                       label=L["paper"] if c == "DWT-STD-LDA" else None)
    ax.axvline(50, color=INK2, lw=0.8, ls=":")
    ax.set_yticks(yy); ax.set_yticklabels(combos); ax.set_xlim(40, 102); ax.set_xlabel(L["acc"])
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels(); ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1, fontsize=7.5)
    fig.savefig(path); plt.close(fig)

def fig_main(L, path):
    order = ["R", "M0", "M4", "M1", "M6", "M0-fis", "M4-fis"]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    yy = np.arange(len(order))[::-1]
    ax.axvspan(62, 100, color="#eef4fc", zorder=0)
    ax.axvline(50, color=INK2, lw=0.8, ls=":"); ax.axvline(62, color=BLUE, lw=0.8, ls="--")
    for i, k in zip(yy, order):
        m = S["metodos"][k]; lo, hi = m["ic95"]
        col = GREY if "fis" in k else INK
        ax.plot([lo, hi], [i, i], color=col, lw=1.6, zorder=2)
        ax.scatter(m["balanceada"], i, s=40, color=BLUE if k == "R" else col, zorder=3)
        ax.text(101.5, i, f"{int(m['electrodos_mediana'])} {L['elec']}", va="center", fontsize=7.5, color=INK2)
        if m.get("espec") == 0.0:
            ax.text(m["balanceada"] + 1.5, i, L["allmci"], va="center", fontsize=6.8, color=INK2)
    ax.scatter(S["R_ca_interna"], yy[0], s=46, marker="D", facecolor="white", edgecolor=ORANGE, lw=1.4, zorder=4)
    ax.text(S["R_ca_interna"] + 2, yy[0], L["internal"], fontsize=6.8, color=ORANGE, ha="left", va="center")
    ax.text(49.5, -0.95, L["chance"], fontsize=7, color=INK2, ha="right"); ax.text(62.5, -0.95, L["thr"], fontsize=7, color=BLUE)
    ax.set_yticks(yy); ax.set_yticklabels([L["labels"][k] for k in order]); ax.set_ylim(-1.3, len(order) - 0.2)
    ax.set_xlim(10, 100); ax.set_xlabel(L["bal"]); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.savefig(path); plt.close(fig)

def fig_diff(L, path):
    order = ["M0-R", "M4-M1", "M6-M1", "M0fis-M0", "M4fis-M4"]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    yy = np.arange(len(order))[::-1]
    ax.axvspan(-60, -10, color="#fbeee8", zorder=0); ax.axvline(-10, color=ORANGE, lw=1, ls="--"); ax.axvline(0, color=INK2, lw=0.8)
    for i, k in zip(yy, order):
        d = S["dif"][k]; ok = d["lo90"] > -10; triv = d["lo90"] == d["hi90"]
        col = GREY if (triv or not ok) else BLUE
        ax.plot([d["lo90"], d["hi90"]], [i, i], color=col, lw=1.8)
        ax.scatter(d["dif"], i, s=36, color=col, zorder=3, marker="s" if triv else "o")
        ax.text(61, i, L["trivial"] if triv else (L["noninf"] if ok else L["inconc"]), va="center", fontsize=7.2, color=BLUE if (ok and not triv) else INK2)
    ax.text(-11, -0.85, L["margin"], fontsize=7, color=ORANGE, ha="right")
    ax.set_yticks(yy); ax.set_yticklabels([L["dlabels"][k] for k in order]); ax.set_xlim(-60, 60); ax.set_ylim(-1.2, len(order) - 0.4)
    ax.set_xlabel(L["diff"]); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.savefig(path); plt.close(fig)

from mci_nsga2.fisico import _cargar
DEL = []
for base in ("EEG_1", "EEG_2"):
    X, Y, sids = _cargar(f"mci_nsga2/externo/{base}.npz")
    rp = np.exp(X).reshape(len(X), 15, 4).mean(1)[:, 0] * 100
    DEL += [{"base": base, "y": int(y), "sid": s_, "delta_rel": v} for y, s_, v in zip(Y, sids, rp)]
CONF = CONF.merge(pd.DataFrame(DEL), on=["base", "sid", "y"], how="left")

def fig_conf(L, path):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6))
    rng = np.random.default_rng(0)
    for ax, var, lab in zip(axs, ["delta_rel", "linea50_log", "frontal_lento_log"], [L["delta"], L["line"], L["front"]]):
        for j, (base, y) in enumerate([("EEG_1", 0), ("EEG_1", 1), ("EEG_2", 0), ("EEG_2", 1)]):
            v = CONF[(CONF.base == base) & (CONF.y == y)][var].dropna().values
            x = j + (0.4 if base == "EEG_2" else 0) + rng.uniform(-0.12, 0.12, len(v))
            ax.scatter(x, v, s=12, color=ORANGE if y else BLUE, alpha=0.8, lw=0)
            ax.plot([j + (0.4 if base == "EEG_2" else 0) - 0.22, j + (0.4 if base == "EEG_2" else 0) + 0.22], [np.median(v)] * 2, color=INK, lw=1.4)
        ax.set_xticks([0, 1, 2.4, 3.4]); ax.set_xticklabels([f"{L['hc']}\nEEG_1", f"{L['mci']}\nEEG_1", f"{L['hc']}\nEEG_2", f"{L['mci']}\nEEG_2"], fontsize=6.8)
        ax.set_ylabel(lab, fontsize=7.5); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

POS = {"Fp1": (-0.31, 0.95), "Fp2": (0.31, 0.95), "F7": (-0.81, 0.59), "F3": (-0.4, 0.5), "Fz": (0, 0.47), "F4": (0.4, 0.5),
       "F8": (0.81, 0.59), "T3": (-1, 0), "C3": (-0.5, 0), "Cz": (0, 0), "C4": (0.5, 0), "T4": (1, 0), "T5": (-0.81, -0.59),
       "P3": (-0.4, -0.5), "Pz": (0, -0.47), "P4": (0.4, -0.5), "T6": (0.81, -0.59), "O1": (-0.31, -0.95), "O2": (0.31, -0.95)}

def head(ax, freq, title):
    ax.add_patch(plt.Circle((0, 0), 1.18, fill=False, color=INK2, lw=0.9))
    ax.plot([-0.12, 0, 0.12], [1.17, 1.32, 1.17], color=INK2, lw=0.9)
    for ch, (x, y) in POS.items():
        f = freq.get(ch, 0)
        if f > 0:
            ax.add_patch(plt.Circle((x, y), 0.17, color=BLUE, alpha=0.3 + 0.7 * f, lw=0))
        else:
            ax.add_patch(plt.Circle((x, y), 0.17, fill=False, color="#c9c8c3", lw=0.8))
        ax.text(x, y, ch, ha="center", va="center", fontsize=6.3, color="white" if f > 0.5 else INK2, fontweight="bold" if f > 0 else "normal")
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.45); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=8.5, color=INK)

def fig_heads(L, path):
    def frec(sel):
        n = len(sel); c = {}
        for s in sel:
            for ch in s: c[ch] = c.get(ch, 0) + 1 / n
        return c
    paper = {c: 1 for c in ["Cz", "F8", "T4", "Fp1"]}
    Rch = {c: 1 for c in (S["R_canales"] or [])}
    m4 = {c: 1 for c in S["metodos"]["M4-fis"]["electrodos_ica"]}
    m4f = {c: 1 for c in S["metodos"]["M4-fis"]["electrodos"]}
    m1 = frec(EXP["estabilidad|M1"]["selecciones"]); m6 = frec(EXP["estabilidad|M6"]["selecciones"])
    fig, axs = plt.subplots(2, 3, figsize=(6.2, 4.3))
    for ax, f, t in zip(axs.ravel(), [paper, Rch, m4, m4f, m1, m6], [L["sel_paper"], L["sel_R"], L["sel_M4"], L["sel_M4f"], L["sel_M1"], L["sel_M6"]]):
        head(ax, f, t)
    fig.savefig(path); plt.close(fig)

for lang in ("en", "es"):
    L = T[lang]; d = f"paper/figs/{lang}_"
    fig_dev(L, d + "dev.pdf"); fig_main(L, d + "main.pdf"); fig_diff(L, d + "diff.pdf"); fig_conf(L, d + "conf.pdf"); fig_heads(L, d + "heads.pdf")
    for n in ("dev", "main", "diff", "conf", "heads"):
        pass
print("ok")
