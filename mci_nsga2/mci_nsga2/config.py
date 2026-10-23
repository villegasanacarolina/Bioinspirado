"""
Configuración y constantes tomadas de:
Aljalal, Aldosari, Molinas & Alturki (2024). "Selecting EEG channels and features
using multi-objective optimization for accurate MCI detection: validation using
leave-one-subject-out strategy". Scientific Reports 14:12483.
https://doi.org/10.1038/s41598-024-63180-y

Cada constante indica si viene del paper (PAPER) o si es un SUPUESTO nuestro
porque el paper no lo especifica.
"""

# --- Datos ---------------------------------------------------------------
# PAPER: 19 canales del sistema 10-20, fs = 256 Hz
CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "T3", "C3", "Cz", "C4", "T4",
            "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
FS = 256

# Nombres alternativos de la nomenclatura 10-10 moderna
CHANNEL_ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
                   "FP1": "Fp1", "FP2": "Fp2", "FZ": "Fz", "CZ": "Cz", "PZ": "Pz"}

# PAPER: solo los primeros 10 min; segmentos de 10 s sin traslape -> 60 por sujeto
DURATION_S = 600
SEGMENT_S = 10

# PAPER: pasa-banda 0.5-64 Hz, referencia promedio común, Cleanline, ICA, inspección visual
BANDPASS = (0.5, 64.0)
# SUPUESTO: la red eléctrica en Irán es de 50 Hz -> notch en 50 (Cleanline en el paper)
LINE_FREQ = 50.0

# --- Descomposición ------------------------------------------------------
# PAPER: DWT de 5 niveles -> A5, D1..D5; se descarta D1 (64-128 Hz).
# SUPUESTO: wavelet madre 'db4' (el paper no la menciona en este artículo).
WAVELET = "db4"
DWT_LEVEL = 5
DWT_BANDS = ["D2", "D3", "D4", "D5", "A5", "Orig"]          # 6 x ch = 114

# PAPER: VMD con 5 VMFs + residuo + señal original (7 x ch = 133).
# SUPUESTO: parámetros por defecto de la función vmd de MATLAB
# (PenaltyFactor=1000, LMUpdateRate=0.01, MaxIterations=500, init 'peaks').
VMD_K = 5
VMD_ALPHA = 1000.0
VMD_TAU = 0.01
VMD_MAX_ITER = 500
VMD_TOL = 1e-6
VMD_BANDS = ["VMF1", "VMF2", "VMF3", "VMF4", "VMF5", "Rest", "Orig"]

# --- Medidas (una por experimento) ----------------------------------------
# PAPER: STD, IQR, LBP, TeEng, ShEn, TShEn, SuEn, ThEn, KFD, HFD
MEASURES = ["STD", "IQR", "LBP", "TeEng", "ShEn", "TShEn", "SuEn", "ThEn", "KFD", "HFD"]
# SUPUESTO: umbral (en microvolts) para sure/threshold entropy; el paper remite
# a trabajos previos sin dar el valor. Es un parámetro que conviene explorar.
ENTROPY_THRESHOLD_UV = 2.0
# SUPUESTO: kmax de Higuchi
HFD_KMAX = 10

# --- Clasificadores ------------------------------------------------------
CLASSIFIERS = ["RF", "DA", "SVM", "KNN"]
# PAPER: K = 1..10 y 10 distancias (en el orden de MATLAB)
KNN_DISTANCES = ["euclidean", "seuclidean", "cityblock", "chebychev", "minkowski",
                 "mahalanobis", "cosine", "correlation", "spearman", "hamming"]
SVM_KERNELS = ["linear", "poly", "rbf"]          # PAPER: 1 lineal, 2 polinomial, 3 RBF
DA_TYPES = ["linear", "quadratic"]               # PAPER: 1 LDA, 2 QDA
RF_DEPTH_RANGE = (1, 35)                         # PAPER
RF_N_TREES = 100                                 # SUPUESTO
QDA_REG = 1e-2                                   # SUPUESTO (evita covarianzas singulares)

# Rango entero de cada parámetro optimizado por NSGA-II (genes enteros)
PARAM_BOUNDS = {
    "KNN": [(1, 10), (1, 10)],        # K, tipo de distancia
    "SVM": [(1, 3)],                  # kernel
    "DA":  [(1, 2)],                  # lineal / cuadrático
    "RF":  [RF_DEPTH_RANGE],          # profundidad
}

# Parámetros "por defecto" para el Experimento I (todas las variables, sin NSGA).
# SUPUESTO: el paper no deja claro qué hiperparámetros usó en el Exp. I.
DEFAULT_PARAMS = {"KNN": [5, 1], "SVM": [1], "DA": [1], "RF": [35]}

# --- NSGA-II (PAPER) ----------------------------------------------------
NSGA_CHANNELS = {"pop_size": 200, "n_gen": 50}
NSGA_FEATURES = {"pop_size": 1000, "n_gen": 100}
