"""
Valores reportados en el texto del paper (Aljalal et al., 2024) para comparar.

Sólo están los que aparecen explícitamente en el texto/abstract. Las tablas
completas (2-13) deben copiarse del artículo; hay huecos marcados con None.
Nota: la Tabla 2 tuvo una fe de erratas (doi:10.1038/s41598-024-64545-z) porque
los encabezados VMD/DWT estaban invertidos en la primera versión.

Clave: (descomposición, medida, clasificador). "DA" = discriminante (LDA/QDA).
"""

# Experimento I: todas las características, LOSO (Tabla 2)
EXP1_LOSO = {
    ("VMD", "KFD", "KNN"): 53.26,
    ("VMD", "SuEn", "KNN"): 62.36,
    ("VMD", "TeEng", "SVM"): 74.24,
    ("VMD", "KFD", "SVM"): 77.36,
    ("DWT", "STD", "DA"): 89.72,     # el mejor del Exp. I (DWT-STD-LDA)
    ("DWT", "STD", "KNN"): 65.56,
    ("DWT", "ThEn", "KNN"): 53.75,
    ("DWT", "TeEng", "DA"): 80.35,
}

# Experimento I con 10-fold (Tabla 3) -> muestra la fuga de información
EXP1_KFOLD = {("VMD", "KFD", "KNN"): 94.03}

# Experimento II: selección de canales. {n_canales: CA}
EXP2 = {
    ("VMD", "KFD", "KNN"): {2: 77.36, 3: 87.22},
    ("DWT", "ThEn", "KNN"): {2: 83.75, 3: 87.86},
    ("VMD", "TeEng", "SVM"): {5: 91.56},
    ("VMD", "KFD", "SVM"): {7: 90.49},
    ("DWT", "TeEng", "DA"): {6: 92.78},
    ("DWT", "ThEn", "DA"): {7: 94.86},   # mejor del Exp. II
}

# Experimento III: selección de características. (n_features, n_canales, CA)
EXP3 = {
    ("VMD", "KFD", "KNN"): (19, 11, 92.57),
    ("VMD", "SuEn", "KNN"): (10, None, 91.53),
    ("DWT", "STD", "KNN"): (13, 8, 91.04),
    ("DWT", "TeEng", "KNN"): (10, 6, 93.13),
    ("VMD", "TeEng", "SVM"): (8, 7, 95.28),
    ("VMD", "KFD", "SVM"): (13, 11, 93.89),
    ("DWT", "TeEng", "DA"): (12, 9, 95.83),   # 4 combinaciones DWT llegan a 95.83
}

# Canales que más aparecen en las soluciones (Figs. 8, 11, 12)
KEY_CHANNELS = ["Cz", "F8", "T4", "Fp1"]
