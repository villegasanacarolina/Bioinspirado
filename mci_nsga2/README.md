# Réplica: selección de canales/características EEG con NSGA-II para detectar MCI

Implementación en Python de **Aljalal, Aldosari, Molinas & Alturki (2024)**,
*Scientific Reports* 14:12483 — doi:10.1038/s41598-024-63180-y.
Los autores trabajaron en MATLAB 2022 y no publicaron código; esto lo reconstruye
a partir de la sección *Materials and methods*.

## 1. Cómo correrlo en tu computadora (paso a paso, desde cero)

**a) Instala Python** (una sola vez). Descárgalo de https://www.python.org/downloads/
(versión 3.10 o más nueva). En Windows, en la primera pantalla del instalador
marca la casilla **"Add Python to PATH"**.

**b) Arma esta estructura de carpetas** (por ejemplo en Documentos):

```
proyecto_mci/
├── mci_nsga2/        <- lo que sale al descomprimir mci_nsga2.zip
│   ├── mci_nsga2/
│   ├── README.md
│   └── requirements.txt
└── data/             <- lo que sale al descomprimir EEG_1.zip
    ├── 1.edf ... 27.edf
    └── states.xlsx   <- de aquí se leen solas las etiquetas MCI / NORMAL
```

Usa **EEG_1.zip** (el de 27 sujetos): es la base que usa el paper.
Se descarga del primer enlace de https://misp.mui.ac.ir/en/eeg-data

**c) Abre una terminal dentro de la carpeta `mci_nsga2`:**
- Windows: abre la carpeta en el Explorador, escribe `cmd` en la barra de
  dirección y presiona Enter.
- Mac: clic derecho sobre la carpeta → *Servicios* → *Nueva terminal en la carpeta*.

**d) Instala las librerías** (una sola vez; en Mac usa `python3` y `pip3`):

```bash
pip install -r requirements.txt
```

**e) Corre los pasos del §3** copiando y pegando cada comando en esa terminal.

## 2. Los datos

Base pública de Isfahan (Kashefpoor et al., 2016): 27 sujetos (11 MCI, 16 HC),
19 canales, 256 Hz, unos 30 min por registro, formato EDF.

- Si la carpeta trae `states.xlsx`, las etiquetas se leen automáticamente.
- Para otra base: el código reconoce `MCI` / `Normal` en el nombre de carpeta o
  archivo, o puedes pasar un `labels.csv` (columnas `file,label`; 1 = MCI, 0 = HC)
  con `--labels-csv`.
- Ojo: el segundo enlace de la página ("34 patients", EEG_2.zip) es otra base
  que el paper **no** usa. Antes de combinarla revisa si repite sujetos con
  `inventario_eeg.py`.

## 3. Correr (en este orden)

```bash
# Paso 1: preprocesar + DWT + VMD + las 10 medidas  (VMD tarda ~30 min en 1 núcleo)
python -m mci_nsga2 extract --data-dir ../data --out-dir features --exclude-noisiest-hc 3
#   revisa features/noise_report.csv para decidir qué HC excluir (ver §5)

# Paso 2: Experimento I — todos los canales (Tabla 2) y 10-fold (Tabla 3)
python -m mci_nsga2 exp1 --kfold

# Paso 3: Experimento II — selección de canales con NSGA-II (Tablas 4, 5, 8, 9)
python -m mci_nsga2 exp2 --methods DWT --measures ThEn --classifiers DA

# Paso 4: Experimento III — selección de características (Tablas 6, 7, 10, 11, 13)
python -m mci_nsga2 exp3 --methods DWT --measures TeEng --classifiers DA
```

Sin `--methods/--measures/--classifiers` corre las 80 combinaciones (2 × 10 × 4).
Los resultados (frente de Pareto, métricas, historial por generación) quedan en
`results/*.json` y en pantalla se imprime al lado el valor del paper cuando lo hay.

Para probar sin el dataset real: `python -m mci_nsga2 synth` genera EDF sintéticos.

### Tiempo esperado (estimado)
| Paso | Configuración del paper | Tiempo aprox. |
|---|---|---|
| extract | 24 sujetos × 60 segmentos | 30 min (1 núcleo), mucho menos con `--n-jobs -1` |
| exp1 | 80 combinaciones | 10–20 min |
| exp2 | pob 200 × 50 gen, por combinación | ~10–30 min (KNN/DA), más con RF |
| exp3 | pob 1000 × 100 gen, por combinación | 1–3 h (KNN/DA), más con RF |

Consejo: empieza por las 8 combinaciones de la Tabla 12 en lugar de las 48, y usa
`--pop 100 --gen 30` para explorar antes de una corrida completa.

## 4. Valores del paper para comparar

| Combinación | Exp I (19 canales) | Exp II (canales) | Exp III (características) |
|---|---|---|---|
| DWT-STD-LDA | **89.72** (mejor del Exp I) | – | – |
| DWT-TeEng-LDA | 80.35 | 92.78 (6 canales) | 95.83 (12 feat., 9 canales) |
| DWT-ThEn-LDA | – | **94.86** (7 canales) | – |
| VMD-TeEng-SVM | 74.24 | 91.56 (5 canales) | 95.28 (8 feat., 7 canales) |
| VMD-KFD-SVM | 77.36 | 90.49 (7 canales) | 93.89 (13 feat., 11 canales) |
| VMD-KFD-KNN | 53.26 (10-fold: 94.03) | 77.36 (2), 87.22 (3) | 92.57 (19 feat., 11 canales) |
| DWT-ThEn-KNN | 53.75 | 83.75 (2), 87.86 (3) | – |
| DWT-STD-KNN | 65.56 | – | 91.04 (13 feat., 8 canales) |
| DWT-TeEng-KNN | – | – | 93.13 (10 feat., 6 canales) |

Sólo están los números que aparecen en el texto; el resto de las tablas hay que
copiarlas del PDF (la Tabla 2 tiene una fe de erratas: doi:10.1038/s41598-024-64545-z).
Canales más elegidos en el paper: **Cz, F8, T4, Fp1**.

## 5. Qué es igual al paper y qué es supuesto

**Igual al paper:** 19 canales 10-20, 256 Hz, primeros 10 min, pasa-banda 0.5–64 Hz,
CAR, segmentos de 10 s (60 por sujeto, 1440 en total), DWT de 5 niveles sin D1
(6 × 19 = 114 features), VMD con 5 modos + residuo + original (7 × 19 = 133),
las 10 medidas, los 4 clasificadores con los rangos de hiperparámetros del paper
(K 1–10 y 10 distancias, 3 kernels, LDA/QDA, profundidad 1–35), LOSO con
promedio de CA por sujeto, NSGA-II con los dos objetivos y cromosoma
binario + genes enteros, población/generaciones 200/50 y 1000/100.

**Supuestos (el paper no los especifica) — todos están en `mci_nsga2/config.py`:**

1. **Preprocesado.** Cleanline → notch 50/100 Hz. ICA → opcional y automática
   (`--ica`). La inspección visual no se puede reproducir.
2. **Qué 3 HC se excluyeron.** El paper no lo dice. `--exclude-noisiest-hc 3`
   quita los 3 HC con más segmentos > 100 µV (ver `noise_report.csv`), o elige a
   mano con `--exclude`. **Esto solo puede mover varios puntos la exactitud.**
3. **Wavelet madre:** `db4`.
4. **VMD:** parámetros por defecto de `vmd` de MATLAB (α = 1000, τ = 0.01,
   500 iteraciones, inicialización por picos).
5. **Umbral de SuEn/ThEn:** 2 µV. Afecta directamente esas dos medidas.
6. **TShEn:** Shannon sobre la energía normalizada. **LBP:** log10 de la potencia
   media. **TeEng:** log10 Σ|x² − x₋₁x₊₁|. **HFD:** kmax = 10.
7. **Estandarización z-score** dentro de cada pliegue (`--no-standardize` la quita).
8. **Hiperparámetros del Exp. I:** KNN K=5 euclidiana, SVM lineal, LDA, RF profundidad 35.
   `exp1 --tune` busca el mejor por LOSO.
9. **Operadores genéticos:** torneo binario, cruza uniforme (0.9), mutación bit-flip
   (1/L), reinicio de genes enteros (0.2). La restricción "≥ 1 canal" se cumple
   por **reparación** (se enciende un bit al azar), no por castigo.
10. **Sensibilidad/especificidad en LOSO:** promedio de CA en sujetos MCI / HC.

Por todo esto, **no esperes los mismos números exactos**; lo que sí debería
replicarse es el patrón: Exp I < Exp II < Exp III, LOSO ≪ 10-fold, y canales
como Cz/F8/T4/Fp1 apareciendo seguido.

## 6. Idea para extender el trabajo

NSGA-II usa la exactitud LOSO como fitness y el paper reporta **esa misma**
exactitud. Eso es optimista (sesgo de selección): los canales se eligen viendo
a los sujetos de prueba. Una extensión natural es un **LOSO anidado**: correr
NSGA-II con 23 sujetos y evaluar la solución elegida en el sujeto 24, repetido
24 veces. También puedes comparar contra un AG de un solo objetivo con suma
ponderada (α·CA − β·#canales).

## 7. Prueba externa: desarrollar con EEG_1 y probar con EEG_2

Diseño: todos los métodos se entrenan y eligen sus electrodos SOLO con EEG_1
(24 personas, igual que el paper) y se prueban UNA sola vez en las 34 personas
de EEG_2, que ningún método vio nunca. Los métodos quedaron fijados antes de
ver EEG_2 (ver `persona.py`).

Coloca `EEG_1` y `EEG_2` descomprimidos en carpetas separadas (cada una con su
`states.xlsx`). Desde la carpeta `mci_nsga2`:

```powershell
# 1) extraer características (unos minutos cada una)
python -m mci_nsga2.externo extraer --data-dir "..\..\EEG_1" --nombre EEG_1 --exclude-noisiest-hc 3
python -m mci_nsga2.externo extraer --data-dir "..\..\EEG_2" --nombre EEG_2

# 2) entrenar con EEG_1 y probar con EEG_2  (~15-30 min)
python -m mci_nsga2.externo probar --train externo\EEG_1.npz --test externo\EEG_2.npz
```

Qué compara (resultados en `results\prueba_externa.json`):

| Método | Qué es |
|---|---|
| R | Réplica del paper: DWT-ThEn-LDA por segmentos, 19 electrodos, NSGA-II 200×50, regla de mayor CA. Imprime también la "CA interna" que reportaría el paper |
| M0 | EEG limpio (ICA), 15 electrodos, potencia relativa, sin selección |
| M4 | Igual, eligiendo 1 electrodo por zona (búsqueda exhaustiva) |
| M1 | Igual, NSGA-II con la regla del paper (5 semillas) |
| M6 | Igual, PSO binario con penalización por electrodo (5 semillas) |

Para cada método: personas bien diagnosticadas, exactitud con IC 95%, exactitud
balanceada con valor p (permutando etiquetas de prueba), sensibilidad y
especificidad. Al final, un diagnóstico de "efecto de base": si un clasificador
distingue EEG_1 de EEG_2 solo por el EEG (50% = indistinguibles).

## 8. Análisis secundario preregistrado: montaje físico (H4)

M0 y M4 calculan cada electrodo después de la referencia promedio y la ICA, que usan
los 19 electrodos. Un aparato con solo 4 electrodos no podría calcular esa señal.
`fisico.py` repite M0 y M4 cambiando SOLO eso: filtros por canal, sin referencia
promedio y sin ICA. Los métodos principales no se tocan.

**Paso 1: extraer (se puede hacer antes del preregistro; no entrena nada).**
Cada base pesa mucho, pero los archivos que salen pesan pocos MB: son los únicos
que hay que compartir.

```powershell
python -m mci_nsga2.externo extraer --data-dir "..\..\EEG_2" --nombre EEG_2
python -m mci_nsga2.fisico  extraer --data-dir "..\..\EEG_2" --nombre EEG_2
python -m mci_nsga2.externo extraer --data-dir "..\..\EEG_1" --nombre EEG_1 --exclude-noisiest-hc 3
python -m mci_nsga2.fisico  extraer --data-dir "..\..\EEG_1" --nombre EEG_1 --igual-que externo\EEG_1.npz
```

**Paso 2: depositar el preregistro en OSF (Preregistration_MCI_EEG.pdf). NO correr `probar` antes.**

La extracción es determinista: volver a extraer da archivos `.npz` idénticos bit a bit, así que
sus huellas SHA-256 (listadas en el preregistro) permiten verificar que se usaron los datos registrados.

**Paso 3: la prueba, una sola vez.**

```powershell
python -m mci_nsga2.externo probar --train externo\EEG_1.npz --test externo\EEG_2.npz
python -m mci_nsga2.fisico  probar --train externo\EEG_1_fisico.npz --test externo\EEG_2_fisico.npz --ica-train externo\EEG_1.npz --ica-test externo\EEG_2.npz
python -m mci_nsga2.analisis_prereg
```

`analisis_prereg` no entrena nada: lee las predicciones guardadas y aplica las reglas
de decisión del preregistro (H1-H4), con resultados en `results\prereg_resultados.json`.

Único cambio a `externo.py` respecto de la versión congelada: `_report` guarda la
predicción de cada persona y R imprime sus canales. Ningún cálculo cambia
(ver `CAMBIOS.md`).

## Estructura

```
mci_nsga2/
  config.py          constantes del paper y supuestos
  data_loader.py     EDF -> preprocesado -> segmentos; reporte de ruido
  decomposition.py   DWT y VMD (vectorizada por canal)
  measures.py        las 10 medidas
  features.py        tensor de características y matriz por medida
  classify.py        RF, LDA/QDA, SVM, KNN (10 distancias) y LOSO / 10-fold
  nsga2.py           NSGA-II desde cero (orden no dominado, hacinamiento, reparación)
  paper_reference.py valores reportados en el paper
  synthetic.py       EDF sintéticos para pruebas
  nested.py          LOSO anidado (guarda cada pliegue al terminar)
  limpieza.py        ICA para artefactos oculares + potencia relativa por banda
  persona.py         selección a nivel persona: NSGA-II, zonas, PSO binario
  externo.py         prueba externa EEG_1 -> EEG_2
  fisico.py          análisis secundario: M0 y M4 con montaje físico (sin CAR ni ICA)
  analisis_prereg.py reglas de decisión del preregistro (H1-H4) sobre las predicciones guardadas
  __main__.py        línea de comandos
inventario_eeg.py    compara dos zips (archivos idénticos / mismo registro)
huella_espectral.py  verificación de duplicados robusta a recortes
```
