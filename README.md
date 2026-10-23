# External validation of NSGA-II electrode selection for EEG-based MCI detection

Preregistered replication and external validation of Aljalal et al. (2024),
*Scientific Reports* 14:12483 ([doi:10.1038/s41598-024-63180-y](https://doi.org/10.1038/s41598-024-63180-y)).

**Authors:** Jessica Milagros Cordero Rivera, Ana Carolina Torres Villegas, Salvador Miguel Hinojosa Cervantes
(Tecnológico de Monterrey, Campus Guadalajara).
**Preregistration:** <https://osf.io/4enh8>

## Main result

| | Internal (EEG_1, 24 people) | External (EEG_2, 34 people) |
|---|---|---|
| Replication of the original method (5 electrodes) | 75.1% LOSO accuracy | 45.8% balanced accuracy (95% CI 31.2–60.1) |
| Four cleaner / parsimonious alternatives | 61–64% (nested LOSO) | 50.0% (all 34 people predicted MCI) |

No method detected MCI above chance in new people. The two datasets, from the same centre and
equipment, were 94.4% distinguishable from EEG features alone.

## Repository layout

```
data/                 EEG recordings (not in git; see "Data / Datos")
mci_nsga2/            frozen analysis package (replication, methods R/M0/M4/M1/M6,
                      physical montage, preregistered decision rules); SHA256SUMS.txt
analysis/             exploratory analyses, results summary and figures
results/confirmatory/ outputs of the single preregistered test (JSON + logs)
results/exploratory/  outputs of the exploratory analyses
paper/                manuscript sources (LaTeX) in English (en/) and Spanish (es/), figures
preregistration/      preregistration as registered on OSF (PDF)
submission/           cover letter and reporting checklist (LaTeX and PDF)
poster/               A0 poster in Spanish and English (PDF) and its source
docs/                 plain-language explanation of the project (Spanish)
```

Open the folder in VS Code; the recommended extensions (Python, LaTeX Workshop) are listed in
`.vscode/extensions.json`, and LaTeX Workshop is configured to build with XeLaTeX.

## Reproducing the analysis

1. Install Python 3.10+ and the requirements: `pip install -r mci_nsga2/requirements.txt`.
2. Download both datasets (see "Data / Datos" below) into `data/EEG_1` and `data/EEG_2`
   (or set the environment variables `EEG1_DIR` and `EEG2_DIR`).
3. Extract features (deterministic; the SHA-256 of the resulting files must match the preregistration):

```bash
cd mci_nsga2
python -m mci_nsga2.externo extraer --data-dir ../data/EEG_1 --nombre EEG_1 --exclude-noisiest-hc 3
python -m mci_nsga2.fisico  extraer --data-dir ../data/EEG_1 --nombre EEG_1 --igual-que externo/EEG_1.npz
python -m mci_nsga2.externo extraer --data-dir ../data/EEG_2 --nombre EEG_2
python -m mci_nsga2.fisico  extraer --data-dir ../data/EEG_2 --nombre EEG_2 --igual-que externo/EEG_2.npz
python -m mci_nsga2.externo extraer --data-dir ../data/EEG_1 --nombre EEG_1_27
python -m mci_nsga2.fisico  extraer --data-dir ../data/EEG_1 --nombre EEG_1_27 --igual-que externo/EEG_1_27.npz
sha256sum externo/*.npz
```

4. Confirmatory test (about 25 minutes on 2 cores):

```bash
python -m mci_nsga2.externo probar --train externo/EEG_1.npz --test externo/EEG_2.npz --results-dir ../results/confirmatory
python -m mci_nsga2.fisico  probar --train externo/EEG_1_fisico.npz --test externo/EEG_2_fisico.npz \
       --ica-train externo/EEG_1.npz --ica-test externo/EEG_2.npz --results-dir ../results/confirmatory
python -m mci_nsga2.analisis_prereg --results-dir ../results/confirmatory
cd ..
```

5. Exploratory analyses, summary and figures (from the repository root):

```bash
for s in desarrollo_cv control_positivo lpo_auc confusores exploratorio_post resumen figs; do python analysis/$s.py; done
```

6. Manuscript: `cd paper/en && latexmk -xelatex main.tex` (and the same in `paper/es`).

## Data / Datos

The EEG recordings are **not** included in this repository (`data/` is in `.gitignore`).
Both datasets are public and are distributed by Isfahan University of Medical Sciences:

**Download page:** <https://misp.mui.ac.ir/en/eeg-data>

| Dataset | Link on that page | People | Put it in | Cite |
|---|---|---|---|---|
| EEG_1 (development) | "Download EEG Signals From Normal and MCI (Mild Cognitive Impairment) Cases" (file `EEG_1.zip`) | 27 (11 MCI, 16 controls) | `data/EEG_1/` (27 `.edf` files + `states.xlsx`) | Kashefpoor, Rabbani & Barekatain (2016), *J Med Signals Sens* 6(1):25–32 |
| EEG_2 (external test) | "Download here (34 patients)" (file `EEG_2.zip`) | 34 (18 MCI, 16 controls) | `data/EEG_2/` (34 `.edf` files + `States.xlsx`) | Kashefpoor, Rabbani & Barekatain (2019), *Biomed Signal Process Control* 53:101559 |

Expected layout:

```
data/
├── EEG_1/   1.edf … 27.edf, states.xlsx
└── EEG_2/   1.edf … 34.edf, States.xlsx
```

The extracted features (`mci_nsga2/externo/*.npz`) are also ignored by git; step 3 below regenerates
them bit for bit, and their SHA-256 must match the preregistration:

| File | SHA-256 |
|---|---|
| `EEG_1.npz` | `5a4cee0f3558edd97ff696ef9ce82700a47eb593703bf04488d0e8550d16f228` |
| `EEG_1_fisico.npz` | `f19a619264d0785fbcedf093efe34ffadc7f7dd3f4693f063b4db2df99a31e0c` |
| `EEG_2.npz` | `cc9236c74d3797e5c10ad7d0b511951495460435e993f2d801ccb9f3b1a5a7d5` |
| `EEG_2_fisico.npz` | `1d829b7a383ac4e74feacf409b2cc66ea850a14cfc40c68892c2c882bff18066` |

*Español:* los registros de EEG no están en el repositorio. Descárgalos de
<https://misp.mui.ac.ir/en/eeg-data>: el primer enlace de la página es EEG_1 (27 personas) y el enlace
"Download here (34 patients)" es EEG_2. Descomprímelos en `data/EEG_1` y `data/EEG_2`.

## License

Code: MIT (see `LICENSE`). Text and figures: CC BY 4.0.

## Citation

See `CITATION.cff`.
