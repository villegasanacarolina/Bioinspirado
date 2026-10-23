# Paquete completo del artículo

**Alta exactitud interna, desempeño externo al nivel del azar: validación externa preregistrada de la
selección de electrodos con NSGA-II para detectar deterioro cognitivo leve con EEG**

Jessica Milagros Cordero Rivera · Ana Carolina Torres Villegas · Salvador Miguel Hinojosa Cervantes
Tecnológico de Monterrey, Campus Guadalajara · Preregistro: https://osf.io/4enh8

## Contenido

| Carpeta | Qué hay |
|---|---|
| `01_Paper/` | `Paper_EN.pdf` y `Paper_ES.pdf` (15 páginas cada uno) y el código LaTeX (`LaTeX/en`, `LaTeX/es`, figuras y bibliografía) |
| `02_Poster/` | Póster científico A0 en español e inglés (PDF) y su fuente (HTML, script y figuras) |
| `03_Envio_a_revista/` | Carta al editor y lista de verificación de reporte (TRIPOD+AI), en PDF y LaTeX |
| `04_Preregistro_OSF/` | El preregistro tal como se registró (PDF y Word) y la guía de OSF |
| `05_Repositorio_GitHub/` | Repositorio listo para publicar: código, resultados, manuscrito, licencia, cita y metadatos de Zenodo. Instrucciones en `PUBLICAR.md` |

## Compilar el LaTeX

Requiere XeLaTeX (TeX Live o MiKTeX) y las fuentes TeX Gyre (vienen con TeX Live). En Overleaf: sube la carpeta
`LaTeX` completa, abre `en/main.tex` (o `es/main.tex`) y en *Menu → Compiler* elige **XeLaTeX**.

```
cd 01_Paper/LaTeX/en && latexmk -xelatex main.tex
cd 01_Paper/LaTeX/es && latexmk -xelatex main.tex
```

## Resultado principal

| | Interno (EEG_1, 24 personas) | Externo (EEG_2, 34 personas) |
|---|---|---|
| Réplica del método original (5 electrodos) | 75.1 % | 45.8 % de exactitud balanceada (IC 95 % 31.2–60.1) |
| Cuatro alternativas más limpias o parsimoniosas | 61–64 % (LOSO anidado) | 50.0 % (las 34 personas predichas como MCI) |

Ningún método superó el azar. Las dos bases, del mismo centro y equipo, se distinguen en 94.4 % solo con el EEG.

## Pendientes antes de enviar a una revista

1. Que el profesor revise el texto completo.
2. Confirmar las contribuciones de cada persona (sección *Author contributions*), el financiamiento (ahora dice que no hubo) y el correo de correspondencia.
3. Publicar el repositorio en GitHub y Zenodo (`05_Repositorio_GitHub/.../PUBLICAR.md`) y poner el DOI de Zenodo en *Code availability*.
4. Elegir la revista. Recomendación: *PLOS ONE* (publica réplicas y resultados negativos con criterio de solidez metodológica; la carta ya está dirigida a ella). Alternativas: *Scientific Reports* (misma revista del artículo original) o *Journal of Neural Engineering*. Revisar el formato y el límite de palabras de la revista elegida.
5. Antes de enviar, considerar compartir el manuscrito con los autores del artículo original; es una práctica de cortesía habitual en réplicas.
