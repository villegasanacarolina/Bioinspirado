# Cambios respecto de la versión congelada (mci_nsga2_2.zip)

Los métodos R, M0, M4, M1 y M6 no cambian. Ningún cálculo cambia.

## Archivos modificados

`mci_nsga2/externo.py` (2 líneas)

1. `_report` guarda también `pred` (predicción de cada persona) y `y` (su etiqueta) en
   `results/prueba_externa.json`. Hace falta para las comparaciones pareadas del
   preregistro (H2 y H3).
2. La línea de R añade la lista de canales elegidos al campo `extra` (antes solo se
   imprimía en pantalla). Hace falta para reportar cuántos electrodos usa R.

## Archivos nuevos

- `mci_nsga2/fisico.py`: análisis secundario preregistrado (H4). Repite M0 y M4 con
  montaje físico: filtros por canal, sin referencia promedio y sin ICA. Mismos 15
  electrodos, bandas, Welch, modelo y búsqueda por zonas que `limpieza.py` y `persona.py`.
  Con `--ica-train/--ica-test` compara cada persona contra la versión con ICA
  (bootstrap pareado estratificado, margen -10 puntos, Holm sobre 2).
- `mci_nsga2/analisis_prereg.py`: aplica las reglas de decisión del preregistro a las
  predicciones guardadas. No entrena nada.
- `CAMBIOS.md` (este archivo), `SHA256SUMS.txt` y la sección 8 del `README.md`.

## Verificación

- Diferencias: `diff` contra el zip original muestra solo las 2 líneas descritas.
- Prueba de humo con EEG_1 como entrenamiento y como prueba (sin tocar EEG_2):
  `externo probar` (con población y generaciones pequeñas), `fisico probar` y
  `analisis_prereg` corren de principio a fin.
- Reproducibilidad: extraer EEG_1 dos veces, en ejecuciones independientes, da archivos `.npz`
  idénticos bit a bit (misma huella SHA-256).

## Cambios posteriores a la prueba externa (versión del repositorio)

- `inventario_eeg.py`: se reescribió una línea del comentario de instrucciones de uso.
  Es una herramienta auxiliar para comparar las dos bases; no forma parte del análisis
  confirmatorio y su código no cambió. Por eso su huella en `SHA256SUMS.txt` difiere de
  la del paquete registrado en OSF (igual que la de este archivo, CAMBIOS.md); todos los
  demás archivos, incluidos todos los del análisis, conservan su huella.
