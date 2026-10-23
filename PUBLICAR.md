# Cómo publicar este repositorio en GitHub y obtener un DOI en Zenodo

Tiempo estimado: 20 minutos. Lo hace una sola persona (recomendado: quien será autora de correspondencia).

## 1. Subir a GitHub

1. Entra a <https://github.com/new>.
2. Nombre del repositorio: `mci-eeg-external-validation`. Visibilidad: **Public**. No marques "Add a README" (ya viene uno).
3. Da clic en **Create repository**.
4. Opción sin terminal: en la página del repositorio vacío da clic en **uploading an existing file**, arrastra todo el contenido de esta carpeta (no la carpeta misma) y da clic en **Commit changes**.
   GitHub no sube carpetas vacías ni archivos que empiezan con punto si arrastras desde algunos navegadores; revisa que aparezcan `.zenodo.json` y `.gitignore`. Si no aparecen, créalos con **Add file → Create new file** y pega su contenido.
5. Opción con terminal (desde esta carpeta):

```bash
git init
git add .
git commit -m "Version 1.0.0: code, results and manuscript"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/mci-eeg-external-validation.git
git push -u origin main
```

6. Agrega a Jessica y al profesor en **Settings → Collaborators**.

## 2. Conectar con Zenodo

1. Entra a <https://zenodo.org> y elige **Log in → Log in with GitHub**.
2. En el menú de tu cuenta ve a **GitHub**. Busca `mci-eeg-external-validation` y activa el interruptor (**On**).
3. Zenodo leerá `.zenodo.json` para el título, las personas autoras y la licencia.

## 3. Crear la versión con DOI

1. En GitHub, en el repositorio, ve a **Releases → Create a new release**.
2. En **Choose a tag** escribe `v1.0.0` y crea la etiqueta.
3. Título: `v1.0.0`. Descripción: "Code, results and manuscript sources for the preregistered external validation (https://osf.io/4enh8)."
4. Da clic en **Publish release**.
5. En unos minutos aparece el registro en Zenodo con su DOI (formato `10.5281/zenodo.XXXXXXX`).

## 4. Después

1. Copia el DOI de Zenodo y ponlo en el manuscrito, secciones **Code availability** / **Disponibilidad del código** (en lugar de "the accompanying public repository").
2. En OSF, en el proyecto asociado al registro, agrega el enlace de GitHub y el DOI de Zenodo (el registro no se modifica; solo se agrega el enlace al proyecto).
3. Si después cambias el código, crea una versión nueva (`v1.0.1`); Zenodo genera un DOI nuevo y conserva el anterior.
