[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/figs/banner.png" alt="LazyingArt banner" />
</p>

# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Status](https://img.shields.io/badge/Status-Active-success)
![Data](https://img.shields.io/badge/Data-Local%20first-4c956c)
![Preview](https://img.shields.io/badge/Preview-Multi--format-f4a261)

OrganoidAgent es un backend ligero con Tornado + un frontend Progressive Web App (PWA) para explorar y previsualizar datasets de organoides localmente. Ofrece previsualizaciones prácticas según el tipo de archivo para tablas, imágenes de microscopía (incluyendo TIFF), archivos comprimidos, archivos de texto gzip y objetos de análisis AnnData `.h5ad`.

## Resumen 🔭

La aplicación principal está diseñada para exploración interactiva de datasets con configuración mínima:

- API backend y motor de previsualización en `app.py`
- Frontend PWA en `web/`
- Scripts de descarga en `scripts/`
- Espacio de trabajo local de datasets en `datasets/` (ignorado por git)

Este repositorio también incluye espacios de trabajo adyacentes de investigación y utilidades (`BioAgent`, `BioAgentUtils`, `references`, `results`, `vendor`, submódulo `papers`). El runtime principal descrito en este README es la aplicación `OrganoidAgent` del nivel superior.

## Características ✨

- Indexación local de datasets con resúmenes de tamaño y cantidad de archivos
- Listado recursivo de archivos de datasets con inferencia del tipo de archivo
- El soporte de previsualización incluye tablas CSV/TSV/XLS/XLSX
- El soporte de previsualización incluye imágenes TIFF/JPG/PNG
- El soporte de previsualización incluye resúmenes de `.h5ad` con generación de vista previa de dispersión de embeddings/PCA
- El soporte de previsualización incluye listado de archivos ZIP/TAR/TGZ + intento de vista previa de la primera imagen
- El soporte de previsualización incluye vista previa de primeras líneas de texto `.gz`
- Endpoint de extracción de archivos comprimidos para datasets grandes empaquetados
- Tarjetas de metadatos a nivel de dataset renderizadas desde Markdown
- Frontend PWA con service worker y manifest
- Sanitización básica de rutas (`safe_dataset_path`) para limitar el acceso a archivos dentro de `datasets/`

### De un vistazo

| Área | Qué proporciona |
|---|---|
| Descubrimiento de datasets | Listado de datasets a nivel de directorio con conteo de archivos y resúmenes de tamaño |
| Exploración de archivos | Listado recursivo e inferencia de tipo (`image`, `table`, `analysis`, `archive`, etc.) |
| Previsualizaciones enriquecidas | Tablas, TIFF/imágenes, fragmentos de texto gzip, contenidos de archivos comprimidos, resúmenes de AnnData |
| Visualizaciones de análisis | Vistas previas de dispersión `.h5ad` desde embeddings `obsm` o fallback a PCA |
| Soporte de empaquetado | Listado de archivos comprimidos + endpoint de extracción para paquetes comprimidos grandes |
| UX web | PWA instalable con recursos de service worker compatibles con uso offline |

## Estructura del Proyecto 🗂️

```text
OrganoidAgent/
├─ app.py
├─ web/
│  ├─ index.html
│  ├─ app.js
│  ├─ styles.css
│  ├─ sw.js
│  ├─ manifest.json
│  └─ icons/
├─ scripts/
│  ├─ download_organoid_datasets.py
│  ├─ download_drug_screening_datasets.py
│  └─ overlay_segmentations.py
├─ datasets/                      # datos descargados y caché de previsualización (ignorado por git)
├─ metadata/
│  └─ zenodo_10643410.md
├─ papers/                        # submódulo: prompt-is-all-you-need
├─ i18n/                          # actualmente presente para archivos README multilingües
├─ BioAgent/                      # aplicación relacionada pero separada
├─ BioAgentUtils/                 # utilidades relacionadas de entrenamiento/datos
├─ references/
├─ results/
└─ vendor/                        # submódulos externos (copilot-sdk, paper-agent, codex)
```

## Requisitos Previos ✅

- Python `3.10+`
- Gestor de entornos recomendado: `conda` o `venv`

Paquetes de Python requeridos/opcionales inferidos del código fuente:

| Paquete | Rol |
|---|---|
| `tornado` | Requerido para iniciar el servidor |
| `pandas` | Opcional: soporte de previsualización de tablas |
| `anndata`, `numpy` | Opcional: previsualización `.h5ad` y gráficos de análisis |
| `Pillow` | Opcional: renderizado de imágenes y previsualizaciones generadas |
| `tifffile` | Opcional: soporte de previsualización TIFF |
| `requests` | Opcional: scripts de descarga de datasets |
| `kaggle` | Opcional: descargas de Kaggle en el script de drug-screening |

Nota de suposición: actualmente no existe `requirements.txt`, `pyproject.toml` ni `environment.yml` en la raíz para la aplicación de nivel superior.

## Instalación ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Opción A: conda (ejemplo)
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# Opción B: solo runtime mínimo
pip install tornado
```

## Uso 🚀

### Inicio Rápido

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # opcional si ya tienes las dependencias
python app.py --port 8080
```

Abre `http://localhost:8080`.

### Prueba Rápida de API

```bash
curl http://localhost:8080/api/datasets
```

### Descargar Datos (Opcional)

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

Los datos descargados se almacenan en `datasets/` (ignorado por git).

## Endpoints de API 🌐

| Método | Endpoint | Propósito |
|---|---|---|
| `GET` | `/api/datasets` | Listar datasets con estadísticas resumidas |
| `GET` | `/api/datasets/{name}` | Listar archivos de un dataset |
| `GET` | `/api/datasets/{name}/metadata` | Devolver tarjeta de metadatos en markdown |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | Listado de archivos orientado por categoría |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | Payload de previsualización según tipo de archivo |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | Extraer archivo comprimido en una carpeta hermana `_extracted` |
| `GET` | `/files/<path>` | Servir archivo de dataset en bruto |
| `GET` | `/previews/<path>` | Servir recurso de previsualización generado |

Ejemplo de llamada de previsualización:

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## Configuración 🧩

La configuración actual del runtime es intencionalmente pequeña:

- Puerto del servidor: argumento `--port` en `app.py` (predeterminado `8080`)
- Directorio de datos: fijado a `datasets/` relativo a la raíz del repositorio
- Caché de previsualizaciones: `datasets/.cache/previews`
- Mapeo de metadatos: diccionario `DATASET_METADATA` en `app.py`
- Token de API de GitHub para el descargador (opcional): variable de entorno `GITHUB_TOKEN` o `--github-token`

Nota de suposición: si necesitas raíces de dataset configurables o ajustes de servidor de producción, todavía no están expuestos en archivos de configuración de nivel superior.

## Ejemplos 🧪

### Explorar archivos específicos por categoría

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### Extraer un archivo comprimido

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### Ejecutar modos de descarga selectivos

```bash
# Datasets de organoides: omitir GEO, mantener Zenodo
python scripts/download_organoid_datasets.py --skip-geo

# Datasets de drug-screening: solo Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## Notas de Desarrollo 🛠️

- El backend sirve recursos estáticos del frontend desde `web/`.
- El service worker y el manifest están en `web/sw.js` y `web/manifest.json`.
- El enrutamiento por tipo de archivo y las previsualizaciones están implementadas en `app.py`.
- Validación manual (guía actual del proyecto): la PWA carga en `http://localhost:8080`
- Validación manual (guía actual del proyecto): `/api/datasets` devuelve JSON
- Validación manual (guía actual del proyecto): las vistas previas se renderizan para CSV/XLSX/imágenes/archivos comprimidos

## Solución de Problemas 🩺

- `ModuleNotFoundError` para librerías de previsualización: instala los paquetes faltantes (`pandas`, `anndata`, `numpy`, `Pillow`, `tifffile`).
- Listado de datasets vacío: confirma que existen datos en `datasets/` y que los directorios no empiezan con punto.
- La vista previa de `.h5ad` no muestra imagen de dispersión: verifica que `anndata`, `numpy` y `Pillow` estén instalados.
- Problemas con vista previa/extracción de archivos comprimidos grandes: usa el endpoint de extracción e inspecciona directamente los archivos extraídos.
- Errores por límite de tasa del descargador de GitHub: proporciona `GITHUB_TOKEN` por variable de entorno o bandera CLI.
- Descarga de Kaggle no funciona: instala `kaggle` y configura credenciales en `~/.kaggle/kaggle.json`.

## Hoja de Ruta 🧭

Posibles próximas mejoras (todavía no implementadas completamente en esta aplicación raíz):

- Añadir manifiesto de dependencias en raíz (`requirements.txt` o `pyproject.toml`)
- Añadir pruebas automatizadas para handlers de API y funciones de previsualización
- Añadir configuración de raíz de datasets y ajustes de caché
- Añadir perfil de ejecución explícito para producción (sin debug, guía de reverse proxy)
- Ampliar documentación multilingüe bajo `i18n/`

## Contribuir 🤝

Las contribuciones son bienvenidas. Flujo práctico:

1. Haz un fork y crea una rama enfocada.
2. Mantén los cambios acotados a una sola área lógica.
3. Valida manualmente el inicio de la app y los endpoints clave.
4. Abre un PR con resumen, comandos ejecutados y capturas de pantalla para cambios de UI.

Convenciones de estilo locales en este repositorio:

- Python: indentación de 4 espacios, funciones/archivos en snake_case, clases en CapWords
- Mantener la lógica frontend en `web/app.js` para esta app (evitar reescrituras de framework innecesarias)
- Mantener comentarios concisos y solo donde la lógica no sea obvia

## Diseño del Proyecto (Resumen Canónico) 📌

- `app.py`: servidor Tornado y rutas de API.
- `web/`: recursos de la PWA.
- `scripts/`: scripts auxiliares de descarga de datasets.
- `datasets/`: almacenamiento local de datos.
- `papers/`: submódulo con materiales de referencia.

## Licencia 📄

Actualmente no existe un archivo `LICENSE` de proyecto en la raíz de este repositorio.

Nota de suposición: hasta que se agregue una licencia en raíz, trata los términos de reutilización/redistribución como no especificados para el codebase de OrganoidAgent de nivel superior.

## Patrocinio y Donaciones ❤️

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Donar: https://chat.lazying.art/donate
- PayPal: https://paypal.me/RongzhouChen
- Stripe: https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400
