[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/figs/banner.png" alt="Bannière LazyingArt" />
</p>

# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Status](https://img.shields.io/badge/Status-Active-success)
![Data](https://img.shields.io/badge/Data-Local%20first-4c956c)
![Preview](https://img.shields.io/badge/Preview-Multi--format-f4a261)

OrganoidAgent est un backend Tornado léger avec une interface frontend Progressive Web App (PWA) pour parcourir et prévisualiser localement des jeux de données d’organoïdes. Il prend en charge des aperçus pratiques, adaptés au type de fichier, pour les tableaux, les images de microscopie (y compris TIFF), les archives, les fichiers texte gzip et les objets d’analyse AnnData `.h5ad`.

## Vue d’ensemble 🔭

L’application principale est conçue pour l’exploration interactive de jeux de données avec une configuration minimale :

- API backend et moteur de prévisualisation dans `app.py`
- Frontend PWA dans `web/`
- Scripts de téléchargement dans `scripts/`
- Espace de travail local des données dans `datasets/` (ignoré par git)

Ce dépôt contient aussi des espaces de travail de recherche et d’utilitaires adjacents (`BioAgent`, `BioAgentUtils`, `references`, `results`, `vendor`, sous-module `papers`). Le runtime principal décrit dans ce README est l’application `OrganoidAgent` à la racine.

## Fonctionnalités ✨

- Indexation locale des jeux de données avec résumés de taille et de nombre de fichiers
- Listing récursif des fichiers de jeux de données avec type de fichier inféré
- Prise en charge des aperçus CSV/TSV/XLS/XLSX
- Prise en charge des aperçus d’images TIFF/JPG/PNG
- Prise en charge des résumés `.h5ad` avec génération d’aperçus de nuages de points embedding/PCA
- Prise en charge du listing d’archives ZIP/TAR/TGZ + tentative d’aperçu de la première image
- Prise en charge de l’aperçu des premières lignes des fichiers texte `.gz`
- Endpoint d’extraction d’archives pour les jeux de données volumineux empaquetés
- Cartes de métadonnées au niveau jeu de données rendues à partir de Markdown
- Frontend PWA avec service worker et manifeste
- Assainissement de chemin de base (`safe_dataset_path`) pour confiner l’accès aux fichiers sous `datasets/`

### En un coup d’œil

| Zone | Ce que cela fournit |
|---|---|
| Découverte des jeux de données | Listing des jeux de données au niveau répertoire avec nombre de fichiers et résumés de taille |
| Exploration de fichiers | Listing récursif et inférence de type (`image`, `table`, `analysis`, `archive`, etc.) |
| Aperçus enrichis | Tableaux, TIFF/images, extraits texte gzip, contenu d’archives, résumés AnnData |
| Visualisations d’analyse | Aperçus de nuages de points `.h5ad` depuis des embeddings `obsm` ou repli PCA |
| Prise en charge des paquets | Listing d’archives + endpoint d’extraction pour les gros bundles compressés |
| Expérience web | PWA installable avec ressources service worker compatibles hors ligne |

## Structure du projet 🗂️

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
├─ datasets/                      # downloaded data and preview cache (git-ignored)
├─ metadata/
│  └─ zenodo_10643410.md
├─ papers/                        # submodule: prompt-is-all-you-need
├─ i18n/                          # currently present for multilingual README files
├─ BioAgent/                      # related but separate app
├─ BioAgentUtils/                 # related training/data utilities
├─ references/
├─ results/
└─ vendor/                        # external submodules (copilot-sdk, paper-agent, codex)
```

## Prérequis ✅

- Python `3.10+`
- Gestionnaire d’environnement recommandé : `conda` ou `venv`

Packages Python requis/optionnels déduits du code source :

| Package | Rôle |
|---|---|
| `tornado` | Requis pour démarrer le serveur |
| `pandas` | Optionnel : prise en charge des aperçus de tableaux |
| `anndata`, `numpy` | Optionnel : aperçu `.h5ad` et tracés d’analyse |
| `Pillow` | Optionnel : rendu d’images et génération d’aperçus |
| `tifffile` | Optionnel : prise en charge des aperçus TIFF |
| `requests` | Optionnel : scripts de téléchargement de données |
| `kaggle` | Optionnel : téléchargements Kaggle dans le script drug-screening |

Note d’hypothèse : il n’existe actuellement ni `requirements.txt`, ni `pyproject.toml`, ni `environment.yml` à la racine pour l’application principale.

## Installation ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Option A: conda (example)
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# Option B: minimal runtime only
pip install tornado
```

## Utilisation 🚀

### Démarrage rapide

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # optional if you already have the deps
python app.py --port 8080
```

Ouvrez `http://localhost:8080`.

### Test rapide de l’API

```bash
curl http://localhost:8080/api/datasets
```

### Télécharger les données (optionnel)

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

Les données téléchargées sont stockées dans `datasets/` (ignoré par git).

## Endpoints API 🌐

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/datasets` | List datasets with summary stats |
| `GET` | `/api/datasets/{name}` | List files for one dataset |
| `GET` | `/api/datasets/{name}/metadata` | Return markdown metadata card |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | Category-oriented file listing |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | File-type-aware preview payload |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | Extract archive into sibling `_extracted` folder |
| `GET` | `/files/<path>` | Raw dataset file serving |
| `GET` | `/previews/<path>` | Generated preview asset serving |

Exemple d’appel d’aperçu :

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## Configuration 🧩

La configuration runtime actuelle est volontairement minimale :

- Port serveur : argument `--port` dans `app.py` (par défaut `8080`)
- Répertoire de données : fixé à `datasets/` relatif à la racine du dépôt
- Cache d’aperçus : `datasets/.cache/previews`
- Mapping de métadonnées : dictionnaire `DATASET_METADATA` dans `app.py`
- Jeton API GitHub pour le downloader (optionnel) : variable d’environnement `GITHUB_TOKEN` ou `--github-token`

Note d’hypothèse : si vous avez besoin de racines de données configurables ou de paramètres serveur de production, ces options ne sont pas encore exposées dans des fichiers de configuration à la racine.

## Exemples 🧪

### Parcourir des fichiers par catégorie

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### Extraire une archive

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### Exécuter des modes de téléchargement sélectifs

```bash
# Organoid datasets: skip GEO, keep Zenodo
python scripts/download_organoid_datasets.py --skip-geo

# Drug-screening datasets: only Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## Notes de développement 🛠️

- Le backend sert les ressources statiques frontend depuis `web/`.
- Le service worker et le manifeste se trouvent dans `web/sw.js` et `web/manifest.json`.
- Le routage par type de fichier et les aperçus sont implémentés dans `app.py`.
- Validation manuelle (guidance actuelle du projet) : la PWA se charge sur `http://localhost:8080`
- Validation manuelle (guidance actuelle du projet) : `/api/datasets` retourne du JSON
- Validation manuelle (guidance actuelle du projet) : les aperçus s’affichent pour CSV/XLSX/images/archives

## Dépannage 🩺

- `ModuleNotFoundError` pour les bibliothèques d’aperçu : installez les packages manquants (`pandas`, `anndata`, `numpy`, `Pillow`, `tifffile`).
- Listing de jeux de données vide : vérifiez que des données existent sous `datasets/` et que les répertoires ne sont pas préfixés par un point.
- Aperçu `.h5ad` sans image de nuage de points : vérifiez que `anndata`, `numpy` et `Pillow` sont installés.
- Problèmes d’aperçu/extraction de grandes archives : utilisez l’endpoint d’extraction et inspectez directement les fichiers extraits.
- Erreurs de limite de débit du downloader GitHub : fournissez `GITHUB_TOKEN` via variable d’environnement ou option CLI.
- Téléchargement Kaggle non fonctionnel : installez `kaggle` et configurez les identifiants `~/.kaggle/kaggle.json`.

## Feuille de route 🧭

Améliorations potentielles suivantes (pas encore entièrement implémentées dans cette app racine) :

- Ajouter un manifeste de dépendances à la racine (`requirements.txt` ou `pyproject.toml`)
- Ajouter des tests automatisés pour les handlers API et les fonctions d’aperçu
- Ajouter une configuration de racine de données et de cache
- Ajouter un profil d’exécution explicite pour la production (non-debug, guide reverse-proxy)
- Étendre la documentation multilingue sous `i18n/`

## Contribuer 🤝

Les contributions sont les bienvenues. Workflow pratique :

1. Forker et créer une branche ciblée.
2. Limiter les changements à une seule zone logique.
3. Valider manuellement le démarrage de l’app et les endpoints clés.
4. Ouvrir une PR avec résumé, commandes exécutées et captures d’écran pour les changements UI.

Conventions de style locales dans ce dépôt :

- Python : indentation 4 espaces, fonctions/fichiers en snake_case, classes en CapWords
- Conserver la logique frontend dans `web/app.js` pour cette app (éviter les réécritures de framework inutiles)
- Garder des commentaires concis et uniquement lorsque la logique n’est pas évidente

## Disposition du projet (Résumé canonique) 📌

- `app.py` : serveur Tornado et routes API.
- `web/` : ressources PWA.
- `scripts/` : scripts d’aide au téléchargement de jeux de données.
- `datasets/` : stockage local des données.
- `papers/` : sous-module avec des documents de référence.

## Licence 📄

Aucun fichier `LICENSE` de projet à la racine n’est actuellement présent dans ce dépôt.

Note d’hypothèse : tant qu’une licence racine n’est pas ajoutée, considérez les conditions de réutilisation/redistribution comme non spécifiées pour la base de code OrganoidAgent au niveau racine.

## Sponsor & Dons ❤️

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Donate: https://chat.lazying.art/donate
- PayPal: https://paypal.me/RongzhouChen
- Stripe: https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400
