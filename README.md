**Language:** English (this draft) | Additional language files will be added under `i18n/`.

<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/figs/banner.png" alt="LazyingArt banner" />
</p>

# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Status](https://img.shields.io/badge/Status-Active-success)

OrganoidAgent is a lightweight Tornado backend + Progressive Web App (PWA) frontend for browsing and previewing organoid datasets locally. It supports practical, file-type-aware previews for tables, microscopy images (including TIFF), archives, gzip text files, and AnnData `.h5ad` analysis objects.

## Overview

The core app is designed for interactive dataset exploration with minimal setup:

- Backend API and preview engine in `app.py`
- PWA frontend in `web/`
- Download helpers in `scripts/`
- Local dataset workspace in `datasets/` (git-ignored)

This repository also contains adjacent research and utility workspaces (`BioAgent`, `BioAgentUtils`, `references`, `results`, `vendor`, `papers` submodule). The primary runtime described in this README is the top-level `OrganoidAgent` app.

## Features

- Local dataset indexing with size and file-count summaries
- Recursive dataset file listing with inferred file kind
- Preview support includes CSV/TSV/XLS/XLSX tables
- Preview support includes TIFF/JPG/PNG images
- Preview support includes `.h5ad` summaries with embedding/PCA scatter preview generation
- Preview support includes ZIP/TAR/TGZ archive listing + first-image preview attempt
- Preview support includes `.gz` text first-lines preview
- Archive extraction endpoint for large packaged datasets
- Dataset-level metadata cards rendered from Markdown
- PWA frontend with service worker and manifest
- Basic path sanitization (`safe_dataset_path`) to confine file access under `datasets/`

## Project Structure

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

## Prerequisites

- Python `3.10+`
- Recommended environment manager: `conda` or `venv`

Required/optional Python packages inferred from source:

- Required for server startup: `tornado`
- Optional for full preview functionality: `pandas` (table preview)
- Optional for full preview functionality: `anndata`, `numpy` (`.h5ad` preview)
- Optional for full preview functionality: `Pillow` (image rendering)
- Optional for full preview functionality: `tifffile` (TIFF preview)
- Optional for data download scripts: `requests`
- Optional for Kaggle downloads in drug-screening script: `kaggle`

Assumption note: there is currently no root `requirements.txt`, `pyproject.toml`, or `environment.yml` for the top-level app.

## Installation

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Option A: conda (example)
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# Option B: minimal runtime only
pip install tornado
```

## Usage

### Quick Start

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # optional if you already have the deps
python app.py --port 8080
```

Open `http://localhost:8080`.

### API Smoke Test

```bash
curl http://localhost:8080/api/datasets
```

### Download Data (Optional)

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

Downloaded data lives in `datasets/` (git-ignored).

## API Endpoints

- `GET /api/datasets`
- `GET /api/datasets/{name}`
- `GET /api/datasets/{name}/metadata`
- `GET /api/category/{datasets|segmentation|features|analysis}`
- `GET /api/preview?path=<relative_path_under_datasets>`
- `POST /api/extract?path=<archive_relative_path_under_datasets>`
- `GET /files/<path>` (raw dataset file serving)
- `GET /previews/<path>` (generated preview assets)

Example preview call:

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## Configuration

Current runtime configuration is intentionally small:

- Server port: `--port` argument in `app.py` (default `8080`)
- Data directory: fixed to `datasets/` relative to repository root
- Preview cache: `datasets/.cache/previews`
- Metadata mapping: `DATASET_METADATA` dictionary in `app.py`
- GitHub API token for downloader (optional): `GITHUB_TOKEN` env var or `--github-token`

Assumption note: if you need configurable dataset roots or production server settings, these are not yet exposed in top-level configuration files.

## Examples

### Browse category-specific files

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### Extract an archive

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### Run selective download modes

```bash
# Organoid datasets: skip GEO, keep Zenodo
python scripts/download_organoid_datasets.py --skip-geo

# Drug-screening datasets: only Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## Development Notes

- Backend serves frontend static assets from `web/`.
- Service worker and manifest are in `web/sw.js` and `web/manifest.json`.
- File-type routing and previews are implemented in `app.py`.
- Manual validation (current project guidance): PWA loads at `http://localhost:8080`
- Manual validation (current project guidance): `/api/datasets` returns JSON
- Manual validation (current project guidance): previews render for CSV/XLSX/images/archives

## Troubleshooting

- `ModuleNotFoundError` for preview libraries: install missing packages (`pandas`, `anndata`, `numpy`, `Pillow`, `tifffile`).
- Empty dataset listing: confirm data exists under `datasets/` and directories are not dot-prefixed.
- `.h5ad` preview missing scatter image: check that `anndata`, `numpy`, and `Pillow` are installed.
- Large archive preview/extraction issues: use extraction endpoint and inspect extracted files directly.
- GitHub downloader rate limit errors: provide `GITHUB_TOKEN` via env var or CLI flag.
- Kaggle download not working: install `kaggle` and configure `~/.kaggle/kaggle.json` credentials.

## Roadmap

Potential next improvements (not yet fully implemented in this root app):

- Add root dependency manifest (`requirements.txt` or `pyproject.toml`)
- Add automated tests for API handlers and preview functions
- Add configurable dataset root and cache settings
- Add explicit production run profile (non-debug, reverse-proxy guidance)
- Expand multilingual documentation under `i18n/`

## Contributing

Contributions are welcome. A practical workflow:

1. Fork and create a focused branch.
2. Keep changes scoped to one logical area.
3. Manually validate app startup and key endpoints.
4. Open a PR with summary, commands run, and screenshots for UI changes.

Local style conventions in this repository:

- Python: 4-space indentation, snake_case functions/files, CapWords classes
- Keep frontend logic in `web/app.js` for this app (avoid unnecessary framework rewrites)
- Keep comments concise and only where logic is non-obvious

## Project Layout (Canonical Summary)

- `app.py`: Tornado server and API routes.
- `web/`: PWA assets.
- `scripts/`: dataset download helpers.
- `datasets/`: local data storage.
- `papers/`: submodule with reference materials.

## License

No top-level project `LICENSE` file is currently present in this repository root.

Assumption note: until a root license is added, treat reuse/redistribution terms as unspecified for the top-level OrganoidAgent codebase.

## Sponsor & Donate

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Donate: https://chat.lazying.art/donate
- PayPal: https://paypal.me/RongzhouChen
- Stripe: https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400
