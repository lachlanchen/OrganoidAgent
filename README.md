# OrganoidAgent

OrganoidAgent is a lightweight Tornado backend + PWA frontend for browsing and previewing organoid datasets (tables, images, archives, and h5ad files).

## Quick Start

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # optional if you already have the deps
python app.py --port 8080
```

Open `http://localhost:8080`.

## Download Data (Optional)

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

Downloaded data lives in `datasets/` (git-ignored).

## Project Layout

- `app.py`: Tornado server and API routes.
- `web/`: PWA assets.
- `scripts/`: dataset download helpers.
- `datasets/`: local data storage.
- `papers/`: submodule with reference materials.
