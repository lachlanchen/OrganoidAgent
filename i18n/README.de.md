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

OrganoidAgent ist ein leichtgewichtiges Tornado-Backend + Progressive Web App (PWA)-Frontend zum lokalen Durchsuchen und Vorschauen von Organoid-Datensätzen. Es unterstützt praxisnahe, dateitypabhängige Vorschauen für Tabellen, Mikroskopiebilder (einschließlich TIFF), Archive, gzip-Textdateien und AnnData-`.h5ad`-Analyseobjekte.

## Überblick 🔭

Die Kernanwendung ist für die interaktive Datensatz-Exploration mit minimalem Setup ausgelegt:

- Backend-API und Vorschau-Engine in `app.py`
- PWA-Frontend in `web/`
- Download-Helfer in `scripts/`
- Lokaler Datensatz-Arbeitsbereich in `datasets/` (git-ignoriert)

Dieses Repository enthält außerdem angrenzende Forschungs- und Utility-Arbeitsbereiche (`BioAgent`, `BioAgentUtils`, `references`, `results`, `vendor`, `papers`-Submodul). Die primäre Laufzeit, die in diesem README beschrieben ist, ist die App `OrganoidAgent` auf oberster Ebene.

## Funktionen ✨

- Lokale Datensatz-Indizierung mit Größen- und Dateianzahl-Zusammenfassungen
- Rekursive Dateiauflistung pro Datensatz mit abgeleitetem Dateityp
- Vorschau-Unterstützung für CSV/TSV/XLS/XLSX-Tabellen
- Vorschau-Unterstützung für TIFF/JPG/PNG-Bilder
- Vorschau-Unterstützung für `.h5ad`-Zusammenfassungen mit Embedding-/PCA-Scatter-Vorschauerzeugung
- Vorschau-Unterstützung für ZIP/TAR/TGZ-Archivauflistung + Versuch einer Erstbild-Vorschau
- Vorschau-Unterstützung für `.gz`-Textvorschau der ersten Zeilen
- Archiv-Extraktions-Endpoint für große paketierte Datensätze
- Datensatz-Metadatenkarten, die aus Markdown gerendert werden
- PWA-Frontend mit Service Worker und Manifest
- Grundlegende Pfad-Sanitization (`safe_dataset_path`), um Dateizugriffe auf `datasets/` zu beschränken

### Auf einen Blick

| Bereich | Was es bereitstellt |
|---|---|
| Datensatzerkennung | Datensatzauflistung auf Verzeichnisebene mit Dateianzahlen und Größenzusammenfassungen |
| Datei-Exploration | Rekursive Auflistung und Typableitung (`image`, `table`, `analysis`, `archive`, usw.) |
| Umfangreiche Vorschauen | Tabellen, TIFF/Bilder, gzip-Textausschnitte, Archivinhalte, AnnData-Zusammenfassungen |
| Analyse-Visualisierungen | `.h5ad`-Scatter-Vorschauen aus `obsm`-Embeddings oder PCA-Fallback |
| Paketierungsunterstützung | Archivauflistung + Extraktions-Endpoint für große komprimierte Bündel |
| Web-UX | Installierbare PWA mit offline-freundlichen Service-Worker-Assets |

## Projektstruktur 🗂️

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
├─ datasets/                      # heruntergeladene Daten und Vorschau-Cache (git-ignoriert)
├─ metadata/
│  └─ zenodo_10643410.md
├─ papers/                        # Submodul: prompt-is-all-you-need
├─ i18n/                          # derzeit vorhanden für mehrsprachige README-Dateien
├─ BioAgent/                      # verwandte, aber separate App
├─ BioAgentUtils/                 # verwandte Trainings-/Daten-Utilities
├─ references/
├─ results/
└─ vendor/                        # externe Submodule (copilot-sdk, paper-agent, codex)
```

## Voraussetzungen ✅

- Python `3.10+`
- Empfohlener Umgebungsmanager: `conda` oder `venv`

Erforderliche/optionale Python-Pakete, aus dem Quellcode abgeleitet:

| Paket | Rolle |
|---|---|
| `tornado` | Erforderlich für den Serverstart |
| `pandas` | Optional: Tabellenvorschau-Unterstützung |
| `anndata`, `numpy` | Optional: `.h5ad`-Vorschau und Analyse-Plotting |
| `Pillow` | Optional: Bild-Rendering und generierte Vorschauen |
| `tifffile` | Optional: TIFF-Vorschau-Unterstützung |
| `requests` | Optional: Datensatz-Download-Skripte |
| `kaggle` | Optional: Kaggle-Downloads im Drug-Screening-Skript |

Annahmenhinweis: Es gibt derzeit kein `requirements.txt`, `pyproject.toml` oder `environment.yml` im Repository-Root für die Top-Level-App.

## Installation ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Option A: conda (Beispiel)
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# Option B: nur minimale Laufzeit
pip install tornado
```

## Nutzung 🚀

### Schnellstart

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # optional, wenn die Abhängigkeiten bereits vorhanden sind
python app.py --port 8080
```

Öffne `http://localhost:8080`.

### API-Smoke-Test

```bash
curl http://localhost:8080/api/datasets
```

### Daten herunterladen (optional)

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

Heruntergeladene Daten liegen unter `datasets/` (git-ignoriert).

## API-Endpoints 🌐

| Method | Endpoint | Zweck |
|---|---|---|
| `GET` | `/api/datasets` | Datensätze mit zusammenfassenden Statistiken auflisten |
| `GET` | `/api/datasets/{name}` | Dateien für einen Datensatz auflisten |
| `GET` | `/api/datasets/{name}/metadata` | Markdown-Metadatenkarte zurückgeben |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | Kategorienorientierte Dateiauflistung |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | Dateitypabhängige Vorschau-Payload |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | Archiv in benachbarten `_extracted`-Ordner extrahieren |
| `GET` | `/files/<path>` | Rohdatei-Auslieferung von Datensätzen |
| `GET` | `/previews/<path>` | Auslieferung generierter Vorschau-Assets |

Beispielaufruf für Vorschau:

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## Konfiguration 🧩

Die aktuelle Laufzeitkonfiguration ist absichtlich klein gehalten:

- Server-Port: `--port`-Argument in `app.py` (Standard `8080`)
- Datenverzeichnis: fest auf `datasets/` relativ zum Repository-Root
- Vorschau-Cache: `datasets/.cache/previews`
- Metadaten-Zuordnung: `DATASET_METADATA`-Dictionary in `app.py`
- GitHub-API-Token für Downloader (optional): `GITHUB_TOKEN`-Umgebungsvariable oder `--github-token`

Annahmenhinweis: Wenn du konfigurierbare Dataset-Roots oder Produktions-Servereinstellungen brauchst, sind diese derzeit noch nicht in Top-Level-Konfigurationsdateien verfügbar.

## Beispiele 🧪

### Kategorie-spezifische Dateien durchsuchen

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### Ein Archiv extrahieren

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### Selektive Download-Modi ausführen

```bash
# Organoid-Datensätze: GEO überspringen, Zenodo behalten
python scripts/download_organoid_datasets.py --skip-geo

# Drug-Screening-Datensätze: nur Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## Entwicklungsnotizen 🛠️

- Das Backend liefert statische Frontend-Assets aus `web/` aus.
- Service Worker und Manifest liegen in `web/sw.js` und `web/manifest.json`.
- Dateityp-Routing und Vorschauen sind in `app.py` implementiert.
- Manuelle Validierung (aktuelle Projektvorgabe): PWA lädt unter `http://localhost:8080`
- Manuelle Validierung (aktuelle Projektvorgabe): `/api/datasets` liefert JSON zurück
- Manuelle Validierung (aktuelle Projektvorgabe): Vorschauen funktionieren für CSV/XLSX/Bilder/Archive

## Fehlerbehebung 🩺

- `ModuleNotFoundError` für Vorschau-Bibliotheken: fehlende Pakete installieren (`pandas`, `anndata`, `numpy`, `Pillow`, `tifffile`).
- Leere Datensatzauflistung: prüfen, ob Daten unter `datasets/` vorhanden sind und Verzeichnisse nicht mit Punktpräfix beginnen.
- `.h5ad`-Vorschau ohne Scatter-Bild: prüfen, ob `anndata`, `numpy` und `Pillow` installiert sind.
- Probleme bei großen Archiv-Vorschauen/-Extraktionen: Extraktions-Endpoint verwenden und extrahierte Dateien direkt prüfen.
- GitHub-Downloader-Ratenlimitfehler: `GITHUB_TOKEN` über Umgebungsvariable oder CLI-Flag bereitstellen.
- Kaggle-Download funktioniert nicht: `kaggle` installieren und Credentials in `~/.kaggle/kaggle.json` konfigurieren.

## Roadmap 🧭

Mögliche nächste Verbesserungen (in dieser Root-App noch nicht vollständig umgesetzt):

- Root-Abhängigkeitsmanifest hinzufügen (`requirements.txt` oder `pyproject.toml`)
- Automatisierte Tests für API-Handler und Vorschaufunktionen hinzufügen
- Konfigurierbares Datensatz-Root und Cache-Einstellungen hinzufügen
- Explizites Produktions-Run-Profil hinzufügen (non-debug, Reverse-Proxy-Hinweise)
- Mehrsprachige Dokumentation unter `i18n/` ausbauen

## Mitwirken 🤝

Beiträge sind willkommen. Ein praktischer Workflow:

1. Fork erstellen und einen fokussierten Branch anlegen.
2. Änderungen auf einen logischen Bereich begrenzen.
3. App-Start und zentrale Endpoints manuell validieren.
4. PR mit Zusammenfassung, ausgeführten Befehlen und Screenshots für UI-Änderungen öffnen.

Lokale Stilkonventionen in diesem Repository:

- Python: 4 Leerzeichen Einrückung, snake_case für Funktionen/Dateien, CapWords für Klassen
- Frontend-Logik für diese App in `web/app.js` halten (unnötige Framework-Rewrites vermeiden)
- Kommentare knapp halten und nur bei nicht offensichtlicher Logik ergänzen

## Projektlayout (kanonische Zusammenfassung) 📌

- `app.py`: Tornado-Server und API-Routen.
- `web/`: PWA-Assets.
- `scripts/`: Datensatz-Download-Helfer.
- `datasets/`: lokaler Datenspeicher.
- `papers/`: Submodul mit Referenzmaterialien.

## Lizenz 📄

Im Repository-Root ist derzeit keine Top-Level-`LICENSE`-Datei vorhanden.

Annahmenhinweis: Bis eine Root-Lizenz hinzugefügt wird, gelten Wiederverwendungs-/Weiterverteilungsbedingungen für die Top-Level-Codebasis von OrganoidAgent als nicht spezifiziert.

## Sponsor & Spenden ❤️

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Spenden: https://chat.lazying.art/donate
- PayPal: https://paypal.me/RongzhouChen
- Stripe: https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400
