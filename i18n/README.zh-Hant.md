[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

**語言：** 中文（繁體）（本檔案） | `i18n/` 目錄用於存放其他語言版本的 README（不重複語言導覽列）。

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

OrganoidAgent 是一個輕量級 Tornado 後端 + 漸進式網頁應用（PWA）前端，用於在本機瀏覽與預覽類器官資料集。它支援實用且可依檔案類型判斷的預覽能力，包括表格、顯微影像（含 TIFF）、壓縮檔、gzip 文字檔，以及 AnnData `.h5ad` 分析物件。

## 概覽 🔭

核心應用以低門檻部署為前提，針對互動式資料集探索設計：

- 後端 API 與預覽引擎位於 `app.py`
- PWA 前端位於 `web/`
- 下載輔助腳本位於 `scripts/`
- 本機資料集工作區位於 `datasets/`（git-ignored）

本儲存庫也包含相鄰的研究與工具工作區（`BioAgent`、`BioAgentUtils`、`references`、`results`、`vendor`、`papers` 子模組）。本 README 主要說明頂層 `OrganoidAgent` 應用的執行方式。

## 功能特色 ✨

- 本機資料集索引，提供大小與檔案數摘要
- 遞迴列出資料集檔案並推斷檔案類型
- 預覽支援 CSV/TSV/XLS/XLSX 表格
- 預覽支援 TIFF/JPG/PNG 影像
- 預覽支援 `.h5ad` 摘要，並可產生 embedding/PCA 散點預覽
- 預覽支援 ZIP/TAR/TGZ 壓縮檔清單 + 首張影像預覽嘗試
- 預覽支援 `.gz` 文字前幾行
- 提供大型封裝資料集的壓縮檔解壓端點
- 從 Markdown 渲染資料集層級的中繼資料卡片
- 具備 service worker 與 manifest 的 PWA 前端
- 基礎路徑淨化（`safe_dataset_path`），將檔案存取限制在 `datasets/` 之下

### 快速一覽

| 區域 | 提供內容 |
|---|---|
| 資料集探索 | 以目錄層級列出資料集，含檔案數與大小摘要 |
| 檔案探索 | 遞迴列出與類型推斷（`image`、`table`、`analysis`、`archive` 等） |
| 豐富預覽 | 表格、TIFF/影像、gzip 文字片段、壓縮檔內容、AnnData 摘要 |
| 分析視覺化 | 從 `obsm` embeddings 或 PCA 備援產生 `.h5ad` 散點預覽 |
| 封裝支援 | 壓縮檔清單 + 針對大型壓縮資料的解壓端點 |
| Web 體驗 | 可安裝的 PWA，搭配離線友好的 service worker 靜態資源 |

## 專案結構 🗂️

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

## 前置需求 ✅

- Python `3.10+`
- 建議的環境管理器：`conda` 或 `venv`

從原始碼推斷的必要/可選 Python 套件：

| 套件 | 角色 |
|---|---|
| `tornado` | 啟動伺服器所需 |
| `pandas` | 可選：表格預覽支援 |
| `anndata`, `numpy` | 可選：`.h5ad` 預覽與分析繪圖 |
| `Pillow` | 可選：影像渲染與預覽生成 |
| `tifffile` | 可選：TIFF 預覽支援 |
| `requests` | 可選：資料集下載腳本 |
| `kaggle` | 可選：藥篩腳本中的 Kaggle 下載 |

假設說明：目前頂層應用尚無根目錄 `requirements.txt`、`pyproject.toml` 或 `environment.yml`。

## 安裝 ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Option A: conda (example)
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# Option B: minimal runtime only
pip install tornado
```

## 使用方式 🚀

### 快速開始

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # optional if you already have the deps
python app.py --port 8080
```

開啟 `http://localhost:8080`。

### API 冒煙測試

```bash
curl http://localhost:8080/api/datasets
```

### 下載資料（可選）

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

下載的資料位於 `datasets/`（git-ignored）。

## API 端點 🌐

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/datasets` | 列出資料集與摘要統計 |
| `GET` | `/api/datasets/{name}` | 列出單一資料集檔案 |
| `GET` | `/api/datasets/{name}/metadata` | 回傳 Markdown 中繼資料卡片 |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | 依類別列出檔案 |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | 回傳依檔案類型處理的預覽 payload |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | 將壓縮檔解壓至同層 `_extracted` 目錄 |
| `GET` | `/files/<path>` | 原始資料集檔案服務 |
| `GET` | `/previews/<path>` | 已生成預覽資產服務 |

範例預覽呼叫：

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## 設定 🧩

目前執行期設定刻意保持精簡：

- 伺服器連接埠：`app.py` 中的 `--port` 參數（預設 `8080`）
- 資料目錄：固定為儲存庫根目錄下的 `datasets/`
- 預覽快取：`datasets/.cache/previews`
- 中繼資料對應：`app.py` 中的 `DATASET_METADATA` 字典
- 下載器的 GitHub API token（可選）：環境變數 `GITHUB_TOKEN` 或 `--github-token`

假設說明：若你需要可設定的資料集根目錄或正式環境伺服器設定，這些能力目前尚未在頂層設定檔中公開。

## 範例 🧪

### 瀏覽特定類別檔案

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### 解壓壓縮檔

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### 執行選擇性下載模式

```bash
# Organoid datasets: skip GEO, keep Zenodo
python scripts/download_organoid_datasets.py --skip-geo

# Drug-screening datasets: only Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## 開發說明 🛠️

- 後端會從 `web/` 提供前端靜態資產。
- Service worker 與 manifest 位於 `web/sw.js` 和 `web/manifest.json`。
- 檔案類型路由與預覽邏輯實作於 `app.py`。
- 手動驗證（目前專案指引）：PWA 可在 `http://localhost:8080` 正常載入
- 手動驗證（目前專案指引）：`/api/datasets` 會回傳 JSON
- 手動驗證（目前專案指引）：CSV/XLSX/影像/壓縮檔預覽可正常渲染

## 疑難排解 🩺

- 預覽函式庫 `ModuleNotFoundError`：安裝缺少的套件（`pandas`、`anndata`、`numpy`、`Pillow`、`tifffile`）。
- 資料集清單為空：確認 `datasets/` 底下有資料，且目錄名稱不是 dot-prefixed。
- `.h5ad` 預覽缺少散點圖：檢查是否已安裝 `anndata`、`numpy`、`Pillow`。
- 大型壓縮檔預覽/解壓問題：使用解壓端點並直接檢查解壓後的檔案。
- GitHub 下載器速率限制錯誤：透過環境變數或 CLI 參數提供 `GITHUB_TOKEN`。
- Kaggle 下載無法運作：安裝 `kaggle`，並在 `~/.kaggle/kaggle.json` 設定憑證。

## 路線圖 🧭

潛在後續改善（在此頂層應用中尚未完整實作）：

- 新增根層級相依性清單（`requirements.txt` 或 `pyproject.toml`）
- 為 API handlers 與預覽函式新增自動化測試
- 新增可設定的資料集根目錄與快取設定
- 新增明確的正式環境執行設定（非 debug、反向代理指引）
- 在 `i18n/` 下擴充多語系文件

## 貢獻 🤝

歡迎貢獻。建議的實務流程：

1. Fork 並建立聚焦分支。
2. 將變更聚焦在單一邏輯範圍。
3. 手動驗證應用啟動與關鍵端點。
4. 建立 PR，附上摘要、執行過的命令與 UI 變更截圖。

本儲存庫的本地風格慣例：

- Python：4 空白縮排，函式/檔名使用 snake_case，類別名稱使用 CapWords
- 本應用前端邏輯維持在 `web/app.js`（避免不必要的框架重寫）
- 註解保持精簡，僅在邏輯不直觀時補充

## 專案佈局（Canonical Summary） 📌

- `app.py`：Tornado 伺服器與 API 路由。
- `web/`：PWA 資產。
- `scripts/`：資料集下載輔助腳本。
- `datasets/`：本機資料儲存。
- `papers/`：包含參考資料的子模組。

## 授權 📄

目前此儲存庫根目錄尚未提供頂層專案 `LICENSE` 檔案。

假設說明：在新增根層級授權之前，頂層 OrganoidAgent 程式碼庫的重用/再散佈條款可視為未明確定義。

## 贊助與捐贈 ❤️

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Donate: https://chat.lazying.art/donate
- PayPal: https://paypal.me/RongzhouChen
- Stripe: https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400
