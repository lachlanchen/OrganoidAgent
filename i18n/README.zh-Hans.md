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

OrganoidAgent 是一个轻量级的 Tornado 后端 + 渐进式 Web 应用（PWA）前端，用于在本地浏览和预览类器官数据集。它支持实用的、按文件类型感知的预览能力，包括表格、显微图像（含 TIFF）、压缩包、gzip 文本文件，以及 AnnData `.h5ad` 分析对象。

## 概览 🔭

该核心应用面向交互式数据集探索，且部署成本低：

- 后端 API 与预览引擎位于 `app.py`
- PWA 前端位于 `web/`
- 下载辅助脚本位于 `scripts/`
- 本地数据集工作区位于 `datasets/`（已被 git 忽略）

此仓库还包含相邻的研究与工具工作区（`BioAgent`、`BioAgentUtils`、`references`、`results`、`vendor`、`papers` 子模块）。本 README 主要说明的是顶层 `OrganoidAgent` 应用的运行方式。

## 功能特性 ✨

- 本地数据集索引，提供大小与文件数汇总
- 递归列出数据集文件并推断文件类型
- 预览支持 CSV/TSV/XLS/XLSX 表格
- 预览支持 TIFF/JPG/PNG 图像
- 预览支持 `.h5ad` 摘要，并可生成 embedding/PCA 散点预览
- 预览支持 ZIP/TAR/TGZ 压缩包内容列表 + 首张图像预览尝试
- 预览支持 `.gz` 文本前几行
- 为大型打包数据集提供压缩包解压接口
- 从 Markdown 渲染数据集级元数据卡片
- 带有 service worker 与 manifest 的 PWA 前端
- 基础路径净化（`safe_dataset_path`），将文件访问限制在 `datasets/` 下

### 快速一览

| 区域 | 提供内容 |
|---|---|
| 数据集发现 | 基于目录层级的数据集列表，含文件数与大小汇总 |
| 文件探索 | 递归列出与类型推断（`image`、`table`、`analysis`、`archive` 等） |
| 丰富预览 | 表格、TIFF/图像、gzip 文本片段、压缩包内容、AnnData 摘要 |
| 分析可视化 | 从 `obsm` embedding 或 PCA 回退生成 `.h5ad` 散点预览 |
| 打包支持 | 压缩包列表 + 面向大型压缩数据的解压接口 |
| Web 体验 | 可安装的 PWA，配备离线友好的 service worker 静态资源 |

## 项目结构 🗂️

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

## 前置要求 ✅

- Python `3.10+`
- 推荐的环境管理器：`conda` 或 `venv`

根据源码推断的必需/可选 Python 包：

| 包 | 作用 |
|---|---|
| `tornado` | 启动服务器所必需 |
| `pandas` | 可选：表格预览支持 |
| `anndata`, `numpy` | 可选：`.h5ad` 预览与分析绘图 |
| `Pillow` | 可选：图像渲染与预览生成 |
| `tifffile` | 可选：TIFF 预览支持 |
| `requests` | 可选：数据集下载脚本 |
| `kaggle` | 可选：药筛脚本中的 Kaggle 下载 |

假设说明：当前顶层应用尚无根目录 `requirements.txt`、`pyproject.toml` 或 `environment.yml`。

## 安装 ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Option A: conda (example)
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# Option B: minimal runtime only
pip install tornado
```

## 使用 🚀

### 快速开始

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # optional if you already have the deps
python app.py --port 8080
```

打开 `http://localhost:8080`。

### API 冒烟测试

```bash
curl http://localhost:8080/api/datasets
```

### 下载数据（可选）

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

下载的数据位于 `datasets/`（git-ignored）。

## API 端点 🌐

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/datasets` | 列出数据集及汇总统计 |
| `GET` | `/api/datasets/{name}` | 列出单个数据集文件 |
| `GET` | `/api/datasets/{name}/metadata` | 返回 Markdown 元数据卡片 |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | 按类别列出文件 |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | 返回按文件类型适配的预览载荷 |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | 将压缩包解压到同级 `_extracted` 目录 |
| `GET` | `/files/<path>` | 原始数据集文件服务 |
| `GET` | `/previews/<path>` | 已生成预览资源服务 |

示例预览调用：

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## 配置 🧩

当前运行时配置刻意保持精简：

- 服务器端口：`app.py` 中的 `--port` 参数（默认 `8080`）
- 数据目录：固定为仓库根目录下的 `datasets/`
- 预览缓存：`datasets/.cache/previews`
- 元数据映射：`app.py` 中的 `DATASET_METADATA` 字典
- 下载器的 GitHub API token（可选）：环境变量 `GITHUB_TOKEN` 或 `--github-token`

假设说明：如果你需要可配置的数据集根目录或生产级服务器设置，这些能力目前尚未在顶层配置文件中公开。

## 示例 🧪

### 浏览特定类别文件

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### 解压压缩包

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### 运行选择性下载模式

```bash
# Organoid datasets: skip GEO, keep Zenodo
python scripts/download_organoid_datasets.py --skip-geo

# Drug-screening datasets: only Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## 开发说明 🛠️

- 后端从 `web/` 提供前端静态资源。
- Service worker 与 manifest 位于 `web/sw.js` 和 `web/manifest.json`。
- 文件类型路由与预览逻辑实现于 `app.py`。
- 手动验证（当前项目指引）：PWA 可在 `http://localhost:8080` 正常加载
- 手动验证（当前项目指引）：`/api/datasets` 返回 JSON
- 手动验证（当前项目指引）：CSV/XLSX/图像/压缩包预览可正常渲染

## 故障排查 🩺

- 预览库 `ModuleNotFoundError`：安装缺失包（`pandas`、`anndata`、`numpy`、`Pillow`、`tifffile`）。
- 数据集列表为空：确认 `datasets/` 下存在数据，且目录名不是点前缀隐藏目录。
- `.h5ad` 预览缺少散点图：检查是否安装 `anndata`、`numpy`、`Pillow`。
- 大型压缩包预览/解压异常：使用解压接口并直接检查解压后的文件。
- GitHub 下载器触发速率限制：通过环境变量或 CLI 参数提供 `GITHUB_TOKEN`。
- Kaggle 下载不可用：安装 `kaggle`，并在 `~/.kaggle/kaggle.json` 配置凭据。

## 路线图 🧭

潜在的后续改进（在此顶层应用中尚未完全实现）：

- 增加根级依赖清单（`requirements.txt` 或 `pyproject.toml`）
- 为 API 处理器和预览函数增加自动化测试
- 增加可配置的数据集根目录与缓存设置
- 增加明确的生产环境运行配置（非 debug、反向代理指引）
- 在 `i18n/` 下扩展多语言文档

## 贡献 🤝

欢迎贡献。推荐的实用流程：

1. Fork 并创建聚焦分支。
2. 让改动聚焦于一个逻辑领域。
3. 手动验证应用启动与关键端点。
4. 提交 PR，附上摘要、执行过的命令以及 UI 改动截图。

本仓库的本地风格约定：

- Python：4 空格缩进，函数/文件名使用 snake_case，类名使用 CapWords
- 本应用前端逻辑保持在 `web/app.js`（避免不必要的框架重写）
- 注释保持简洁，仅在逻辑不明显时添加

## 项目布局（规范摘要） 📌

- `app.py`：Tornado 服务器与 API 路由。
- `web/`：PWA 资源。
- `scripts/`：数据集下载辅助脚本。
- `datasets/`：本地数据存储。
- `papers/`：包含参考资料的子模块。

## 许可证 📄

当前仓库根目录尚未提供顶层项目 `LICENSE` 文件。

假设说明：在根级许可证补充前，顶层 OrganoidAgent 代码库的复用/再分发条款可视为未明确。

## 赞助与捐赠 ❤️

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Donate: https://chat.lazying.art/donate
- PayPal: https://paypal.me/RongzhouChen
- Stripe: https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400
