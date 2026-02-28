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

OrganoidAgent は、オルガノイドデータセットをローカルで閲覧・プレビューするための軽量な Tornado バックエンド + Progressive Web App (PWA) フロントエンドです。テーブル、顕微鏡画像（TIFF を含む）、アーカイブ、gzip テキストファイル、AnnData `.h5ad` 解析オブジェクトに対して、実用的でファイル種別に応じたプレビューを提供します。

## 概要 🔭

コアアプリは、最小限のセットアップで対話的なデータセット探索ができるよう設計されています。

- バックエンド API とプレビューエンジンは `app.py`
- PWA フロントエンドは `web/`
- ダウンロード補助スクリプトは `scripts/`
- ローカルデータセット作業領域は `datasets/`（git 無視）

このリポジトリには、関連する研究・ユーティリティ作業領域（`BioAgent`、`BioAgentUtils`、`references`、`results`、`vendor`、`papers` サブモジュール）も含まれます。この README で主に説明するランタイムは、トップレベルの `OrganoidAgent` アプリです。

## 機能 ✨

- サイズとファイル数の要約付きローカルデータセット索引
- ファイル種別推定付きの再帰的なデータセットファイル一覧
- CSV/TSV/XLS/XLSX テーブルのプレビュー対応
- TIFF/JPG/PNG 画像のプレビュー対応
- `.h5ad` サマリーと、埋め込み/PCA 散布図プレビュー生成に対応
- ZIP/TAR/TGZ アーカイブ一覧 + 先頭画像プレビュー試行に対応
- `.gz` テキスト先頭行プレビューに対応
- 大規模パッケージ済みデータセット向けアーカイブ展開エンドポイント
- Markdown からレンダリングされるデータセット単位のメタデータカード
- サービスワーカーとマニフェストを備えた PWA フロントエンド
- `datasets/` 配下にファイルアクセスを限定する基本的なパスサニタイズ（`safe_dataset_path`）

### ひと目で分かるポイント

| 領域 | 提供内容 |
|---|---|
| データセット検出 | ファイル数とサイズ要約を含むディレクトリ単位のデータセット一覧 |
| ファイル探索 | 再帰一覧と種別推定（`image`, `table`, `analysis`, `archive` など） |
| 高機能プレビュー | テーブル、TIFF/画像、gzip テキスト断片、アーカイブ内容、AnnData サマリー |
| 解析可視化 | `obsm` 埋め込み、または PCA フォールバックからの `.h5ad` 散布図プレビュー |
| パッケージ対応 | 大きな圧縮バンドル向けのアーカイブ一覧 + 展開エンドポイント |
| Web UX | オフラインフレンドリーなサービスワーカー資産を備えたインストール可能 PWA |

## プロジェクト構成 🗂️

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
├─ datasets/                      # ダウンロード済みデータとプレビューキャッシュ（git 無視）
├─ metadata/
│  └─ zenodo_10643410.md
├─ papers/                        # サブモジュール: prompt-is-all-you-need
├─ i18n/                          # 多言語 README ファイル用（現在存在）
├─ BioAgent/                      # 関連するが別アプリ
├─ BioAgentUtils/                 # 関連する学習/データユーティリティ
├─ references/
├─ results/
└─ vendor/                        # 外部サブモジュール（copilot-sdk, paper-agent, codex）
```

## 前提条件 ✅

- Python `3.10+`
- 推奨環境管理: `conda` または `venv`

ソースから推定される必須/任意 Python パッケージ:

| パッケージ | 役割 |
|---|---|
| `tornado` | サーバ起動に必須 |
| `pandas` | 任意: テーブルプレビュー対応 |
| `anndata`, `numpy` | 任意: `.h5ad` プレビューと解析プロット |
| `Pillow` | 任意: 画像レンダリングと生成プレビュー |
| `tifffile` | 任意: TIFF プレビュー対応 |
| `requests` | 任意: データセットダウンロードスクリプト |
| `kaggle` | 任意: 薬剤スクリーニングスクリプトでの Kaggle ダウンロード |

想定メモ: 現時点でトップレベルアプリにはルート `requirements.txt`、`pyproject.toml`、`environment.yml` がありません。

## インストール ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Option A: conda (example)
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# Option B: minimal runtime only
pip install tornado
```

## 使い方 🚀

### クイックスタート

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # optional if you already have the deps
python app.py --port 8080
```

`http://localhost:8080` を開いてください。

### API スモークテスト

```bash
curl http://localhost:8080/api/datasets
```

### データのダウンロード（任意）

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

ダウンロードしたデータは `datasets/`（git 無視）に保存されます。

## API エンドポイント 🌐

| Method | Endpoint | 目的 |
|---|---|---|
| `GET` | `/api/datasets` | 要約統計付きでデータセット一覧を取得 |
| `GET` | `/api/datasets/{name}` | 1つのデータセットのファイル一覧を取得 |
| `GET` | `/api/datasets/{name}/metadata` | Markdown メタデータカードを返す |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | カテゴリ指向のファイル一覧 |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | ファイル種別に応じたプレビューペイロード |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | アーカイブを同階層の `_extracted` フォルダへ展開 |
| `GET` | `/files/<path>` | 生のデータセットファイル配信 |
| `GET` | `/previews/<path>` | 生成済みプレビュー資産配信 |

プレビュー呼び出し例:

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## 設定 🧩

現在のランタイム設定は意図的にシンプルです。

- サーバポート: `app.py` の `--port` 引数（デフォルト `8080`）
- データディレクトリ: リポジトリルート相対の `datasets/` に固定
- プレビューキャッシュ: `datasets/.cache/previews`
- メタデータマッピング: `app.py` の `DATASET_METADATA` 辞書
- ダウンローダー用 GitHub API トークン（任意）: `GITHUB_TOKEN` 環境変数または `--github-token`

想定メモ: データセットルートの可変化や本番向けサーバ設定が必要な場合、現時点ではトップレベル設定ファイルとして公開されていません。

## 例 🧪

### カテゴリ別ファイルを閲覧

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### アーカイブを展開

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### 選択的ダウンロードモードを実行

```bash
# Organoid datasets: skip GEO, keep Zenodo
python scripts/download_organoid_datasets.py --skip-geo

# Drug-screening datasets: only Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## 開発メモ 🛠️

- バックエンドは `web/` からフロントエンド静的アセットを配信します。
- サービスワーカーとマニフェストは `web/sw.js` と `web/manifest.json` にあります。
- ファイル種別ルーティングとプレビューは `app.py` に実装されています。
- 手動検証（現行プロジェクト指針）: PWA が `http://localhost:8080` で読み込まれること
- 手動検証（現行プロジェクト指針）: `/api/datasets` が JSON を返すこと
- 手動検証（現行プロジェクト指針）: CSV/XLSX/画像/アーカイブのプレビューが描画されること

## トラブルシューティング 🩺

- プレビューライブラリの `ModuleNotFoundError`: 不足パッケージ（`pandas`, `anndata`, `numpy`, `Pillow`, `tifffile`）をインストールしてください。
- データセット一覧が空: `datasets/` 配下にデータがあり、ディレクトリ名がドット始まりでないことを確認してください。
- `.h5ad` プレビューで散布図画像が出ない: `anndata`, `numpy`, `Pillow` がインストールされているか確認してください。
- 大きなアーカイブのプレビュー/展開で問題がある: 展開エンドポイントを使い、展開後ファイルを直接確認してください。
- GitHub ダウンローダーでレート制限エラー: `GITHUB_TOKEN` を環境変数または CLI フラグで渡してください。
- Kaggle ダウンロードが動作しない: `kaggle` をインストールし、`~/.kaggle/kaggle.json` 認証情報を設定してください。

## ロードマップ 🧭

今後の改善候補（このルートアプリでは未実装または未完了）:

- ルート依存関係マニフェスト（`requirements.txt` または `pyproject.toml`）を追加
- API ハンドラとプレビュー関数の自動テストを追加
- データセットルートとキャッシュ設定の可変化
- 明示的な本番実行プロファイル（非デバッグ、リバースプロキシ指針）を追加
- `i18n/` 配下の多言語ドキュメントを拡張

## コントリビュート 🤝

コントリビューション歓迎です。実践的なワークフロー:

1. フォークして焦点を絞ったブランチを作成する。
2. 変更は 1 つの論理領域にスコープする。
3. アプリ起動と主要エンドポイントを手動検証する。
4. 要約、実行コマンド、UI 変更時はスクリーンショット付きで PR を作成する。

このリポジトリのローカルスタイル規約:

- Python: 4 スペースインデント、関数/ファイルは snake_case、クラスは CapWords
- このアプリではフロントエンドロジックを `web/app.js` に集約（不要なフレームワーク書き換えは避ける）
- コメントは簡潔にし、ロジックが自明でない箇所のみに付与

## プロジェクトレイアウト（正準サマリー） 📌

- `app.py`: Tornado サーバーと API ルート。
- `web/`: PWA アセット。
- `scripts/`: データセットダウンロード補助。
- `datasets/`: ローカルデータ保存。
- `papers/`: 参考資料を含むサブモジュール。

## ライセンス 📄

現在、このリポジトリルートにはトップレベルの `LICENSE` ファイルがありません。

想定メモ: ルートライセンスが追加されるまでは、トップレベル OrganoidAgent コードベースの再利用/再配布条件は未指定として扱ってください。

## スポンサー & 寄付 ❤️

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Donate: https://chat.lazying.art/donate
- PayPal: https://paypal.me/RongzhouChen
- Stripe: https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400
