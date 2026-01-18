#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import mimetypes
import os
import tarfile
import time
import zipfile
from pathlib import Path

import tornado.ioloop
import tornado.web

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional at runtime
    pd = None

try:
    import anndata as ad
except Exception:  # pragma: no cover - optional at runtime
    ad = None

try:
    import numpy as np
except Exception:  # pragma: no cover - optional at runtime
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional at runtime
    Image = None

try:
    import tifffile
except Exception:  # pragma: no cover - optional at runtime
    tifffile = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
WEB_DIR = BASE_DIR / "web"
CACHE_DIR = DATA_DIR / ".cache"
PREVIEW_DIR = CACHE_DIR / "previews"


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TABLE_EXTS = {".csv", ".tsv", ".xlsx", ".xls"}
ANALYSIS_EXTS = {".h5ad", ".h5", ".hdf5"}
ARCHIVE_EXTS = {".zip", ".tgz", ".tar", ".tar.gz"}
DOC_EXTS = {".pdf", ".docx"}


def format_bytes(num):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}PB"


def safe_dataset_path(rel_path):
    rel_path = rel_path.lstrip("/").replace("..", "")
    full = (DATA_DIR / rel_path).resolve()
    if not str(full).startswith(str(DATA_DIR.resolve())):
        raise ValueError("Invalid path")
    return full


def file_kind(path):
    ext = path.suffix.lower()
    if path.name.endswith(".tar.gz"):
        return "archive"
    if ext == ".gz":
        return "gzip"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in TABLE_EXTS:
        return "table"
    if ext in ANALYSIS_EXTS:
        return "analysis"
    if ext in DOC_EXTS:
        return "document"
    if ext in ARCHIVE_EXTS:
        return "archive"
    return "file"


def list_datasets():
    datasets = []
    for entry in sorted(DATA_DIR.iterdir()):
        if not entry.is_dir():
            continue
        size = 0
        count = 0
        for path in entry.rglob("*"):
            if path.is_file():
                count += 1
                size += path.stat().st_size
        datasets.append(
            {
                "name": entry.name,
                "path": entry.name,
                "file_count": count,
                "size_bytes": size,
                "size_human": format_bytes(size),
            }
        )
    return datasets


def list_files(dataset_path):
    files = []
    for path in sorted(dataset_path.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(DATA_DIR)
        files.append(
            {
                "name": path.name,
                "path": str(rel),
                "size_bytes": path.stat().st_size,
                "size_human": format_bytes(path.stat().st_size),
                "kind": file_kind(path),
                "ext": path.suffix.lower(),
            }
        )
    return files


def list_archive(path, limit=200):
    entries = []
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist()[:limit]:
                entries.append(
                    {
                        "name": info.filename,
                        "size_bytes": info.file_size,
                        "size_human": format_bytes(info.file_size),
                    }
                )
    elif path.suffix.lower() in {".tgz", ".tar"} or path.name.endswith(".tar.gz"):
        mode = "r:gz" if path.name.endswith(".tar.gz") or path.suffix == ".tgz" else "r"
        with tarfile.open(path, mode) as tf:
            for member in tf.getmembers()[:limit]:
                entries.append(
                    {
                        "name": member.name,
                        "size_bytes": member.size,
                        "size_human": format_bytes(member.size),
                    }
                )
    elif path.suffix.lower() == ".gz":
        entries.append({"name": path.name, "size_bytes": path.stat().st_size})
    return entries


def preview_table(path, max_rows=50):
    if pd is None:
        return {"error": "pandas not installed"}
    ext = path.suffix.lower()
    if ext == ".tsv":
        df = pd.read_csv(path, sep="\t", nrows=max_rows)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path, nrows=max_rows)
    else:
        df = pd.read_csv(path, nrows=max_rows, engine="python")
    return {
        "columns": list(df.columns),
        "rows": df.fillna("").values.tolist(),
    }


def preview_anndata(path):
    if ad is None:
        return {"error": "anndata not installed"}
    data = ad.read_h5ad(path, backed="r")
    return {
        "n_obs": int(data.n_obs),
        "n_vars": int(data.n_vars),
        "obs_columns": list(data.obs_keys())[:50],
        "var_columns": list(data.var_keys())[:50],
        "uns_keys": list(data.uns_keys())[:50],
    }


def preview_text_gz(path, max_lines=50):
    lines = []
    with gzip.open(path, "rt", errors="ignore") as fh:
        for _ in range(max_lines):
            line = fh.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    return {"lines": lines}


def ensure_preview_dir():
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def preview_tiff(path):
    if tifffile is None or Image is None or np is None:
        return {"error": "tifffile/Pillow/numpy not installed"}
    ensure_preview_dir()
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    preview_name = f"{digest}.png"
    preview_path = PREVIEW_DIR / preview_name
    if not preview_path.exists():
        arr = tifffile.imread(path)
        if arr.ndim > 2:
            arr = arr[..., 0]
        arr = arr.astype("float32")
        arr -= arr.min()
        denom = arr.max() if arr.max() else 1.0
        arr = (arr / denom * 255.0).clip(0, 255).astype("uint8")
        image = Image.fromarray(arr)
        image.thumbnail((1024, 1024))
        image.save(preview_path)
    return {"preview_url": f"/previews/{preview_name}"}


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        index_path = WEB_DIR / "index.html"
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.write(index_path.read_text())


class DatasetsHandler(tornado.web.RequestHandler):
    def get(self):
        self.write({"datasets": list_datasets()})


class DatasetFilesHandler(tornado.web.RequestHandler):
    def get(self, dataset_name):
        dataset_path = safe_dataset_path(dataset_name)
        if not dataset_path.exists():
            self.set_status(404)
            self.write({"error": "Dataset not found"})
            return
        self.write({"files": list_files(dataset_path)})


class CategoryHandler(tornado.web.RequestHandler):
    def get(self, category):
        category = category.lower()
        results = []
        for entry in list_datasets():
            dataset_path = safe_dataset_path(entry["path"])
            for info in list_files(dataset_path):
                kind = info["kind"]
                if category == "datasets":
                    results.append(info)
                elif category == "segmentation" and kind in {"image", "archive"}:
                    results.append(info)
                elif category == "features" and kind in {"table", "gzip"}:
                    results.append(info)
                elif category == "analysis" and kind == "analysis":
                    results.append(info)
        self.write({"files": results})


class PreviewHandler(tornado.web.RequestHandler):
    def get(self):
        rel_path = self.get_argument("path")
        path = safe_dataset_path(rel_path)
        if not path.exists():
            self.set_status(404)
            self.write({"error": "File not found"})
            return

        kind = file_kind(path)
        payload = {
            "name": path.name,
            "path": rel_path,
            "kind": kind,
            "size_bytes": path.stat().st_size,
            "size_human": format_bytes(path.stat().st_size),
        }

        if kind == "table":
            payload["preview"] = preview_table(path)
        elif kind == "analysis":
            payload["preview"] = preview_anndata(path)
        elif kind == "archive":
            payload["preview"] = {
                "entries": list_archive(path),
                "note": "Use Extract to unpack large archives.",
            }
        elif kind == "gzip":
            payload["preview"] = preview_text_gz(path)
        elif kind == "document":
            payload["preview"] = {"download_url": f"/files/{rel_path}"}
        elif kind == "image":
            if path.suffix.lower() in {".tif", ".tiff"}:
                payload["preview"] = preview_tiff(path)
            else:
                payload["preview"] = {"preview_url": f"/files/{rel_path}"}
        else:
            payload["preview"] = {"download_url": f"/files/{rel_path}"}

        self.write(payload)


class ExtractHandler(tornado.web.RequestHandler):
    def post(self):
        rel_path = self.get_argument("path")
        path = safe_dataset_path(rel_path)
        if not path.exists():
            self.set_status(404)
            self.write({"error": "File not found"})
            return

        target_dir = path.parent / f"{path.stem}_extracted"
        target_dir.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                zf.extractall(target_dir)
        elif path.suffix.lower() in {".tgz", ".tar"} or path.name.endswith(".tar.gz"):
            mode = "r:gz" if path.name.endswith(".tar.gz") or path.suffix == ".tgz" else "r"
            with tarfile.open(path, mode) as tf:
                tf.extractall(target_dir)
        else:
            self.set_status(400)
            self.write({"error": "Unsupported archive type"})
            return

        rel_target = target_dir.relative_to(DATA_DIR)
        self.write({"extracted_to": str(rel_target)})


def make_app():
    return tornado.web.Application(
        [
            (r"/", IndexHandler),
            (r"/api/datasets", DatasetsHandler),
            (r"/api/datasets/([^/]+)", DatasetFilesHandler),
            (r"/api/category/(datasets|segmentation|features|analysis)", CategoryHandler),
            (r"/api/preview", PreviewHandler),
            (r"/api/extract", ExtractHandler),
            (r"/files/(.*)", tornado.web.StaticFileHandler, {"path": str(DATA_DIR)}),
            (r"/previews/(.*)", tornado.web.StaticFileHandler, {"path": str(PREVIEW_DIR)}),
            (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": str(WEB_DIR)}),
        ],
        debug=True,
    )


def main():
    parser = argparse.ArgumentParser(description="OrganoidAgent Tornado app")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    app = make_app()
    app.listen(args.port)
    print(f"OrganoidAgent running on http://localhost:{args.port}")
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
