#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import io
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
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional at runtime
    Image = None
    ImageDraw = None

try:
    import tifffile
except Exception:  # pragma: no cover - optional at runtime
    tifffile = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
WEB_DIR = BASE_DIR / "web"
METADATA_DIR = BASE_DIR / "metadata"
CACHE_DIR = DATA_DIR / ".cache"
PREVIEW_DIR = CACHE_DIR / "previews"


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TABLE_EXTS = {".csv", ".tsv", ".xlsx", ".xls"}
ANALYSIS_EXTS = {".h5ad", ".h5", ".hdf5"}
ARCHIVE_EXTS = {".zip", ".tgz", ".tar", ".tar.gz"}
DOC_EXTS = {".pdf", ".docx"}
MAX_ARCHIVE_PREVIEW_BYTES = 50 * 1024 * 1024

DATASET_METADATA = {
    "zenodo_10643410": "zenodo_10643410.md",
}


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
        if entry.name.startswith("."):
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


def load_dataset_metadata(dataset_name):
    filename = DATASET_METADATA.get(dataset_name)
    if not filename:
        return None
    path = (METADATA_DIR / filename).resolve()
    if not path.exists():
        return None
    return {"markdown": path.read_text(encoding="utf-8")}


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


def _sample_indices(total, max_items, rng):
    if total <= max_items:
        return np.arange(total)
    return rng.choice(total, size=max_items, replace=False)


def _embedding_from_obsm(data, max_points):
    obsm_keys = list(data.obsm_keys())
    for key in ("X_umap", "X_tsne", "X_pca"):
        if key in obsm_keys:
            emb = data.obsm[key]
            if emb is None:
                continue
            try:
                n_obs = emb.shape[0]
            except Exception:
                emb = np.asarray(emb)
                n_obs = emb.shape[0]
            idx = _sample_indices(n_obs, max_points, np.random.default_rng(0))
            try:
                coords = np.asarray(emb[idx][:, :2])
            except Exception:
                coords = np.asarray(emb)[:, :2]
            return coords, key
    return None, None


def _pca_preview(data, max_points=2000, max_vars=2000):
    if np is None:
        return None
    rng = np.random.default_rng(0)
    obs_idx = _sample_indices(int(data.n_obs), max_points, rng)
    var_idx = _sample_indices(int(data.n_vars), max_vars, rng)
    view = data[obs_idx, var_idx]
    matrix = view.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype="float32")
    if matrix.size == 0:
        return None
    matrix -= matrix.mean(axis=0, keepdims=True)
    try:
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    coords = u[:, :2] * s[:2]
    return coords


def _render_scatter(coords, preview_name, size=900, point_radius=2):
    if Image is None or ImageDraw is None or np is None:
        return None, "Pillow/numpy not installed"
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return None, "Invalid embedding"
    mask = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    coords = coords[mask]
    if coords.size == 0:
        return None, "No finite points"
    ensure_preview_dir()
    preview_path = PREVIEW_DIR / preview_name
    if preview_path.exists():
        return preview_path, None
    x = coords[:, 0]
    y = coords[:, 1]
    min_x, max_x = float(x.min()), float(x.max())
    min_y, max_y = float(y.min()), float(y.max())
    span_x = max_x - min_x or 1.0
    span_y = max_y - min_y or 1.0
    pad = 24
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for px, py in coords:
        sx = pad + (px - min_x) / span_x * (size - 2 * pad)
        sy = pad + (py - min_y) / span_y * (size - 2 * pad)
        left = sx - point_radius
        top = size - sy - point_radius
        right = sx + point_radius
        bottom = size - sy + point_radius
        draw.ellipse((left, top, right, bottom), fill=(40, 102, 194))
    img.save(preview_path)
    return preview_path, None


def preview_anndata(path):
    if ad is None or np is None:
        return {"error": "anndata/numpy not installed"}
    data = ad.read_h5ad(path, backed="r")
    summary = {
        "n_obs": int(data.n_obs),
        "n_vars": int(data.n_vars),
        "obs_columns": list(data.obs_keys())[:50],
        "var_columns": list(data.var_keys())[:50],
        "uns_keys": list(data.uns_keys())[:50],
    }
    coords, source = _embedding_from_obsm(data, max_points=2500)
    if coords is None:
        coords = _pca_preview(data)
        source = "pca_preview" if coords is not None else None
    if coords is not None:
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
        preview_name = f"{digest}-{source}.png"
        preview_path, plot_error = _render_scatter(coords, preview_name)
        if preview_path:
            summary["preview_url"] = f"/previews/{preview_name}"
            summary["preview_points"] = int(coords.shape[0])
            summary["preview_source"] = source
        elif plot_error:
            summary["preview_error"] = plot_error
    else:
        summary["preview_error"] = "No embedding available"
    if getattr(data, "isbacked", False) and getattr(data, "file", None) is not None:
        data.file.close()
    return summary


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


def _is_image_name(name):
    return Path(name).suffix.lower() in IMAGE_EXTS


def _normalize_frame(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[-1] in {3, 4}:
            arr = arr[..., 0]
        elif arr.shape[0] in {3, 4}:
            arr = arr[0]
        else:
            arr = arr[arr.shape[0] // 2]
    elif arr.ndim > 3:
        while arr.ndim > 3:
            arr = arr[arr.shape[0] // 2]
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
    arr = arr.astype("float32")
    arr -= arr.min()
    denom = arr.max() if arr.max() else 1.0
    return (arr / denom * 255.0).clip(0, 255).astype("uint8")


def _read_tiff_frame(source):
    with tifffile.TiffFile(source) as tif:
        series = tif.series[0]
        if series.pages and len(series.pages) > 1:
            page = series.pages[len(series.pages) // 2]
            arr = page.asarray()
        else:
            arr = series.asarray()
    return _normalize_frame(arr)


def _write_preview_image(arr, preview_name):
    ensure_preview_dir()
    preview_path = PREVIEW_DIR / preview_name
    if not preview_path.exists():
        image = Image.fromarray(arr)
        image.thumbnail((1024, 1024))
        image.save(preview_path)
    return preview_path


def preview_tiff(path):
    if tifffile is None or Image is None or np is None:
        return {"error": "tifffile/Pillow/numpy not installed"}
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    preview_name = f"{digest}.png"
    try:
        arr = _read_tiff_frame(path)
        _write_preview_image(arr, preview_name)
    except Exception as exc:
        return {"error": f"tiff preview failed: {exc}"}
    return {"preview_url": f"/previews/{preview_name}"}


def _preview_image_bytes(data, name_hint, seed):
    if Image is None or np is None:
        return {"error": "Pillow/numpy not installed"}
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    preview_name = f"{digest}.png"
    preview_path = PREVIEW_DIR / preview_name
    if preview_path.exists():
        return {"preview_url": f"/previews/{preview_name}"}
    if name_hint.lower().endswith((".tif", ".tiff")) and tifffile is not None:
        arr = _read_tiff_frame(io.BytesIO(data))
        _write_preview_image(arr, preview_name)
        return {"preview_url": f"/previews/{preview_name}"}
    ensure_preview_dir()
    image = Image.open(io.BytesIO(data))
    image.thumbnail((1024, 1024))
    image.save(preview_path)
    return {"preview_url": f"/previews/{preview_name}"}


def preview_archive(path, limit=200):
    preview = {"entries": list_archive(path, limit=limit)}
    if Image is None:
        return preview
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                for info in zf.infolist():
                    if info.is_dir() or not _is_image_name(info.filename):
                        continue
                    if info.file_size > MAX_ARCHIVE_PREVIEW_BYTES:
                        continue
                    with zf.open(info) as fh:
                        data = fh.read()
                    result = _preview_image_bytes(data, info.filename, f"{path}:{info.filename}")
                    if "preview_url" in result:
                        preview.update(result)
                        preview["preview_entry"] = info.filename
                        break
        elif path.suffix.lower() in {".tgz", ".tar"} or path.name.endswith(".tar.gz"):
            mode = "r:gz" if path.name.endswith(".tar.gz") or path.suffix == ".tgz" else "r"
            with tarfile.open(path, mode) as tf:
                for member in tf.getmembers():
                    if not member.isfile() or not _is_image_name(member.name):
                        continue
                    if member.size > MAX_ARCHIVE_PREVIEW_BYTES:
                        continue
                    handle = tf.extractfile(member)
                    if handle is None:
                        continue
                    data = handle.read()
                    result = _preview_image_bytes(data, member.name, f"{path}:{member.name}")
                    if "preview_url" in result:
                        preview.update(result)
                        preview["preview_entry"] = member.name
                        break
    except Exception as exc:
        preview["preview_error"] = str(exc)
    return preview


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


class DatasetMetadataHandler(tornado.web.RequestHandler):
    def get(self, dataset_name):
        metadata = load_dataset_metadata(dataset_name)
        if not metadata:
            self.set_status(404)
            self.write({"error": "Metadata not found"})
            return
        self.write(metadata)


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
            payload["preview"] = preview_archive(path)
            payload["preview"]["note"] = "Use Extract to unpack large archives."
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
            (r"/api/datasets/([^/]+)/metadata", DatasetMetadataHandler),
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
