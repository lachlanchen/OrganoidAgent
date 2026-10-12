#!/usr/bin/env python3
import argparse
import asyncio
import functools
import gzip
import hashlib
import io
import json
import math
import mimetypes
import os
import re
import shutil
import ssl
import sqlite3
import subprocess
import sys
import tarfile
import threading
import time
import uuid
import zipfile
from pathlib import Path


def configure_windows_ca_fallback():
    """Use certifi only when Python cannot parse the Windows certificate store."""
    if os.name != "nt":
        return
    try:
        ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    except ssl.SSLError as exc:
        if "ASN1" not in str(exc) and "NOT_ENOUGH_DATA" not in str(exc):
            raise
        try:
            import certifi
        except ImportError:
            raise RuntimeError(
                "The Windows certificate store is unreadable and certifi is not installed."
            ) from exc

        ca_bundle = certifi.where()

        def load_certifi_defaults(context, purpose=ssl.Purpose.SERVER_AUTH):
            context.load_verify_locations(cafile=ca_bundle)

        ssl.SSLContext.load_default_certs = load_certifi_defaults


configure_windows_ca_fallback()

import tornado.ioloop
import tornado.web


async def run_in_thread(function, *args, **kwargs):
    """Python 3.8-compatible equivalent of asyncio.to_thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(function, *args, **kwargs))

from agent_studio import CodexJobError, CodexJobManager, ParseError, StudioChatStore, parse_aaps, tail_text
from review_store import ReviewStore

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
WORKFLOW_SETTINGS_PATH = BASE_DIR / "config" / "workflows.json"
METADATA_DIR = BASE_DIR / "metadata"
CACHE_DIR = DATA_DIR / ".cache"
PREVIEW_DIR = CACHE_DIR / "previews"
STUDIO_DIR = BASE_DIR / "analysis-outputs" / "organoid_agent_studio"
AGENT_REPORT_DIR = STUDIO_DIR / "reports"
FLUORESCENCE_DIR = BASE_DIR / "fluorescence_prediction"
FLUORESCENCE_OUTPUT_DIR = BASE_DIR / "analysis-outputs" / "fluorescence_prediction" / "web"
PROFILER_OUTPUT_DIR = BASE_DIR / "analysis-outputs" / "organoid_profiler" / "web"
FLUORESCENCE_LOCK = threading.Lock()
PROFILER_LOCK = threading.Lock()
FLUORESCENCE_ACTIVE_JOB_ID = None
FLUORESCENCE_ACTIVE_PROCESS = None
MORPHOLOGY_OUTPUT_DIR = BASE_DIR / "analysis-outputs" / "morphology_jobs"
MORPHOLOGY_LOCK = threading.Lock()
MORPHOLOGY_ACTIVE = {"job_id": None, "process": None}
MORPHOLOGY_TASKS = set()
MORPHOLOGY_WORKFLOW_ADAPTERS = {
    "neo_cpsam": {
        "experiment": "neo_cpsam",
        "dataset_dirs": {"07_NEO"},
        "filename_pattern": r".+_img\.jpg$",
        "extensions": {".jpg", ".jpeg", ".png"},
        "entrypoint": "scripts/run_neo_cpsam_morphology.py",
    },
    "zhengyu": {
        "experiment": "density",
        "dataset_dirs": {"01_Density_experiment_10x"},
        "filename_pattern": r"(?:low|middle|high).*_D\d{2}_",
        "entrypoint": "scripts/run_deo_original_managed.py",
        "entrypoint_args": ["--workflow", "app81"],
    },
    "y27632": {
        "experiment": "y27632",
        "dataset_dirs": {"03_Y-27632_experiment_10x"},
        "filename_pattern": r"Y27632_(?:0|10|20|50|100)uM_D\d{2}_",
        "entrypoint": "scripts/run_deo_original_managed.py",
        "entrypoint_args": ["--workflow", "app80"],
    },
    "sodium_alginate": {
        "experiment": "sodium_alginate",
        "dataset_dirs": {"02_Sodium_alginate_experiment_10x"},
        "filename_pattern": r"(?:control|alginate_0\.0[25]pct)_D\d{2}_",
        "entrypoint": "scripts/run_deo_original_managed.py",
        "entrypoint_args": ["--workflow", "app65"],
    },
}
CHAT_STORE = StudioChatStore(STUDIO_DIR / "chat")
CODEX_JOBS = CodexJobManager(BASE_DIR, STUDIO_DIR)
REVIEW_STORE = ReviewStore(STUDIO_DIR / "reviews")


def load_workflow_settings():
    try:
        payload = json.loads(WORKFLOW_SETTINGS_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


WORKFLOW_SETTINGS = load_workflow_settings()


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TABLE_EXTS = {".csv", ".tsv", ".xlsx", ".xls"}
ANALYSIS_EXTS = {".h5ad", ".h5", ".hdf5"}
ARCHIVE_EXTS = {".zip", ".tgz", ".tar", ".tar.gz"}
DOC_EXTS = {".pdf", ".docx"}
MAX_ARCHIVE_PREVIEW_BYTES = 50 * 1024 * 1024
MAX_FLUORESCENCE_FILES = 30

DATASET_METADATA = {
    "zenodo_10643410": "zenodo_10643410.md",
}

DATABASE_SCAN_ROOTS = [
    ("OrganoidAgent", BASE_DIR / "analysis-outputs"),
    ("Zhengyu", BASE_DIR.parent / "Zhengyu"),
]


def format_bytes(num):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}PB"


def safe_dataset_path(raw_path):
    """Resolve relative or absolute inputs, but never leave the datasets root."""
    dataset_root = DATA_DIR.resolve()
    raw_text = str(raw_path).strip()
    candidate = Path(raw_text)
    if candidate.is_absolute():
        full = candidate.resolve()
    else:
        relative = raw_text.replace("\\", "/")
        if relative == "datasets":
            relative = ""
        elif relative.startswith("datasets/"):
            relative = relative.removeprefix("datasets/")
        full = (dataset_root / relative).resolve()
    try:
        full.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError("Invalid path: inputs must be inside datasets.") from exc
    return full


def safe_visualization_path(raw_path):
    """Resolve a visualization item within datasets or analysis-outputs."""
    raw_text = str(raw_path).strip().replace("\\", "/")
    if raw_text.startswith("analysis-outputs/"):
        root = (BASE_DIR / "analysis-outputs").resolve()
        full = (BASE_DIR / raw_text).resolve()
        try:
            full.relative_to(root)
        except ValueError as exc:
            raise ValueError("Invalid analysis output path.") from exc
        return full, "analysis-output"
    return safe_dataset_path(raw_text), "dataset"


def morphology_output_options(instruction):
    """Translate explicit output requests without enabling unsolicited figures."""
    text = re.sub(r"\s+", " ", str(instruction or "")).strip().lower()

    def explicitly_requested(keywords, negative_keywords=()):
        if not any(keyword in text for keyword in keywords):
            return False
        negative_leads = ("do not", "don't", "without", "no ", "不要", "不用", "无需", "不生成", "不需要")
        for keyword in (*keywords, *negative_keywords):
            if keyword not in text:
                continue
            position = text.find(keyword)
            prefix = text[max(0, position - 18) : position]
            if any(lead in prefix for lead in negative_leads):
                return False
        return True

    return {
        "overlays": True,
        "report": True,
        "statistics_figures": explicitly_requested(
            ("statistical figure", "statistical chart", "statistics figure", "statistics chart", "metric figure", "summary figure", "line graph", "line chart", "graph", "chart", "plot", "统计图", "统计图片", "折线图", "曲线图", "柱状图")
        ),
        "comparison_gallery": explicitly_requested(("comparison gallery", "comparison figure", "对比图", "比较图", "拼图")),
        "object_crops": explicitly_requested(("object crop", "organoid crop", "cropped object", "裁剪图", "裁切图")),
        "diagnostic_images": explicitly_requested(("diagnostic image", "intermediate image", "instance mask", "signal image", "support mask", "中间图", "诊断图", "实例掩膜", "信号图", "支撑掩膜")),
        "results_csv": explicitly_requested(("csv", "spreadsheet", "metric table", "data table", "数据表", "指标表")),
    }


def fluorescence_runtime_status():
    python_path = workflow_python("viability_windows_python", "ORGANOID_FLUORESCENCE_PYTHON")
    required = {
        "viability": FLUORESCENCE_DIR / "weights" / "viability_best.pth",
        "yolo": FLUORESCENCE_DIR / "weights" / "yolo_organoid_best.pt",
        "sam": FLUORESCENCE_DIR / "weights" / "sam_vit_b_01ec64.pth",
    }
    progress = None
    if FLUORESCENCE_ACTIVE_JOB_ID:
        progress_path = FLUORESCENCE_OUTPUT_DIR / FLUORESCENCE_ACTIVE_JOB_ID / "progress.json"
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            progress = None
    return {
        "ready": python_path is not None and all(path.is_file() for path in required.values()),
        "python": str(python_path) if python_path else None,
        "weights": {name: path.is_file() for name, path in required.items()},
        "busy": FLUORESCENCE_LOCK.locked(),
        "active_job_id": FLUORESCENCE_ACTIVE_JOB_ID,
        "progress": progress,
        "max_files": MAX_FLUORESCENCE_FILES,
    }


def workflow_python(config_key, environment_variable):
    """Resolve a workflow interpreter while preserving configured Windows environments."""
    configured_env = os.environ.get(environment_variable, "").strip()
    configured_file = str(WORKFLOW_SETTINGS.get("environments", {}).get(config_key, "")).strip()
    candidates = [configured_env, configured_file, sys.executable]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    return None


def run_fluorescence_subprocess(command):
    global FLUORESCENCE_ACTIVE_PROCESS
    with FLUORESCENCE_LOCK:
        proc = subprocess.Popen(
            command,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        FLUORESCENCE_ACTIVE_PROCESS = proc
        try:
            stdout, stderr = proc.communicate(timeout=7200)
            return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise subprocess.TimeoutExpired(command, 7200, output=stdout, stderr=stderr)
        finally:
            if FLUORESCENCE_ACTIVE_PROCESS is proc:
                FLUORESCENCE_ACTIVE_PROCESS = None


def run_morphology_subprocess(command, job_id):
    with MORPHOLOGY_LOCK:
        proc = subprocess.Popen(
            command,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        MORPHOLOGY_ACTIVE.update({"job_id": job_id, "process": proc})
        try:
            stdout, stderr = proc.communicate(timeout=12 * 60 * 60)
            return proc.returncode, stdout, stderr
        finally:
            MORPHOLOGY_ACTIVE.update({"job_id": None, "process": None})


def read_runtime_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


async def execute_morphology_job(job_id, command, job_dir):
    job_path = job_dir / "job.json"
    job = read_runtime_json(job_path, {})
    if job.get("status") == "cancelled":
        MORPHOLOGY_ACTIVE.update({"job_id": None, "process": None})
        return
    job.update({"status": "running", "started_at": time.time()})
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        returncode, stdout, stderr = await run_in_thread(run_morphology_subprocess, command, job_id)
        (job_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (job_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        results_path = job_dir / "results.json"
        status = "succeeded" if returncode == 0 and results_path.exists() else "failed"
        current_job = read_runtime_json(job_path, {})
        if current_job.get("status") == "cancelled":
            return
        job.update({"status": status, "returncode": returncode, "finished_at": time.time()})
        if status == "failed":
            job["error"] = "morphology_runner_failed"
    except subprocess.TimeoutExpired:
        job.update({"status": "failed", "error": "timeout", "finished_at": time.time()})
    except Exception as exc:
        job.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_at": time.time()})
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    progress_path = job_dir / "progress.json"
    progress = read_runtime_json(progress_path, {})
    final_status = job.get("status", "failed")
    progress.update({"status": final_status, "percent": 100.0 if final_status == "succeeded" else progress.get("percent", 0.0), "finished_at": job.get("finished_at")})
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if MORPHOLOGY_ACTIVE.get("job_id") == job_id:
        MORPHOLOGY_ACTIVE.update({"job_id": None, "process": None})


def analysis_output_asset_url(raw_path):
    try:
        relative = Path(raw_path).resolve().relative_to((BASE_DIR / "analysis-outputs").resolve())
        return "/analysis-output-files/" + relative.as_posix()
    except (OSError, ValueError, TypeError):
        return None


def morphology_result_payload(job_id, job_dir, results_path=None):
    results_path = results_path or (job_dir / "results.json")
    result = read_runtime_json(results_path)
    if not isinstance(result, dict):
        return None

    for row in result.get("rows", []):
        row["asset_urls"] = {
            key: analysis_output_asset_url(row.get(path_key))
            for key, path_key in {
                "mask": "mask_path",
                "overlay": "overlay_path",
                "instances": "instance_rgb_path",
                "cellpose_mask": "cellpose_mask_path",
                "cellpose_overlay": "cellpose_overlay_path",
                "cellpose_instances": "cellpose_instance_rgb_path",
                "signal": "signal_path",
                "support": "support_path",
            }.items()
        }
    outputs = result.get("outputs", {})
    result["download_urls"] = {
        "csv": analysis_output_asset_url(outputs.get("results_csv")),
        "json": analysis_output_asset_url(outputs.get("results_json")),
        "report": analysis_output_asset_url(outputs.get("report_md")),
        "gallery": analysis_output_asset_url(outputs.get("comparison_gallery")),
        "summary_figure": analysis_output_asset_url(outputs.get("summary_metrics")),
    }
    try:
        result["report_text"] = Path(outputs.get("report_md") or results_path.with_name("report.md")).read_text(encoding="utf-8")
    except OSError:
        result["report_text"] = ""
    result["job_id"] = job_id
    return result


def discover_agent_morphology_artifacts(job_id):
    job_dir = CODEX_JOBS.job_dir(job_id)
    payload = read_runtime_json(job_dir / "input.json", {})
    context = payload.get("extra_context", {}) if isinstance(payload, dict) else {}
    analysis_input = context.get("analysis_input", {}) if isinstance(context, dict) else {}
    selected = analysis_input.get("selected_files", []) if isinstance(analysis_input, dict) else []
    if not selected and isinstance(analysis_input, dict) and analysis_input.get("selected_dataset", {}).get("absolute_path"):
        selected = [{"absolute_path": str(path)} for path in Path(analysis_input["selected_dataset"]["absolute_path"]).rglob("*.tif")]
    selected_paths = {
        str(Path(item.get("absolute_path", "")).resolve()).lower()
        for item in selected
        if isinstance(item, dict) and item.get("absolute_path")
    }
    if not selected_paths:
        return None
    earliest = (job_dir / "input.json").stat().st_mtime - 5
    managed_job_dir = None
    managed_job = None
    managed_root = BASE_DIR / "analysis-outputs" / "morphology_jobs"
    if managed_root.exists():
        for candidate in managed_root.iterdir():
            candidate_job = read_runtime_json(candidate / "job.json") if candidate.is_dir() else None
            if not isinstance(candidate_job, dict) or candidate_job.get("parent_job_id") != job_id:
                continue
            if float(candidate_job.get("created_at", 0)) < earliest:
                continue
            if managed_job is None or float(candidate_job.get("created_at", 0)) > float(managed_job.get("created_at", 0)):
                managed_job_dir = candidate
                managed_job = candidate_job
    managed_progress = read_runtime_json(managed_job_dir / "progress.json", {}) if managed_job_dir else {}
    if managed_job_dir and managed_job and managed_job.get("status") == "succeeded":
        result_payload = morphology_result_payload(managed_job["job_id"], managed_job_dir)
        if result_payload:
            result_payload.update(
                {
                    "managed_job_id": managed_job["job_id"],
                    "status": managed_job["status"],
                    "progress": managed_progress,
                    "completed": int(managed_progress.get("completed", len(result_payload.get("rows", [])))),
                    "total": int(managed_progress.get("total", len(selected_paths))),
                }
            )
    matched_rows = []
    matched_metrics = []
    roots = [BASE_DIR / "analysis-outputs" / "density_growth", BASE_DIR / "analysis-outputs" / "morphology_jobs"]
    for root in roots:
        if not root.exists():
            continue
        for metric_path in root.rglob("*_metrics.json"):
            if metric_path.stat().st_mtime < earliest:
                continue
            row = read_runtime_json(metric_path)
            if not isinstance(row, dict):
                continue
            source = row.get("source_tif") or row.get("source_path")
            if not source or str(Path(source).resolve()).lower() not in selected_paths:
                continue
            row["asset_urls"] = {
                key: analysis_output_asset_url(row.get(path_key))
                for key, path_key in {"mask": "mask_path", "overlay": "overlay_path", "instances": "instance_rgb_path", "cellpose_mask": "cellpose_mask_path", "cellpose_overlay": "cellpose_overlay_path", "cellpose_instances": "cellpose_instance_rgb_path", "signal": "signal_path", "support": "support_path"}.items()
            }
            matched_rows.append(row)
            matched_metrics.append(metric_path)
    result_payload = None
    for metric_path in matched_metrics:
        for parent in metric_path.parents:
            candidate = parent / "results.json"
            if candidate.exists() and candidate.stat().st_mtime >= earliest:
                result_payload = morphology_result_payload(job_id, parent, candidate)
                break
        if result_payload:
            break
    if result_payload:
        result_payload["completed"] = len(result_payload.get("rows", []))
        result_payload["total"] = len(selected_paths)
        return result_payload
    if managed_job is None and not matched_rows:
        return None
    return {
        "job_id": managed_job.get("job_id") if managed_job else job_id,
        "managed_job_id": managed_job.get("job_id") if managed_job else None,
        "status": managed_job.get("status") if managed_job else "pending",
        "progress": managed_progress,
        "workflow": (
            f"{MORPHOLOGY_WORKFLOW_ADAPTERS.get(managed_job.get('workflow_id'), {}).get('experiment', 'density')}_multiscale_cellpose_quality_selection"
            if managed_job
            else "density_multiscale_cellpose_quality_selection"
        ),
        "rows": matched_rows,
        "completed": int(managed_progress.get("completed", len(matched_rows))),
        "total": int(managed_progress.get("total", len(selected_paths))),
        "partial": True,
        "download_urls": {},
        "report_text": "",
    }


def discover_agent_viability_artifacts(job_id):
    """Find a fluorescence result created by an Agent job from selected inputs."""
    job_dir = CODEX_JOBS.job_dir(job_id)
    payload = read_runtime_json(job_dir / "input.json", {})
    context = payload.get("extra_context", {}) if isinstance(payload, dict) else {}
    selected = context.get("analysis_input", {}).get("selected_files", []) if isinstance(context, dict) else []
    selected_paths = {
        str(Path(item.get("absolute_path", "")).resolve()).lower()
        for item in selected
        if isinstance(item, dict) and item.get("absolute_path")
    }
    if not selected_paths:
        return None
    earliest = (job_dir / "input.json").stat().st_mtime - 5
    best = None
    if not FLUORESCENCE_OUTPUT_DIR.exists():
        return None
    for candidate in FLUORESCENCE_OUTPUT_DIR.iterdir():
        if not candidate.is_dir():
            continue
        results_path = candidate / "results.json"
        inputs_path = candidate / "inputs.json"
        if not results_path.exists() or results_path.stat().st_mtime < earliest or not inputs_path.exists():
            continue
        try:
            inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
            input_paths = {str(Path(item).resolve()).lower() for item in inputs if isinstance(item, str)}
        except (OSError, ValueError, TypeError):
            continue
        if input_paths != selected_paths:
            continue
        if best is None or results_path.stat().st_mtime > best[0]:
            best = (results_path.stat().st_mtime, candidate, results_path)
    if best is None:
        return None
    _, output_dir, results_path = best
    result = read_runtime_json(results_path, {})
    if not isinstance(result, dict):
        return None
    result["job_id"] = output_dir.name
    base_url = f"/fluorescence-outputs/{output_dir.name}"
    for sample in result.get("samples", []):
        sample["asset_urls"] = {key: f"{base_url}/{value}" for key, value in sample.get("assets", {}).items()}
    result.update({
        "results_url": f"{base_url}/results.json",
        "csv_url": f"{base_url}/results.csv",
        "report_url": f"{base_url}/report.md",
        "status": result.get("status", "succeeded"),
    })
    return result


def discover_agent_profiler_artifacts(job_id):
    """Find a 24D profiler result created by an Agent job from selected inputs."""
    job_dir = CODEX_JOBS.job_dir(job_id)
    payload = read_runtime_json(job_dir / "input.json", {})
    context = payload.get("extra_context", {}) if isinstance(payload, dict) else {}
    analysis_input = context.get("analysis_input", {}) if isinstance(context, dict) else {}
    selected = analysis_input.get("selected_files", []) if isinstance(analysis_input, dict) else []
    if not selected and isinstance(analysis_input, dict) and analysis_input.get("selected_dataset", {}).get("absolute_path"):
        selected = [{"absolute_path": str(path)} for path in Path(analysis_input["selected_dataset"]["absolute_path"]).rglob("*.tif")]
    selected_paths = {str(Path(item.get("absolute_path", "")).resolve()).lower() for item in selected if isinstance(item, dict) and item.get("absolute_path")}
    if not selected_paths or not PROFILER_OUTPUT_DIR.exists():
        return None
    earliest = (job_dir / "input.json").stat().st_mtime - 5
    best = None
    for candidate in PROFILER_OUTPUT_DIR.iterdir():
        inputs_path, results_path = candidate / "inputs.json", candidate / "results.json"
        if not candidate.is_dir() or not inputs_path.exists() or inputs_path.stat().st_mtime < earliest:
            continue
        try:
            input_paths = {str(Path(item).resolve()).lower() for item in json.loads(inputs_path.read_text(encoding="utf-8"))}
        except (OSError, ValueError, TypeError):
            continue
        if input_paths == selected_paths and (best is None or inputs_path.stat().st_mtime > best[0]):
            best = (inputs_path.stat().st_mtime, candidate, results_path)
    if best is None:
        return None
    _, output_dir, results_path = best
    progress = read_runtime_json(output_dir / "progress.json", {})
    if not results_path.exists():
        return {"job_id": output_dir.name, "status": progress.get("status", "running"), "progress": progress, "rows": [], "total": int(progress.get("total", len(selected_paths))), "completed": int(progress.get("completed", 0))}
    result = read_runtime_json(results_path, {})
    if not isinstance(result, dict):
        return None
    base_url = f"/profiler-outputs/{output_dir.name}"
    for row in result.get("rows", []):
        row["asset_urls"] = {key: f"{base_url}/{Path(row[key]).relative_to(output_dir).as_posix()}" for key in ("overlay", "crop", "mask")}
    result["chart_urls"] = []
    for chart in result.get("charts", []):
        chart = dict(chart)
        chart_path = Path(chart["path"])
        chart["url"] = f"{base_url}/{chart_path.relative_to(output_dir).as_posix()}"
        result["chart_urls"].append(chart)
    result.update({"job_id": output_dir.name, "results_url": f"{base_url}/results.json", "csv_url": f"{base_url}/results.csv", "progress": progress, "completed": int(progress.get("completed", len(result.get("rows", [])))), "total": int(progress.get("total", len(result.get("rows", []))))})
    return result


def profiler_fallback_report(artifacts):
    """Create a factual report when the external Agent dies after the tool finishes."""
    rows = artifacts.get("rows", []) if isinstance(artifacts, dict) else []
    lines = [
        "# 24-Dimension Organoid Profiling Report",
        "",
        "The YOLO + SAM profiler completed successfully. This report was finalized by the local workflow because the outer Agent process did not return after the profiler completed.",
        "",
        f"- Images: {len(rows)}",
        "- Units: pixel space (pixel size was not supplied)",
        "- Evidence: results JSON/CSV, masks, overlays, crops, and generated charts",
        "",
        "## Per-image measured summary",
        "",
        "| Day | Group | Image | Area (px²) | Growth rate | YOLO confidence | SAM score |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {day} | {group} | {image} | {area} | {growth} | {yolo} | {sam} |".format(
                day=row.get("day", "NA"),
                group=row.get("group", "NA"),
                image=str(row.get("image_name", "")).replace("|", "\\|"),
                area=row.get("area", "NA"),
                growth=row.get("growth_rate", "NA"),
                yolo=row.get("yolo_confidence", "NA"),
                sam=row.get("sam_score", "NA"),
            )
        )
    chart_urls = artifacts.get("chart_urls", []) if isinstance(artifacts, dict) else []
    if chart_urls:
        lines.extend(["", "## Generated charts", ""])
        for chart in chart_urls:
            lines.append(f"- [{chart.get('metric', 'chart')}]({chart.get('url', '')})")
    lines.extend([
        "",
        "The complete 24-dimensional values remain in the generated results.csv and results.json. Images with low confidence or questionable masks should be reviewed using the individual overlay and manual-review controls.",
    ])
    return "\n".join(lines) + "\n"


def reconcile_orphaned_profiler_agent(job_id, profiler):
    """Close an Agent job whose profiler child finished but Codex disappeared."""
    if not isinstance(profiler, dict) or profiler.get("progress", {}).get("status") != "succeeded":
        return False
    job = CODEX_JOBS.read_job(job_id)
    if job.get("status") != "running":
        return False
    # Do not interrupt a newly-started Codex process. Once the profiler has
    # completed, however, a missing process after this grace period is an
    # orphaned Agent wrapper rather than an active analysis.
    started = (CODEX_JOBS.job_dir(job_id) / "input.json").stat().st_mtime
    if time.time() - started < 30:
        return False
    with CODEX_JOBS._process_lock:
        process_alive = job_id in CODEX_JOBS._processes
    if process_alive:
        return False
    output = profiler_fallback_report(profiler)
    job_dir = CODEX_JOBS.job_dir(job_id)
    job_dir.joinpath("output.txt").write_text(output, encoding="utf-8")
    task = CODEX_JOBS._tasks.get(job_id)
    if task and not task.done():
        task.cancel()
    CODEX_JOBS.update_job(
        job_id,
        {
            "status": "succeeded",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": round(time.time() - started, 2),
            "returncode": 0,
            "detail": "Profiler completed; local fallback finalized the report after the Agent wrapper stopped responding.",
        },
    )
    return True


def persist_agent_result_summary(payload, report_text):
    """Append an Agent result summary to the workflow's canonical report file."""
    if not isinstance(payload, dict) or not report_text.strip():
        return None
    prompt = str(payload.get("prompt") or "")
    matches = re.findall(r"(analysis-outputs/[A-Za-z0-9_./-]+/results\.json)", prompt)
    for raw_path in matches:
        result_path = (BASE_DIR / raw_path).resolve()
        try:
            result_path.relative_to((BASE_DIR / "analysis-outputs").resolve())
        except ValueError:
            continue
        if not result_path.is_file():
            continue
        report_path = result_path.with_name("report.md")
        marker = "## Agent result interpretation"
        existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
        if marker not in existing:
            section = f"\n\n{marker}\n\n{report_text.strip()}\n"
            report_path.write_text(existing.rstrip() + section, encoding="utf-8")
        return report_path
    return None


def persist_viability_results_table(result, report_path):
    """Write the measured viability table before the asynchronous AI summary."""
    if not isinstance(result, dict) or not report_path:
        return
    report_path = Path(report_path)
    if not report_path.is_file():
        return
    marker = "## Workflow results table"
    existing = report_path.read_text(encoding="utf-8")
    if marker in existing:
        return
    rows = []
    for sample in result.get("samples", []):
        segmentation = sample.get("segmentation") or {}
        rows.append({
            "image": sample.get("filename", ""),
            "viability": float(sample.get("viability", 0.0)),
            "percent": sample.get("viability_percent", "NA"),
            "level": sample.get("level", "Unknown"),
            "regions": 1 if segmentation else 0,
            "status": "Succeeded",
            "failure": "—",
        })
    for failure in result.get("failures", []):
        if isinstance(failure, dict):
            name = failure.get("filename") or failure.get("image") or "unknown"
            reason = failure.get("error") or failure.get("reason") or "analysis failed"
        else:
            name, reason = str(failure), "analysis failed"
        rows.append({"image": name, "viability": -1.0, "percent": "NA", "level": "Failed", "regions": 0, "status": "Failed", "failure": reason})
    rows.sort(key=lambda row: row["viability"], reverse=True)
    lines = [
        marker,
        "",
        "This measured table is generated directly from the completed YOLO/SAM and viability results before the asynchronous Agent interpretation.",
        "",
        "| Image | Predicted viability | Classification | Valid detection regions | Analysis status | Failure reason |",
        "|---|---:|---|---:|---|---|",
    ]
    for row in rows:
        lines.append("| {image} | {percent} | {level} | {regions} | {status} | {failure} |".format(**{key: str(value).replace("|", "\\|") for key, value in row.items() if key != "viability"}))
    report_path.write_text(existing.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
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


def _sqlite_table_counts(path, limit=12):
    counts = []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "select name from sqlite_master where type='table' and name not like 'sqlite_%' order by name"
            ).fetchall()
            for (name,) in rows[:limit]:
                try:
                    count = conn.execute(f'select count(*) from "{name}"').fetchone()[0]
                except Exception:
                    count = None
                counts.append({"table": name, "rows": count})
        finally:
            conn.close()
    except Exception as exc:
        return [], str(exc)
    return counts, None


def _nearby_summary(path):
    candidates = [
        path.parent / "summary.json",
        path.parent.parent / "summary.json",
        path.parent.parent / "manifests" / "summary.json",
        path.parent.parent / "metadata" / "summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            data = read_json_file(candidate)
            if isinstance(data, dict):
                overview = {}
                for key, value in data.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        overview[key] = value
                    elif isinstance(value, dict):
                        overview[f"{key}_keys"] = len(value)
                    elif isinstance(value, list):
                        overview[f"{key}_items"] = len(value)
                    if len(overview) >= 12:
                        break
                return {"path": str(candidate), "overview": overview}
    return None


def read_json_file(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_studio_databases():
    items = []
    for project, root in DATABASE_SCAN_ROOTS:
        if not root.exists():
            continue
        for current_root, dirs, files in os.walk(root):
            dirs[:] = sorted([name for name in dirs if not name.startswith(".")])
            for filename in sorted(files):
                path = Path(current_root) / filename
                if path.suffix.lower() not in {".sqlite", ".db"}:
                    continue
                tables, error = _sqlite_table_counts(path)
                items.append(
                    {
                        "project": project,
                        "name": path.name,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "size_human": format_bytes(path.stat().st_size),
                        "tables": tables,
                        "error": error,
                        "summary": _nearby_summary(path),
                    }
                )
                if len(items) >= 200:
                    return items
            if len(items) >= 200:
                continue
    return items


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


def _output_job_context(path):
    output_root = (BASE_DIR / "analysis-outputs").resolve()
    current = path.parent
    while current != output_root and output_root in current.parents:
        inputs_path = current / "inputs.json"
        job_path = current / "job.json"
        inputs = read_runtime_json(inputs_path, []) if inputs_path.exists() else []
        job = read_runtime_json(job_path, {}) if job_path.exists() else {}
        if isinstance(job, dict) and not inputs:
            inputs = job.get("inputs", [])
        if inputs or job:
            first = str(inputs[0]) if isinstance(inputs, list) and inputs else ""
            dataset = ""
            if first:
                try:
                    dataset = Path(first).resolve().relative_to(DATA_DIR.resolve()).parts[0]
                except (ValueError, IndexError):
                    dataset = ""
            return dataset, current.name
        current = current.parent
    relative = path.resolve().relative_to(output_root)
    return "", relative.parts[0] if relative.parts else "analysis-outputs"


def list_segmentation_outputs(limit=300):
    root = BASE_DIR / "analysis-outputs"
    if not root.exists():
        return []
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        name = path.name.lower()
        if not any(token in name for token in ("mask", "overlay", "segmentation")):
            continue
        candidates.append(path)
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    results = []
    for path in candidates[:limit]:
        dataset, job_id = _output_job_context(path)
        relative = path.relative_to(BASE_DIR).as_posix()
        results.append(
            {
                "name": path.name,
                "path": relative,
                "size_bytes": path.stat().st_size,
                "size_human": format_bytes(path.stat().st_size),
                "kind": "image",
                "ext": path.suffix.lower(),
                "dataset": dataset or "Generated analysis output",
                "group": job_id,
                "source": "analysis-output",
            }
        )
    return results


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
        self.write(index_path.read_text(encoding="utf-8"))


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
        if category == "segmentation":
            self.write(
                {
                    "files": list_segmentation_outputs(),
                    "empty_message": "No generated segmentation masks or overlays were found. Run a segmentation workflow first.",
                }
            )
            return
        results = []
        for entry in list_datasets():
            dataset_path = safe_dataset_path(entry["path"])
            for info in list_files(dataset_path):
                kind = info["kind"]
                info["dataset"] = entry["name"]
                if category == "datasets":
                    results.append(info)
                elif category == "features" and kind in {"table", "gzip"}:
                    results.append(info)
                elif category == "analysis" and kind == "analysis":
                    results.append(info)
        empty_messages = {
            "features": "No feature or manifest tables were found in the datasets.",
            "analysis": "No AnnData/HDF5 files are present. This view is intended for .h5ad, .h5, or .hdf5 downstream-analysis datasets, not CSV morphology results.",
        }
        self.write({"files": results, "empty_message": empty_messages.get(category, "No items found.")})


class PreviewHandler(tornado.web.RequestHandler):
    def get(self):
        rel_path = self.get_argument("path")
        path, source = safe_visualization_path(rel_path)
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
                if source == "analysis-output":
                    output_rel = path.relative_to((BASE_DIR / "analysis-outputs").resolve()).as_posix()
                    payload["preview"] = {"preview_url": f"/analysis-output-files/{output_rel}"}
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


class AgentStateHandler(tornado.web.RequestHandler):
    def get(self):
        active_jobs = CODEX_JOBS.list_active_jobs()
        morphology_job_id = str(MORPHOLOGY_ACTIVE.get("job_id") or "")
        active_morphology_jobs = []
        if morphology_job_id:
            morphology_dir = MORPHOLOGY_OUTPUT_DIR / morphology_job_id
            morphology_job = read_runtime_json(morphology_dir / "job.json", {})
            if morphology_job.get("status") in {"queued", "running"}:
                active_morphology_jobs.append(
                    {
                        **morphology_job,
                        "id": morphology_job_id,
                        "progress": read_runtime_json(morphology_dir / "progress.json", {}),
                    }
                )
        self.write(
            {
                "studio_root": str(STUDIO_DIR),
                "codex_available": bool(shutil.which("codex")),
                "default_model": CODEX_JOBS.default_model,
                "active_jobs": active_jobs,
                "active_morphology_jobs": active_morphology_jobs,
                "active_job_count": len(active_jobs) + len(active_morphology_jobs),
                "recent_jobs": CODEX_JOBS.list_jobs(limit=10),
            }
        )


class AgentDatabasesHandler(tornado.web.RequestHandler):
    def get(self):
        self.write({"databases": list_studio_databases()})


class AgentSessionHandler(tornado.web.RequestHandler):
    def post(self):
        payload = json.loads(self.request.body.decode("utf-8") or "{}")
        session = CHAT_STORE.new_session(str(payload.get("title") or "OrganoidAgent chat"))
        self.write({"session": session, "messages": []})

    def get(self):
        session_id = self.get_argument("id", "")
        if not session_id:
            self.set_status(400)
            self.write({"error": "session id is required"})
            return
        try:
            self.write({"session": CHAT_STORE.get_session(session_id), "messages": CHAT_STORE.list_messages(session_id)})
        except FileNotFoundError:
            self.set_status(404)
            self.write({"error": "session not found"})


class AgentPipelineParseHandler(tornado.web.RequestHandler):
    def post(self):
        payload = json.loads(self.request.body.decode("utf-8") or "{}")
        try:
            ir = parse_aaps(str(payload.get("text") or ""))
            self.write({"ok": True, "ir": ir})
        except ParseError as exc:
            self.set_status(400)
            self.write(exc.to_dict())


class AgentCodexJobHandler(tornado.web.RequestHandler):
    def post(self):
        payload = json.loads(self.request.body.decode("utf-8") or "{}")
        try:
            self.write(CODEX_JOBS.submit_job(payload))
        except CodexJobError as exc:
            self.set_status(400)
            self.write({"error": exc.code, "detail": exc.detail})

    def get(self):
        job_id = self.get_argument("id", "")
        if not job_id:
            self.set_status(400)
            self.write({"error": "job id is required"})
            return
        try:
            response = CODEX_JOBS.job_status(job_id, include_logs=True, include_output=True)
            morphology_artifacts = discover_agent_morphology_artifacts(job_id)
            viability_artifacts = discover_agent_viability_artifacts(job_id)
            profiler_artifacts = discover_agent_profiler_artifacts(job_id)
            if reconcile_orphaned_profiler_agent(job_id, profiler_artifacts):
                response = CODEX_JOBS.job_status(job_id, include_logs=True, include_output=True)
            if response.get("job", {}).get("status") == "succeeded" and morphology_artifacts is None:
                job_payload = read_runtime_json(CODEX_JOBS.job_dir(job_id) / "input.json", {})
                execution_policy = job_payload.get("extra_context", {}).get("execution_policy", {})
                workflow_policy = execution_policy.get("workflow_policy", {})
                job_purpose = job_payload.get("extra_context", {}).get("job_purpose", "analysis_run")
                if workflow_policy.get("execution_mode") == "managed_backend" and job_purpose != "result_summary":
                    CODEX_JOBS.update_job(
                        job_id,
                        {
                            "status": "failed",
                            "error": "managed_submission_failed",
                            "detail": "Codex finished without creating the required managed morphology job.",
                        },
                    )
                    response = CODEX_JOBS.job_status(job_id, include_logs=True, include_output=True)
            response["morphology_artifacts"] = morphology_artifacts
            response["viability_artifacts"] = viability_artifacts
            response["profiler_artifacts"] = profiler_artifacts
            if response.get("job", {}).get("status") in {"succeeded", "failed", "cancelled"}:
                report_text = str(response.get("output_text") or "").strip()
                if report_text:
                    job_payload = read_runtime_json(CODEX_JOBS.job_dir(job_id) / "input.json", {})
                    if isinstance(job_payload, dict) and job_payload.get("extra_context", {}).get("job_purpose") == "result_summary":
                        persist_agent_result_summary(job_payload, report_text)
                    AGENT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
                    report_path = AGENT_REPORT_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]', '', job_id)}.md"
                    report_path.write_text(report_text + "\n", encoding="utf-8")
                    response["report_url"] = "/analysis-output-files/organoid_agent_studio/reports/" + report_path.name
            self.write(response)
        except FileNotFoundError:
            self.set_status(404)
            self.write({"error": "job not found"})


class AgentCodexResultHandler(tornado.web.RequestHandler):
    def get(self):
        job_id = self.get_argument("id", "")
        if not job_id:
            self.set_status(400)
            self.write({"error": "job id is required"})
            return
        try:
            self.write(CODEX_JOBS.job_status(job_id, include_logs=True, include_output=True))
        except FileNotFoundError:
            self.set_status(404)
            self.write({"error": "job not found"})


class AgentCodexCancelHandler(tornado.web.RequestHandler):
    def post(self):
        payload = json.loads(self.request.body.decode("utf-8") or "{}")
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            self.set_status(400)
            self.write({"error": "job_id is required"})
            return
        try:
            self.write(CODEX_JOBS.cancel_job(job_id))
        except FileNotFoundError:
            self.set_status(404)
            self.write({"error": "job not found"})
        except ValueError as exc:
            self.set_status(400)
            self.write({"error": str(exc)})


class AgentChatHandler(tornado.web.RequestHandler):
    def post(self):
        payload = json.loads(self.request.body.decode("utf-8") or "{}")
        session_id = str(payload.get("session_id") or "")
        if not session_id:
            session = CHAT_STORE.new_session("OrganoidAgent chat")
            session_id = session["id"]
        else:
            try:
                session = CHAT_STORE.get_session(session_id)
            except FileNotFoundError:
                session = CHAT_STORE.new_session("OrganoidAgent chat")
                session_id = session["id"]

        message = str(payload.get("message") or "").strip()
        if not message:
            self.set_status(400)
            self.write({"error": "message is required"})
            return
        selected_dataset = str(payload.get("selected_dataset") or "").strip()
        selected_files = payload.get("selected_files") or []
        if not isinstance(selected_files, list):
            self.set_status(400)
            self.write({"error": "selected_files must be a list"})
            return
        dataset_context = {}
        try:
            if selected_dataset:
                dataset_path = safe_dataset_path(selected_dataset)
                if not dataset_path.is_dir():
                    raise ValueError("Selected dataset was not found")
                dataset_context = {
                    "selected_dataset": {
                        "relative_path": selected_dataset,
                        "absolute_path": str(dataset_path),
                    },
                    "selected_files": [],
                    "selection_rule": "Use only the selected files when present; otherwise use the selected dataset directory.",
                }
                for raw_path in selected_files:
                    rel_path = str(raw_path).strip().replace("\\", "/")
                    file_path = safe_dataset_path(rel_path)
                    if not file_path.is_file() or dataset_path not in file_path.parents:
                        raise ValueError(f"Selected file is outside the selected dataset: {rel_path}")
                    dataset_context["selected_files"].append(
                        {"relative_path": rel_path, "absolute_path": str(file_path)}
                    )
            elif selected_files:
                raise ValueError("Select a dataset before selecting files")
        except ValueError as exc:
            self.set_status(400)
            self.write({"error": "invalid_dataset_selection", "detail": str(exc)})
            return
        workflow_id = str(payload.get("workflow_id") or "organoid_analysis").strip()
        workflows = WORKFLOW_SETTINGS.get("workflows", {})
        workflow_policy = workflows.get(workflow_id, workflows.get("organoid_analysis", {}))
        execution_policy = {
            "workflow_id": workflow_id,
            "global_policy": WORKFLOW_SETTINGS.get("global_policy", {}),
            "workflow_policy": workflow_policy,
            "configured_environments": WORKFLOW_SETTINGS.get("environments", {}),
        }
        user_message = CHAT_STORE.append_message(session_id, "user", message)
        transcript = CHAT_STORE.list_messages(session_id)[-12:]
        tool = str(payload.get("tool") or "response")
        job_payload = {
            "tool": tool,
            "prompt": message,
            "session_id": session_id,
            "transcript": transcript,
            "pipeline_text": str(payload.get("pipeline_text") or ""),
            "allow_edits": bool(payload.get("allow_edits", tool == "assistant")),
            "model": payload.get("model"),
            "reasoning": payload.get("reasoning"),
            "extra_context": {
                "analysis_input": dataset_context,
                "execution_policy": execution_policy,
                "job_purpose": str(payload.get("job_purpose") or "analysis_run"),
            },
        }
        try:
            job_status = CODEX_JOBS.submit_job(job_payload)
        except CodexJobError as exc:
            self.set_status(400)
            self.write({"error": exc.code, "detail": exc.detail})
            return
        assistant_message = CHAT_STORE.append_message(
            session_id,
            "assistant",
            f"Started Codex {tool} job {job_status['job']['id']}.",
            job_id=job_status["job"]["id"],
            status="queued",
        )
        self.write(
            {
                "session": session,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "job": job_status["job"],
            }
        )


class FluorescenceStatusHandler(tornado.web.RequestHandler):
    def get(self):
        self.write(fluorescence_runtime_status())


class FluorescenceCancelHandler(tornado.web.RequestHandler):
    def post(self):
        global FLUORESCENCE_ACTIVE_PROCESS, FLUORESCENCE_ACTIVE_JOB_ID
        payload = json.loads(self.request.body.decode("utf-8") or "{}")
        requested_id = str(payload.get("job_id") or "").strip()
        if not FLUORESCENCE_ACTIVE_JOB_ID or (requested_id and requested_id != FLUORESCENCE_ACTIVE_JOB_ID):
            self.set_status(404)
            self.write({"error": "No matching active fluorescence job was found."})
            return
        proc = FLUORESCENCE_ACTIVE_PROCESS
        if proc is not None and proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
            else:
                proc.kill()
        job_id = FLUORESCENCE_ACTIVE_JOB_ID
        progress_path = FLUORESCENCE_OUTPUT_DIR / job_id / "progress.json"
        progress = read_runtime_json(progress_path, {})
        progress.update({"status": "cancelled", "cancelled": True})
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        FLUORESCENCE_ACTIVE_JOB_ID = None
        self.write({"job_id": job_id, "status": "cancelled"})


class FluorescenceImagesHandler(tornado.web.RequestHandler):
    def get(self):
        dataset = self.get_argument("dataset", "").strip()
        if not dataset:
            self.set_status(400)
            self.write({"error": "dataset is required"})
            return
        try:
            root = safe_dataset_path(dataset)
        except ValueError as exc:
            self.set_status(400)
            self.write({"error": str(exc)})
            return
        if not root.is_dir():
            self.set_status(404)
            self.write({"error": "dataset not found"})
            return
        images = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                stat = path.stat()
                images.append(
                    {
                        "name": path.name,
                        "path": str(path.relative_to(DATA_DIR)).replace("\\", "/"),
                        "size_bytes": stat.st_size,
                        "size_human": format_bytes(stat.st_size),
                    }
                )
        self.write({"dataset": dataset, "images": images})


class ReviewHandler(tornado.web.RequestHandler):
    def get(self):
        job_id = str(self.get_argument("job_id", "")).strip()
        target_key = str(self.get_argument("target_key", "")).strip()
        self.write({"reviews": REVIEW_STORE.list(job_id, target_key)})

    def post(self):
        try:
            payload = json.loads(self.request.body.decode("utf-8") or "{}")
            self.write(REVIEW_STORE.upsert(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            self.set_status(400)
            self.write({"error": str(exc)})


class FluorescenceRunHandler(tornado.web.RequestHandler):
    async def post(self):
        global FLUORESCENCE_ACTIVE_JOB_ID
        try:
            payload = json.loads(self.request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self.set_status(400)
            self.write({"error": "invalid JSON body"})
            return
        raw_paths = payload.get("paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            self.set_status(400)
            self.write({"error": "select at least one dataset image"})
            return
        if len(raw_paths) > MAX_FLUORESCENCE_FILES:
            self.set_status(400)
            self.write({"error": f"a batch may contain at most {MAX_FLUORESCENCE_FILES} images"})
            return
        try:
            paths = []
            for raw_path in raw_paths:
                path = safe_dataset_path(str(raw_path))
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                    raise ValueError(f"unsupported dataset image: {raw_path}")
                paths.append(path)
        except ValueError as exc:
            self.set_status(400)
            self.write({"error": str(exc)})
            return
        status = fluorescence_runtime_status()
        if not status["ready"]:
            self.set_status(503)
            self.write({"error": "fluorescence runtime or weights are not ready", "status": status})
            return
        if FLUORESCENCE_LOCK.locked():
            self.set_status(409)
            self.write({"error": "another fluorescence prediction job is using the GPU"})
            return
        order = "asc" if str(payload.get("order", "desc")).lower() == "asc" else "desc"
        instruction = re.sub(r"\s+", " ", str(payload.get("instruction") or "Detect organoid viability and explain the image evidence.")).strip()[:500]
        language = str(payload.get("language") or "auto").lower()
        if language not in {"auto", "en", "zh"}:
            self.set_status(400)
            self.write({"error": "language must be auto, en, or zh"})
            return
        job_id = "fluorescence-" + time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        output_dir = FLUORESCENCE_OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        inputs_path = output_dir / "inputs.json"
        inputs_path.write_text(json.dumps([str(path) for path in paths], ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "status": "starting",
                    "total": len(paths),
                    "completed": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "current_index": 0,
                    "current_file": paths[0].name if paths else None,
                    "percent": 0.0,
                    "elapsed_seconds": 0.0,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        viability_python = workflow_python("viability_windows_python", "ORGANOID_FLUORESCENCE_PYTHON")
        if viability_python is None:
            self.set_status(500)
            self.write({"error": "No usable viability Python was found. Set ORGANOID_FLUORESCENCE_PYTHON or configure viability_windows_python."})
            return
        command = [
            str(viability_python),
            str(BASE_DIR / "fluorescence_prediction" / "run.py"),
            "--inputs-json",
            str(inputs_path),
            "--output-dir",
            str(output_dir),
            "--order",
            order,
            "--instruction",
            instruction,
            "--language",
            language,
            "--max-files",
            str(MAX_FLUORESCENCE_FILES),
        ]
        FLUORESCENCE_ACTIVE_JOB_ID = job_id
        try:
            completed = await run_in_thread(run_fluorescence_subprocess, command)
        except subprocess.TimeoutExpired:
            self.set_status(504)
            self.write({"error": "fluorescence prediction exceeded the two-hour timeout", "job_id": job_id})
            return
        finally:
            FLUORESCENCE_ACTIVE_JOB_ID = None
        result_path = output_dir / "results.json"
        if not result_path.exists():
            self.set_status(500)
            self.write(
                {
                    "error": "fluorescence workflow did not create results.json",
                    "job_id": job_id,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                }
            )
            return
        result = json.loads(result_path.read_text(encoding="utf-8"))
        base_url = f"/fluorescence-outputs/{job_id}"
        for sample in result.get("samples", []):
            sample["asset_urls"] = {key: f"{base_url}/{value}" for key, value in sample.get("assets", {}).items()}
        result.update(
            {
                "job_id": job_id,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
                "results_url": f"{base_url}/results.json",
                "csv_url": f"{base_url}/results.csv",
                "report_url": f"{base_url}/report.md",
            }
        )
        persist_viability_results_table(result, output_dir / "report.md")
        self.set_status(200 if result.get("samples") else 500)
        self.write(result)


class OrganoidProfilerRunHandler(tornado.web.RequestHandler):
    async def post(self):
        payload = json.loads(self.request.body.decode("utf-8") or "{}")
        raw_paths = payload.get("paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            self.set_status(400)
            self.write({"error": "Select one or more profiler TIFF files."})
            return
        if PROFILER_LOCK.locked():
            self.set_status(409)
            self.write({"error": "Another organoid profiler job is already running."})
            return
        try:
            paths = []
            for item in raw_paths:
                path = safe_dataset_path(str(item))
                if path.is_dir():
                    paths.extend(sorted(path.rglob("*.tif")))
                else:
                    paths.append(path)
            paths = list(dict.fromkeys(paths))
            if any(not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"} for path in paths):
                raise ValueError("The organoid profiler accepts TIFF files only.")
            dataset_dirs = {path.resolve().relative_to(DATA_DIR.resolve()).parts[0] for path in paths}
            if dataset_dirs != {"06_OrganoidProfiler_49"}:
                raise ValueError("This workflow accepts only the 06_OrganoidProfiler_49 dataset.")
            pixel_size = payload.get("pixel_size")
            if pixel_size in (None, "", "null"):
                pixel_size = None
            else:
                pixel_size = float(pixel_size)
                if not math.isfinite(pixel_size) or pixel_size <= 0:
                    raise ValueError("pixel_size must be a positive number.")
        except (ValueError, TypeError) as exc:
            self.set_status(400)
            self.write({"error": str(exc)})
            return
        job_id = "profiler-" + time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        output_dir = PROFILER_OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        inputs_path = output_dir / "inputs.json"
        inputs_path.write_text(json.dumps([str(path) for path in paths], ensure_ascii=False, indent=2), encoding="utf-8")
        python_path = workflow_python("viability_windows_python", "ORGANOID_FLUORESCENCE_PYTHON")
        if python_path is None:
            self.set_status(500)
            self.write({"error": "No usable navila Python was found for YOLO+SAM profiler."})
            return
        command = [str(python_path), str(BASE_DIR / "scripts" / "run_organoid_profiler.py"), "--inputs-json", str(inputs_path), "--output-dir", str(output_dir)]
        if pixel_size is not None:
            command.extend(["--pixel-size", str(pixel_size)])
        if bool(payload.get("generate_charts", False)):
            command.append("--generate-charts")
        try:
            completed = await run_in_thread(run_profiler_subprocess, command)
        except subprocess.TimeoutExpired:
            self.set_status(504)
            self.write({"error": "organoid profiler exceeded the timeout", "job_id": job_id})
            return
        result_path = output_dir / "results.json"
        if completed.returncode != 0 or not result_path.exists():
            self.set_status(500)
            self.write({"error": "organoid profiler did not create results.json", "job_id": job_id, "returncode": completed.returncode, "stderr": completed.stderr[-4000:]})
            return
        result = json.loads(result_path.read_text(encoding="utf-8"))
        base_url = f"/profiler-outputs/{job_id}"
        for row in result.get("rows", []):
            row["asset_urls"] = {key: f"{base_url}/{Path(row[key]).relative_to(output_dir).as_posix()}" for key in ("overlay", "crop", "mask")}
        result["chart_urls"] = []
        for chart in result.get("charts", []):
            chart = dict(chart)
            chart_path = Path(chart["path"])
            chart["url"] = f"{base_url}/{chart_path.relative_to(output_dir).as_posix()}"
            result["chart_urls"].append(chart)
        result.update({"job_id": job_id, "results_url": f"{base_url}/results.json", "csv_url": f"{base_url}/results.csv"})
        self.write(result)


class AnalysisChartHandler(tornado.web.RequestHandler):
    async def post(self):
        payload = json.loads(self.request.body.decode("utf-8") or "{}")
        raw_input = str(payload.get("input") or "").strip()
        if not raw_input:
            self.set_status(400)
            self.write({"error": "input must point to a results.json or results.csv under analysis-outputs."})
            return
        try:
            candidate = Path(raw_input)
            input_path = candidate.resolve() if candidate.is_absolute() else (BASE_DIR / candidate).resolve()
            input_path.relative_to((BASE_DIR / "analysis-outputs").resolve())
            if not input_path.is_file() or input_path.suffix.lower() not in {".json", ".csv"}:
                raise ValueError("input must be an existing results.json or results.csv under analysis-outputs")
        except (OSError, ValueError) as exc:
            self.set_status(400)
            self.write({"error": str(exc)})
            return
        output_dir = BASE_DIR / "analysis-outputs" / "charts" / ("chart-" + time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
        python_path = workflow_python("viability_windows_python", "ORGANOID_FLUORESCENCE_PYTHON")
        if python_path is None:
            self.set_status(500)
            self.write({"error": "No configured Python environment is available for chart generation."})
            return
        command = [str(python_path), str(BASE_DIR / "scripts" / "generate_analysis_charts.py"), "--input", str(input_path), "--output-dir", str(output_dir), "--metrics", str(payload.get("metrics") or "area,growth_rate")]
        completed = await run_in_thread(subprocess.run, command, cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        charts_path = output_dir / "charts.json"
        if completed.returncode != 0 or not charts_path.exists():
            self.set_status(500)
            self.write({"error": "chart generation failed", "stderr": completed.stderr[-4000:]})
            return
        charts = json.loads(charts_path.read_text(encoding="utf-8")).get("charts", [])
        base_url = "/analysis-output-files/" + output_dir.relative_to(BASE_DIR / "analysis-outputs").as_posix()
        for chart in charts:
            chart["url"] = f"{base_url}/{Path(chart['filename']).as_posix()}"
        self.write({"status": "succeeded", "charts": charts, "output_dir": str(output_dir)})


def run_profiler_subprocess(command):
    with PROFILER_LOCK:
        return subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)


class MorphologyRunHandler(tornado.web.RequestHandler):
    async def post(self):
        payload = json.loads(self.request.body.decode("utf-8") or "{}")
        workflow_id = str(payload.get("workflow_id") or "zhengyu").strip()
        workflow_adapter = MORPHOLOGY_WORKFLOW_ADAPTERS.get(workflow_id)
        if workflow_adapter is None:
            self.set_status(400)
            self.write({"error": f"Unsupported managed morphology workflow: {workflow_id}."})
            return
        raw_paths = payload.get("paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            self.set_status(400)
            self.write({"error": "Select one or more images before starting morphology analysis."})
            return
        if MORPHOLOGY_LOCK.locked() or MORPHOLOGY_ACTIVE.get("job_id"):
            self.set_status(409)
            self.write({"error": "Another morphology job is already running."})
            return
        try:
            paths = [safe_dataset_path(str(item)) for item in raw_paths]
            allowed_extensions = workflow_adapter.get("extensions", {".tif", ".tiff"})
            if any(not path.is_file() or path.suffix.lower() not in allowed_extensions for path in paths):
                allowed = ", ".join(sorted(allowed_extensions))
                raise ValueError(f"Workflow {workflow_id} accepts only these image types: {allowed}.")
            selected_dataset_dirs = {path.resolve().relative_to(DATA_DIR.resolve()).parts[0] for path in paths}
            unexpected_dirs = selected_dataset_dirs - workflow_adapter["dataset_dirs"]
            if unexpected_dirs:
                expected = ", ".join(sorted(workflow_adapter["dataset_dirs"]))
                actual = ", ".join(sorted(unexpected_dirs))
                raise ValueError(
                    f"Workflow {workflow_id} expects dataset {expected}; selected files came from {actual}."
                )
            invalid_names = [
                path.name
                for path in paths
                if re.search(workflow_adapter["filename_pattern"], path.name, flags=re.IGNORECASE) is None
            ]
            if invalid_names:
                raise ValueError(
                    f"Workflow {workflow_id} cannot parse condition/day metadata from: {', '.join(invalid_names[:3])}."
                )
        except ValueError as exc:
            self.set_status(400)
            self.write({"error": str(exc)})
            return
        magnifications = set()
        for path in paths:
            match = re.search(r"__(\d+)x", path.name, flags=re.IGNORECASE)
            if match:
                magnifications.add(f"{match.group(1)}x")
        if len(paths) > 1 and len(magnifications) > 1:
            self.set_status(400)
            self.write({"error": f"Mixed magnifications are not comparable: {', '.join(sorted(magnifications))}. Select images with one magnification or run them separately."})
            return
        job_id = "morphology-" + time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        job_dir = MORPHOLOGY_OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        progress_path = job_dir / "progress.json"
        instruction = re.sub(r"\s+", " ", str(payload.get("instruction") or "Analyze organoid morphology and report segmentation and quantitative evidence.")).strip()[:1000]
        output_options = morphology_output_options(instruction)
        parent_job_id = re.sub(r"[^A-Za-z0-9_.-]", "", str(payload.get("parent_job_id") or ""))
        job = {"job_id": job_id, "status": "queued", "created_at": time.time(), "inputs": [str(path) for path in paths], "magnifications": sorted(magnifications), "output_dir": str(job_dir), "workflow_id": workflow_id, "instruction": instruction, "output_options": output_options, "parent_job_id": parent_job_id or None}
        (job_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        progress_path.write_text(json.dumps({"status": "queued", "total": len(paths), "completed": 0, "percent": 0.0}, indent=2), encoding="utf-8")
        cellpose_python = workflow_python("cellpose_windows_python", "ORGANOID_CELLPOSE_PYTHON")
        if cellpose_python is None:
            self.set_status(503)
            self.write(
                {
                    "error": (
                        "No usable Cellpose Python was found. Set ORGANOID_CELLPOSE_PYTHON "
                        "or install requirements-analysis.txt in the server environment."
                    )
                }
            )
            return
        entrypoint = BASE_DIR / workflow_adapter.get("entrypoint", "scripts/run_density_growth_d12_analysis.py")
        command = [str(cellpose_python), str(entrypoint), *workflow_adapter.get("entrypoint_args", [])]
        command.extend(["--out-dir", str(job_dir), "--inputs", *[str(path) for path in paths], "--progress-file", str(progress_path), "--instruction", instruction])
        for option, flag in {
            "statistics_figures": "--statistics-figures",
            "comparison_gallery": "--comparison-gallery",
            "object_crops": "--object-crops",
            "diagnostic_images": "--diagnostic-images",
            "results_csv": "--results-csv",
        }.items():
            if output_options[option]:
                command.append(flag)
        if bool(payload.get("overwrite")):
            command.append("--overwrite")
        MORPHOLOGY_ACTIVE.update({"job_id": job_id, "process": None})
        task = asyncio.create_task(execute_morphology_job(job_id, command, job_dir))
        MORPHOLOGY_TASKS.add(task)
        task.add_done_callback(MORPHOLOGY_TASKS.discard)
        self.set_status(202)
        self.write({"job_id": job_id, "status": "queued", "poll_url": f"/api/morphology/status?id={job_id}", "output_dir": str(job_dir)})


class MorphologyStatusHandler(tornado.web.RequestHandler):
    def get(self):
        job_id = re.sub(r"[^A-Za-z0-9_.-]", "", self.get_argument("id", ""))
        if not job_id:
            job_id = str(MORPHOLOGY_ACTIVE.get("job_id") or "")
        job_dir = MORPHOLOGY_OUTPUT_DIR / job_id
        job = read_runtime_json(job_dir / "job.json")
        if not job:
            self.set_status(404)
            self.write({"error": "Morphology job not found."})
            return
        self.write({"job": job, "progress": read_runtime_json(job_dir / "progress.json", {}), "stdout_tail": tail_text(job_dir / "stdout.log"), "stderr_tail": tail_text(job_dir / "stderr.log"), "result": morphology_result_payload(job_id, job_dir) if job.get("status") == "succeeded" else None})


class MorphologyCancelHandler(tornado.web.RequestHandler):
    def post(self):
        payload = json.loads(self.request.body.decode("utf-8") or "{}")
        job_id = re.sub(r"[^A-Za-z0-9_.-]", "", str(payload.get("job_id") or ""))
        proc = MORPHOLOGY_ACTIVE.get("process") if MORPHOLOGY_ACTIVE.get("job_id") == job_id else None
        if proc and proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
            else:
                proc.kill()
        job_dir = MORPHOLOGY_OUTPUT_DIR / job_id
        job = read_runtime_json(job_dir / "job.json", {})
        job.update({"status": "cancelled", "finished_at": time.time(), "error": "cancelled_by_user"})
        (job_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        self.write({"job": job})


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
            (r"/api/agent/state", AgentStateHandler),
            (r"/api/agent/databases", AgentDatabasesHandler),
            (r"/api/agent/session", AgentSessionHandler),
            (r"/api/agent/chat", AgentChatHandler),
            (r"/api/reviews", ReviewHandler),
            (r"/api/agent/pipeline/parse", AgentPipelineParseHandler),
            (r"/api/agent/codex/job", AgentCodexJobHandler),
            (r"/api/agent/codex/result", AgentCodexResultHandler),
            (r"/api/agent/codex/cancel", AgentCodexCancelHandler),
            (r"/api/fluorescence/status", FluorescenceStatusHandler),
            (r"/api/fluorescence/cancel", FluorescenceCancelHandler),
            (r"/api/fluorescence/images", FluorescenceImagesHandler),
            (r"/api/fluorescence/run", FluorescenceRunHandler),
            (r"/api/organoid-profiler/run", OrganoidProfilerRunHandler),
            (r"/api/charts/generate", AnalysisChartHandler),
            (r"/api/morphology/run", MorphologyRunHandler),
            (r"/api/morphology/status", MorphologyStatusHandler),
            (r"/api/morphology/cancel", MorphologyCancelHandler),
            (r"/morphology-outputs/(.*)", tornado.web.StaticFileHandler, {"path": str(MORPHOLOGY_OUTPUT_DIR)}),
            (r"/analysis-output-files/(.*)", tornado.web.StaticFileHandler, {"path": str(BASE_DIR / "analysis-outputs")}),
            (r"/files/(.*)", tornado.web.StaticFileHandler, {"path": str(DATA_DIR)}),
            (r"/previews/(.*)", tornado.web.StaticFileHandler, {"path": str(PREVIEW_DIR)}),
            (r"/fluorescence-outputs/(.*)", tornado.web.StaticFileHandler, {"path": str(FLUORESCENCE_OUTPUT_DIR)}),
            (r"/profiler-outputs/(.*)", tornado.web.StaticFileHandler, {"path": str(PROFILER_OUTPUT_DIR)}),
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
