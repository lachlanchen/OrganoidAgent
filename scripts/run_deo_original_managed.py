#!/usr/bin/env python3
"""Managed selected-file adapter for the original App81/App65/App80 pipelines."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
APP80 = ROOT / "analysis-tools" / "app80_first_replicate_multiscale_cellpose"
APP81 = ROOT / "analysis-tools" / "app81_density_multiscale_cellpose"
APP65 = ROOT / "analysis-tools" / "app65_alginate_multiscale_cellpose"
for directory in (APP80, APP81, APP65):
    sys.path.insert(0, str(directory))

import run_multiscale_dateaware_cellpose as seg  # noqa: E402
import run_app80_10uM_all_replicates_large_recovery as app80_base  # noqa: E402
import run_app81_main_density_large_recovery as app81  # noqa: E402
import run_app65_all_conditions_large_recovery as app65  # noqa: E402
import backfill_app81_full_metrics as full_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", required=True, choices=("app81", "app65", "app80"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--progress-file")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--statistics-figures", action="store_true")
    parser.add_argument("--comparison-gallery", action="store_true")
    parser.add_argument("--object-crops", action="store_true")
    parser.add_argument("--diagnostic-images", action="store_true")
    parser.add_argument("--results-csv", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def day_of(path: Path) -> int:
    match = re.search(r"_D(\d{2})_", path.name, re.I)
    if not match:
        raise ValueError(f"Cannot parse relative day from {path.name}")
    return int(match.group(1))


def condition_of(path: Path, workflow: str) -> str:
    name = path.name.lower()
    if workflow == "app81":
        for condition in ("low", "middle", "high"):
            if name.startswith(condition + "_"):
                return condition
    elif workflow == "app65":
        if name.startswith("control_"):
            return "No"
        match = re.search(r"alginate_(0\.0[25])pct_", name)
        if match:
            return f"{match.group(1)}% Alginate"
    else:
        match = re.search(r"y27632_(\d+)um_", name)
        if match:
            value = int(match.group(1))
            return "No" if value == 0 else f"{value}uM"
    raise ValueError(f"Cannot parse condition from {path.name} for {workflow}")


def app80_config(day: int) -> dict:
    if day <= 0:
        return {"stage": "early_cluster", "diameters": [90, 180, 320]}
    if day <= 2:
        return {"stage": "cystic_early", "diameters": [180, 320, 480]}
    if day == 3:
        return {"stage": "cystic_mid", "diameters": [220, 380, 560]}
    if day == 4:
        return {"stage": "cystic_mid", "diameters": [200, 340, 520]}
    if day == 5:
        return {"stage": "fused_large", "diameters": [380, 700, 980]}
    if day == 6:
        return {"stage": "fused_large", "diameters": [420, 780, 1100]}
    return {"stage": "differentiated_irregular", "diameters": [320, 600, 900]}


class ProgressModel:
    def __init__(self, model, callback):
        self.model = model
        self.callback = callback
        self.index = 0

    def eval(self, image, *args, **kwargs):
        self.index += 1
        self.callback(self.index, kwargs.get("diameter"))
        return self.model.eval(image, *args, **kwargs)


def enrich(metric: dict, source: Path, condition: str, day: int, workflow: str) -> dict:
    mask_path = Path(metric["mask_path"])
    signal_path = Path(metric["signal_path"])
    labels = full_metrics.load_labels(mask_path)
    signal = full_metrics.load_gray_png(signal_path)
    source_gray = full_metrics.load_source_gray(source)
    areas = np.bincount(labels.ravel())[1:].astype(np.float64)
    areas = areas[areas > 0]
    total_area = float(areas.sum()) if areas.size else 0.0
    stats = json.loads(Path(metric["stats_path"]).read_text(encoding="utf-8"))
    metric.update(
        {
            "condition": condition,
            "experiment": {"app81": "density", "app65": "sodium_alginate", "app80": "y27632"}[workflow],
            "day": day,
            "date_key": f"D{day:02d}",
            "date_label": f"D{day:02d}",
            "relative_day": day,
            "source_tif": str(source),
            "diameters_px": stats.get("diameters_px", []),
            "segmentation_method": f"{workflow}_original_multiscale_with_signal_recovery",
            "segmentation_selection_reason": "original_pipeline_candidate_merge",
            "segmentation_quality_score": None,
            "cellpose_quality_score": None,
            "fallback_quality_score": None,
            "cellpose_quality_passed": None,
            "cellpose_count": None,
            "largest_object_area_px": float(areas.max()) if areas.size else 0.0,
            "mean_object_area_px": float(areas.mean()) if areas.size else 0.0,
            "median_object_area_px": float(np.median(areas)) if areas.size else 0.0,
            "mean_equivalent_diameter_px": float(np.mean(np.sqrt(4 * areas / math.pi))) if areas.size else 0.0,
            "total_equivalent_diameter_px": math.sqrt(4 * total_area / math.pi) if total_area else 0.0,
            "fused_mass_fraction": float(areas.max() / total_area) if total_area else 0.0,
            "working_mask_path": str(mask_path),
            "cellpose_mask_path": None,
            "cellpose_instance_rgb_path": None,
            "cellpose_overlay_path": None,
            "support_path": None,
            "object_metrics_path": None,
        }
    )
    metric.update(full_metrics.compute_signal_metrics(signal))
    metric.update(full_metrics.compute_edge_proxy_metrics(source_gray))
    metric.update(full_metrics.compute_internal_edge_metrics(labels, signal))
    metric.update(full_metrics.compute_center_weighted_metrics(labels, signal))
    metric.update(full_metrics.compute_darkness_metrics(labels, source_gray))
    return metric


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (list, dict))})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_figure(path: Path, rows: list[dict]) -> None:
    labels = [f"{row['condition']} D{row['day']:02d}" for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(rows) * 1.8), 4.5))
    axes[0].bar(labels, [row["count"] for row in rows], color="#3973b8")
    axes[0].set_ylabel("Instance count")
    axes[1].bar(labels, [row["total_area_px"] for row in rows], color="#d64b73")
    axes[1].set_ylabel("Total projected area (px²)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    runs_dir = out_dir / "runs"
    progress_path = Path(args.progress_file).resolve() if args.progress_file else None
    inputs = [Path(item).resolve() for item in args.inputs]
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = seg.models.CellposeModel(gpu=bool(torch.cuda.is_available()))
    rows = []
    for image_index, source in enumerate(inputs, 1):
        condition = condition_of(source, args.workflow)
        day = day_of(source)
        per_out = runs_dir / re.sub(r"[^A-Za-z0-9_.-]+", "_", condition).strip("_") / f"D{day:02d}"
        per_out.mkdir(parents=True, exist_ok=True)
        config = app81.stage_and_diameters(condition, day) if args.workflow == "app81" else (app65.stage_and_diameters(condition, day) if args.workflow == "app65" else app80_config(day))

        def progress(scale_index, diameter):
            percent = 100 * ((image_index - 1) + (scale_index - 1) / max(1, len(config["diameters"]))) / len(inputs)
            atomic_json(progress_path, {"status": "running", "phase": "cellpose", "total": len(inputs), "completed": image_index - 1, "current_index": image_index, "current_file": source.name, "current_condition": condition, "scale_index": scale_index, "scale_total": len(config["diameters"]), "current_scale": diameter, "percent": round(percent, 1), "elapsed_seconds": round(time.perf_counter() - started, 1)})

        wrapped = ProgressModel(model, progress)
        if args.workflow == "app81":
            metric = app81.process_one(wrapped, source, per_out, day, condition, args.overwrite)
        elif args.workflow == "app65":
            metric = app65.process_one(wrapped, source, per_out, day, condition, args.overwrite)
        else:
            staged_dir = out_dir / "staged_inputs" / f"D{day:02d}"
            staged_dir.mkdir(parents=True, exist_ok=True)
            staged = staged_dir / source.name
            if not staged.exists():
                try:
                    os.link(source, staged)
                except OSError:
                    shutil.copy2(source, staged)
            seg.DATE_CONFIG[staged_dir.name] = config
            metric = app80_base.process_one(wrapped, staged, per_out, args.overwrite)
        rows.append(enrich(metric, source, condition, day, args.workflow))
        atomic_json(progress_path, {"status": "running" if image_index < len(inputs) else "finalizing", "phase": "finalizing", "total": len(inputs), "completed": image_index, "current_file": source.name, "percent": round(100 * image_index / len(inputs), 1), "elapsed_seconds": round(time.perf_counter() - started, 1)})

    csv_path = out_dir / "results.csv"
    figure_path = out_dir / "summary_metrics.png"
    write_csv(csv_path, rows)
    if args.statistics_figures:
        make_figure(figure_path, rows)
    workflow_name = {"app81": "App81 original density multiscale pipeline", "app65": "App65 original alginate multiscale pipeline", "app80": "App80 original Y-27632 date-aware pipeline"}[args.workflow]
    report = [f"# {workflow_name}", "", f"Processed {len(rows)} selected image(s).", "", "| Image | Condition | Day | Instances | Total area (px²) | Mean area (px²) |", "|---|---|---:|---:|---:|---:|"]
    report.extend(f"| {row['image_name']} | {row['condition']} | {row['day']} | {row['count']} | {row['total_area_px']:.0f} | {row['mean_object_area_px']:.1f} |" for row in rows)
    report.extend(["", "Segmentation used the original dataset-specific multiscale Cellpose candidate filtering, signal-based large-object recovery, and overlap-aware merge implementation."])
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    result = {
        "analysis": f"{args.workflow}_original_selected_inputs",
        "workflow": workflow_name,
        "experiment": rows[0]["experiment"] if rows else "",
        "inputs": [str(path) for path in inputs],
        "used_gpu": bool(torch.cuda.is_available()),
        "instruction": args.instruction,
        "output_options": {"overlays": True, "report": True, "statistics_figures": args.statistics_figures, "comparison_gallery": False, "object_crops": False, "diagnostic_images": True, "results_csv": True},
        "outputs": {"results_csv": str(csv_path), "results_json": str(out_dir / "results.json"), "report_md": str(out_dir / "report.md"), "comparison_gallery": None, "summary_metrics": str(figure_path) if figure_path.exists() else None, "crops_dir": None, "runs_dir": str(runs_dir)},
        "rows": rows,
        "summary": {"implementation": args.workflow, "interpretation": "Dataset-specific original pipeline output; biological interpretation requires matched conditions and dates."},
    }
    (out_dir / "results.json").write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    atomic_json(progress_path, {"status": "succeeded", "phase": "complete", "total": len(inputs), "completed": len(inputs), "percent": 100.0, "elapsed_seconds": round(time.perf_counter() - started, 1), "results_json": str(out_dir / "results.json")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
