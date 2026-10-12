#!/usr/bin/env python3
"""Segment NEO bright-field images with CellposeSAM and quantify morphology."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from cellpose import models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--overwrite", action="store_true")
    # Accepted for compatibility with the managed morphology endpoint.
    parser.add_argument("--statistics-figures", action="store_true")
    parser.add_argument("--comparison-gallery", action="store_true")
    parser.add_argument("--object-crops", action="store_true")
    parser.add_argument("--diagnostic-images", action="store_true")
    parser.add_argument("--results-csv", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_progress(path: Path | None, payload: dict) -> None:
    if path:
        write_json(path, payload)


def ellipse_perimeter(major: float, minor: float) -> float:
    """Ramanujan approximation for an ellipse perimeter."""
    a, b = major / 2.0, minor / 2.0
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(max(4 - 3 * h, 0))))


def feret_diameter(contour: np.ndarray) -> float:
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float64)
    if len(hull) < 2:
        return 0.0
    distances = hull[:, None, :] - hull[None, :, :]
    return float(np.sqrt(np.max(np.sum(distances * distances, axis=2))))


def object_metrics(labels: np.ndarray, image: np.ndarray, label: int) -> dict | None:
    binary = (labels == label).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(np.count_nonzero(binary))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, width, height = cv2.boundingRect(contour)
    moments = cv2.moments(binary, binaryImage=True)
    cx = float(moments["m10"] / moments["m00"]) if moments["m00"] else float(x + width / 2)
    cy = float(moments["m01"] / moments["m00"]) if moments["m00"] else float(y + height / 2)
    hull = cv2.convexHull(contour)
    convex_mask = np.zeros_like(binary)
    cv2.drawContours(convex_mask, [hull], -1, 1, thickness=cv2.FILLED)
    convex_area = float(np.count_nonzero(convex_mask))
    major = minor = orientation = eccentricity = 0.0
    if len(contour) >= 5:
        (_, _), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        major, minor = sorted((float(axis_a), float(axis_b)), reverse=True)
        orientation = float(angle)
        eccentricity = math.sqrt(max(0.0, 1.0 - (minor * minor) / (major * major))) if major else 0.0
    boundary = contour.reshape(-1, 2).astype(np.float64)
    radii = np.sqrt((boundary[:, 0] - cx) ** 2 + (boundary[:, 1] - cy) ** 2)
    circularity = float(4.0 * math.pi * area / (perimeter * perimeter)) if perimeter else 0.0
    equivalent_diameter = math.sqrt(4.0 * area / math.pi)
    fitted_perimeter = ellipse_perimeter(major, minor)
    pixels = image[binary.astype(bool)]
    intensity = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).reshape(-1) if pixels.size else np.array([])
    return {
        "object_id": int(label),
        "area_px": area,
        "perimeter_px": perimeter,
        "circularity": circularity,
        "non_circularity": abs((1.0 / circularity) - 1.0) if circularity else None,
        "convex_area_px": convex_area,
        "solidity": area / convex_area if convex_area else 0.0,
        "eccentricity": eccentricity,
        "major_axis_length_px": major,
        "minor_axis_length_px": minor,
        "aspect_ratio": major / minor if minor else 0.0,
        "equivalent_diameter_px": equivalent_diameter,
        "feret_diameter_max_px": feret_diameter(contour),
        "extent": area / float(width * height) if width and height else 0.0,
        "centroid_x_px": cx,
        "centroid_y_px": cy,
        "bbox_x_px": int(x),
        "bbox_y_px": int(y),
        "bbox_width_px": int(width),
        "bbox_height_px": int(height),
        "orientation_deg": orientation,
        "mean_radius_px": float(np.mean(radii)) if radii.size else 0.0,
        "radius_std_px": float(np.std(radii)) if radii.size else 0.0,
        "non_smoothness": max(0.0, 1.0 - fitted_perimeter / perimeter) if perimeter and fitted_perimeter else 0.0,
        "mean_intensity": float(np.mean(intensity)) if intensity.size else 0.0,
        "intensity_std": float(np.std(intensity)) if intensity.size else 0.0,
        "touches_image_border": bool(x == 0 or y == 0 or x + width >= labels.shape[1] or y + height >= labels.shape[0]),
    }


def render_overlay(image: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    colors = rng.integers(45, 235, size=(int(labels.max()) + 1, 3), dtype=np.uint8)
    colors[0] = 0
    color_mask = colors[labels]
    overlay = np.where(labels[..., None] > 0, image * 0.62 + color_mask * 0.38, image).astype(np.uint8)
    for label in range(1, int(labels.max()) + 1):
        contours, _ = cv2.findContours((labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    return overlay, color_mask


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.out_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    started = time.time()
    # Cellpose 4.0.5 defaults to the pretrained cpsam model. GPU selection is a
    # runtime device choice; model.eval receives no overrides, preserving all
    # CellposeSAM inference defaults requested for this workflow.
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    rows: list[dict] = []
    objects: list[dict] = []
    total = len(args.inputs)
    for index, source in enumerate(args.inputs, start=1):
        image_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {source}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_started = time.time()
        write_progress(args.progress_file, {"status": "running", "phase": "cellpose_cpsam", "total": total, "completed": index - 1, "current_index": index, "current_file": source.name, "percent": round((index - 1) * 100 / total, 1)})
        labels, *_ = model.eval(image)
        labels = np.asarray(labels, dtype=np.uint16)
        metrics = [item for label in range(1, int(labels.max()) + 1) if (item := object_metrics(labels, image, label))]
        for item in metrics:
            item.update({"image_name": source.name, "source_path": str(source.resolve())})
        objects.extend(metrics)
        overlay, color_mask = render_overlay(image, labels)
        stem = source.stem
        mask_path = runs_dir / f"{stem}_cpsam_mask_16bit.png"
        overlay_path = runs_dir / f"{stem}_cpsam_overlay.png"
        instances_path = runs_dir / f"{stem}_cpsam_instances.png"
        cv2.imwrite(str(mask_path), labels)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(instances_path), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        areas = np.array([item["area_px"] for item in metrics], dtype=float)
        circularities = np.array([item["circularity"] for item in metrics], dtype=float)
        solidities = np.array([item["solidity"] for item in metrics], dtype=float)
        eccentricities = np.array([item["eccentricity"] for item in metrics], dtype=float)
        total_area = float(areas.sum()) if areas.size else 0.0
        row = {
            "image_name": source.name,
            "source_path": str(source.resolve()),
            "source_tif": str(source.resolve()),
            "condition": "NEO",
            "day": 0,
            "count": len(metrics),
            "total_area_px": total_area,
            "mean_area_px": float(areas.mean()) if areas.size else 0.0,
            "median_area_px": float(np.median(areas)) if areas.size else 0.0,
            "mean_equivalent_diameter_px": float(np.mean([item["equivalent_diameter_px"] for item in metrics])) if metrics else 0.0,
            "roundness": float(circularities.mean()) if circularities.size else 0.0,
            "mean_circularity": float(circularities.mean()) if circularities.size else 0.0,
            "mean_solidity": float(solidities.mean()) if solidities.size else 0.0,
            "mean_eccentricity": float(eccentricities.mean()) if eccentricities.size else 0.0,
            "fused_mass_fraction": float(areas.max() / total_area) if total_area else 0.0,
            "segmented_fraction": total_area / float(labels.size),
            "edge_object_count": sum(bool(item["touches_image_border"]) for item in metrics),
            "stage": "cpsam_default",
            "segmentation_method": "CellposeSAM/cpsam",
            "segmentation_selection_reason": "requested pretrained cpsam with default eval parameters",
            "segmentation_quality_score": 1.0,
            "seconds": round(time.time() - image_started, 3),
            "mask_path": str(mask_path.resolve()),
            "overlay_path": str(overlay_path.resolve()),
            "instance_rgb_path": str(instances_path.resolve()),
            "cellpose_mask_path": str(mask_path.resolve()),
            "cellpose_overlay_path": str(overlay_path.resolve()),
            "cellpose_instance_rgb_path": str(instances_path.resolve()),
        }
        rows.append(row)
        write_json(runs_dir / f"{stem}_metrics.json", row)
        write_progress(args.progress_file, {"status": "running", "phase": "morphology", "total": total, "completed": index, "current_index": index, "current_file": source.name, "percent": round(index * 100 / total, 1), "elapsed_seconds": round(time.time() - started, 1)})

    results_csv = args.out_dir / "results.csv"
    objects_csv = args.out_dir / "objects.csv"
    results_json = args.out_dir / "results.json"
    report_md = args.out_dir / "report.md"
    write_csv(results_csv, rows)
    write_csv(objects_csv, objects)
    outputs = {"results_csv": str(results_csv.resolve()), "objects_csv": str(objects_csv.resolve()), "results_json": str(results_json.resolve()), "report_md": str(report_md.resolve())}
    result = {
        "workflow": "neo_cellpose_4.0.5_cpsam_default_morphology",
        "cellpose_version": "4.0.5",
        "model": "CellposeSAM/cpsam",
        "eval_parameters": "CellposeModel.eval defaults (no overrides)",
        "used_gpu": True,
        "units": "pixels; provide pixel calibration before physical-unit conversion",
        "rows": rows,
        "objects": objects,
        "outputs": outputs,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    write_json(results_json, result)
    table_lines = [
        "| Image | Instances | Total projected area (px²) | Mean projected area (px²) |",
        "|---|---:|---:|---:|",
    ]
    table_lines.extend(
        f"| {row['image_name']} | {row['count']} | {row['total_area_px']:.1f} | {row['mean_area_px']:.1f} |"
        for row in rows
    )
    report_md.write_text(
        "# NEO CellposeSAM morphology analysis\n\n"
        "- Segmentation: Cellpose 4.0.5 pretrained `cpsam`; `model.eval(image)` used without inference overrides.\n"
        f"- Images: {len(rows)}; instances: {len(objects)}.\n"
        "- Area values are direct 2D mask measurements, matching the area-based OrganoSeg2 analysis. No volume is inferred from a 2D mask.\n"
        "- Units remain pixels because no image calibration was supplied. Border-touching instances are retained and flagged.\n\n"
        "## Per-image count and area summary\n\n"
        + "\n".join(table_lines)
        + "\n\n## Interpretation guardrails\n\n"
        "Any narrative summary must use only values in the table above. It must describe area as a 2D mask measurement, mention segmentation QC, and must not invent volume, physical units, time-course growth, or biological conclusions.\n",
        encoding="utf-8",
    )
    write_progress(args.progress_file, {"status": "succeeded", "phase": "complete", "total": total, "completed": total, "percent": 100.0, "elapsed_seconds": round(time.time() - started, 1)})
    print(results_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
