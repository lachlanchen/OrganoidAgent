#!/usr/bin/env python3
"""YOLO+SAM segmentation followed by the 24-dimensional organoid profile."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fluorescence_prediction.segmentation import OrganoidSegmentor


FIELDS = [
    "growth_rate", "roundness", "aspect_ratio", "corrected_total_intensity",
    "skewness", "kurtosis", "circularity", "solidity", "corrected_mean_intensity",
    "integrated_intensity", "mean_intensity", "mode_intensity", "median_intensity",
    "intensity_std", "perimeter", "ellipse_minor_axis", "ellipse_major_axis",
    "feret_minimum", "feret_maximum", "equivalent_circle_diameter", "area",
    "centroid_to_center_of_mass", "eccentricity",
]
DAY_RE = re.compile(r"_d(\d+)_", re.IGNORECASE)
GROUP_RE = re.compile(r"_(\d+)\.tif$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pixel-size", type=float, default=None, help="physical length per pixel; omit for pixel units")
    p.add_argument("--max-files", type=int, default=1000)
    p.add_argument("--generate-charts", action="store_true")
    return p.parse_args()


def read_inputs(path: Path, max_files: int) -> list[Path]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError("inputs JSON must contain a list")
    files = []
    for value in values:
        p = Path(str(value)).resolve()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}:
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(x.resolve() for x in p.glob("*.tif")))
    files = list(dict.fromkeys(files))
    if not files:
        raise ValueError("No TIFF inputs found")
    if len(files) > max_files:
        raise ValueError(f"Too many inputs: {len(files)} > {max_files}")
    return files


def metadata(path: Path) -> tuple[str, str]:
    day_match = DAY_RE.search(path.name)
    group_match = GROUP_RE.search(path.name)
    if not day_match or not group_match:
        raise ValueError(f"Cannot parse day/group from {path.name}")
    return day_match.group(1).zfill(2), group_match.group(1).zfill(3)


def unbiased_skew_kurtosis(values: np.ndarray) -> tuple[float, float]:
    n = values.size
    if n < 4:
        return float("nan"), float("nan")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    if sd == 0:
        return 0.0, 0.0
    z = (values - mean) / sd
    skew = n / ((n - 1) * (n - 2)) * float(np.sum(z ** 3))
    kurt = (
        n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * float(np.sum(z ** 4))
        - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    )
    return -skew, kurt


def feret_values(contour: np.ndarray) -> tuple[float, float]:
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
    if len(hull) < 2:
        return 0.0, 0.0
    # The source ImageJ workflow reports boundary Feret values. Hull pairwise
    # distances give the same extrema and are tractable for organoid contours.
    diff = hull[:, None, :] - hull[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    maximum = float(distances.max())
    minimum = float(min(cv2.minAreaRect(hull)[1]))
    return minimum, maximum


def calculate_profile(image: np.ndarray, mask: np.ndarray, pixel_size: float | None) -> dict:
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("empty mask")
    contour = max(contours, key=cv2.contourArea)
    area_px = float(cv2.contourArea(contour))
    perimeter_px = float(cv2.arcLength(contour, True))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    circularity = 4 * math.pi * area_px / (perimeter_px * perimeter_px) if perimeter_px else 0.0
    solidity = area_px / hull_area if hull_area else 0.0
    if len(contour) >= 5:
        (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        major_px, minor_px = max(axis_a, axis_b), min(axis_a, axis_b)
    else:
        major_px = minor_px = 0.0
    aspect = major_px / minor_px if minor_px else 0.0
    roundness = 4 * area_px / (math.pi * major_px * major_px) if major_px else 0.0
    eccentricity = math.sqrt(max(0.0, 1 - (minor_px / major_px) ** 2)) if major_px else 0.0
    feret_min_px, feret_max_px = feret_values(contour)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.astype(np.uint8)
    inverted = 255.0 - gray.astype(np.float32)
    inside = inverted[binary > 0]
    outside = inverted[binary == 0]
    if inside.size == 0:
        raise ValueError("mask contains no pixels")
    background = float(outside.mean()) if outside.size else 0.0
    corrected_total = float(inside.sum() - area_px * background)
    corrected_mean = corrected_total / area_px if area_px else 0.0
    mean = float(inside.mean())
    std = float(inside.std(ddof=1)) if inside.size > 1 else 0.0
    values, counts = np.unique(inside.astype(np.uint8), return_counts=True)
    mode = float(values[np.argmax(counts)])
    skewness, kurtosis = unbiased_skew_kurtosis(inside)
    moments = cv2.moments(binary, binaryImage=True)
    cx = moments["m10"] / moments["m00"] if moments["m00"] else 0.0
    cy = moments["m01"] / moments["m00"] if moments["m00"] else 0.0
    weight = np.maximum(inverted, 0.0) * binary
    total_weight = float(weight.sum())
    wx = float((weight * np.indices(weight.shape)[1]).sum() / total_weight) if total_weight else cx
    wy = float((weight * np.indices(weight.shape)[0]).sum() / total_weight) if total_weight else cy
    center_distance_px = math.hypot(cx - wx, cy - wy)
    scale_length = pixel_size if pixel_size is not None else 1.0
    scale_area = scale_length * scale_length
    return {
        "roundness": roundness, "aspect_ratio": aspect,
        "corrected_total_intensity": corrected_total, "skewness": skewness,
        "kurtosis": kurtosis, "circularity": circularity, "solidity": solidity,
        "corrected_mean_intensity": corrected_mean, "integrated_intensity": float(inside.sum()),
        "mean_intensity": mean, "mode_intensity": mode, "median_intensity": float(np.median(inside)),
        "intensity_std": std, "perimeter": perimeter_px * scale_length,
        "ellipse_minor_axis": minor_px * scale_length, "ellipse_major_axis": major_px * scale_length,
        "feret_minimum": feret_min_px * scale_length, "feret_maximum": feret_max_px * scale_length,
        "equivalent_circle_diameter": 2 * math.sqrt(area_px / math.pi) * scale_length,
        "area": area_px * scale_area, "centroid_to_center_of_mass": center_distance_px * scale_length,
        "eccentricity": eccentricity, "_area_px": area_px,
    }


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = read_inputs(Path(args.inputs_json), args.max_files)
    segmentor = OrganoidSegmentor()
    rows = []
    started = time.time()
    for index, path in enumerate(files, 1):
        day, group = metadata(path)
        image = cv2.cvtColor(cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        segmentation = segmentor.segment(path.read_bytes())
        mask = cv2.imdecode(np.frombuffer(segmentation.mask_png, np.uint8), cv2.IMREAD_GRAYSCALE)
        sample_dir = output / "samples" / f"d{day}_{group}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "overlay.jpg").write_bytes(segmentation.overlay_jpeg)
        (sample_dir / "crop.jpg").write_bytes(segmentation.crop_jpeg)
        (sample_dir / "mask.png").write_bytes(segmentation.mask_png)
        profile = calculate_profile(image, mask, args.pixel_size)
        row = {"day": day, "group": group, "image_name": path.name, "source_path": str(path),
               "pixel_size": args.pixel_size, "yolo_confidence": segmentation.yolo_confidence,
               "sam_score": segmentation.sam_score, "overlay": str(sample_dir / "overlay.jpg"),
               "crop": str(sample_dir / "crop.jpg"), "mask": str(sample_dir / "mask.png"), **profile}
        rows.append(row)
        (output / "progress.json").write_text(json.dumps({"status":"running","completed":index,"total":len(files),"percent":100*index/len(files)}, indent=2), encoding="utf-8")
    day0 = {row["group"]: row["_area_px"] for row in rows if row["day"] == "00"}
    for row in rows:
        baseline = day0.get(row["group"])
        row["growth_rate"] = row["_area_px"] / baseline if baseline else None
        row.pop("_area_px", None)
    result_payload = {"status":"succeeded","pixel_size":args.pixel_size,"units":"physical" if args.pixel_size else "pixel","fields":FIELDS,"rows":rows,"elapsed_seconds":time.time()-started}
    (output / "results.json").write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.generate_charts:
        chart_script = Path(__file__).with_name("generate_analysis_charts.py")
        completed = subprocess.run([sys.executable, str(chart_script), "--input", str(output / "results.json"), "--output-dir", str(output / "charts"), "--metrics", "area,growth_rate"], cwd=ROOT, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"chart generation failed: {completed.stderr[-2000:]}")
        result_payload["charts"] = json.loads((output / "charts" / "charts.json").read_text(encoding="utf-8")).get("charts", [])
        (output / "results.json").write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["day","group","image_name","source_path","pixel_size","yolo_confidence","sam_score",*FIELDS])
        writer.writeheader(); writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in rows)
    (output / "progress.json").write_text(json.dumps({"status":"succeeded","completed":len(files),"total":len(files),"percent":100}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
