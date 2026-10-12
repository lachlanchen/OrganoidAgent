#!/usr/bin/env python3
"""Run the CellposeSAM reference settings on selected OrganoidBasic JPEGs."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from cellpose import models


INPUTS = [
    Path("/media/yinghan/My Passport/review/OrganoidBasic_v20211206/train/0002_1141294_ckm3hgcub0fyg345zcgc2fous_img.jpg"),
    Path("/media/yinghan/My Passport/review/OrganoidBasic_v20211206/train/0002_1400272_cknohk6hx30723q67t8a4g2a4_img.jpg"),
    Path("/media/yinghan/My Passport/review/OrganoidBasic_v20211206/train/0002_1534741_ckm3jcpdj447m345zx9i6qear_img.jpg"),
    Path("/media/yinghan/My Passport/review/OrganoidBasic_v20211206/train/0002_1599524_cknom79kl3xsn3q677utjngdz_img.jpg"),
]
OUTPUT = Path("/media/yinghan/My Passport/review/cellpose_result_organoidbasic_reference")


def overlay(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = image.copy()
    rng = np.random.default_rng(42)
    colors = rng.integers(40, 240, size=(int(labels.max()) + 1, 3), dtype=np.uint8)
    colors[0] = 0
    color_mask = colors[labels]
    result = (result.astype(np.float32) * 0.62 + color_mask.astype(np.float32) * 0.38).astype(np.uint8)
    for label in range(1, int(labels.max()) + 1):
        component = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 255, 255), 1)
    return result


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = models.CellposeModel(gpu=False, pretrained_model="cpsam")
    settings = {
        "model": "CellposeSAM/cpsam",
        "cellpose_version": "4.1.0 (current navila environment)",
        "batch_size": 1,
        "min_size": 200,
        "cellprob_threshold": 0.0,
        "flow_threshold": 0.4,
        "normalize": True,
        "do_3D": False,
        "exclude_edge_masks": False,
        "augment": False,
        "diameter": None,
        "gpu": False,
    }
    all_results = {"settings": settings, "images": []}
    for index, path in enumerate(INPUTS, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        started = time.time()
        print(f"[{index}/{len(INPUTS)}] {path.name}", flush=True)
        labels, *_ = model.eval(
            gray,
            batch_size=1,
            channels=[0, 0],
            normalize=True,
            diameter=None,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            min_size=200,
            augment=False,
            do_3D=False,
        )
        labels = labels.astype(np.uint16)
        stem = path.stem
        mask_path = OUTPUT / f"{stem}_cp_masks.png"
        overlay_path = OUTPUT / f"{stem}_overlay.png"
        outlines_path = OUTPUT / f"{stem}_outlines.png"
        cv2.imwrite(str(mask_path), labels)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay(rgb, labels), cv2.COLOR_RGB2BGR))
        outlines = np.zeros(labels.shape, dtype=np.uint8)
        for label in range(1, int(labels.max()) + 1):
            component = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(outlines, contours, -1, 255, 1)
        cv2.imwrite(str(outlines_path), outlines)
        areas = [int((labels == label).sum()) for label in range(1, int(labels.max()) + 1)]
        row = {
            "image": str(path),
            "instances": len(areas),
            "total_segmented_pixels": int(sum(areas)),
            "min_area_px": min(areas) if areas else 0,
            "median_area_px": float(np.median(areas)) if areas else 0,
            "max_area_px": max(areas) if areas else 0,
            "seconds": round(time.time() - started, 1),
            "mask": str(mask_path),
            "overlay": str(overlay_path),
            "outlines": str(outlines_path),
        }
        all_results["images"].append(row)
        print(json.dumps(row), flush=True)
    (OUTPUT / "results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
