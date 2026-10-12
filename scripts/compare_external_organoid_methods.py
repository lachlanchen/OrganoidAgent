"""Run OrgaQuant and compare OrgaQuant/OrganoID/CellposeSAM on three NEO images."""

from __future__ import annotations

import csv
import json
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(r"F:\review\OrganoidAgent-main")
VAL = Path(r"F:\review\OrganoidBasic_v20211206\val")
OUT = ROOT / "analysis-outputs" / "external_method_comparison_3_images"
OQ_ROOT = Path(r"F:\OrgaQuant")
OQ_MODEL = OUT / "orgaquant_intestinal_v3.h5"
OID_OUT = OUT / "organoid_default"
OS2_OUT = OUT / "organoseg2_default"
CP_OUT = ROOT / "analysis-outputs" / "morphology_jobs" / "morphology-20261010T152202-2ac33ac5" / "runs"
STEMS = [
    "20210727_16294_ckunspgrq2th20z824ozpde5m",
    "20210727_18955_ckw3jwvx34z2310aoet4o6hwk",
    "20210727_20499_ckw3l33aa8yqg119087nsbx46",
]


def load_label(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path))
    if a.ndim > 2:
        a = a[..., 0]
    return a.astype(np.int64)


def relabel(a: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape, np.int32)
    for new, old in enumerate(np.unique(a[a > 0]), 1):
        out[a == old] = new
    return out


def mask_metrics(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> dict:
    pred, gt = relabel(pred), relabel(gt)
    npred, ngt = int(pred.max()), int(gt.max())
    pa = np.bincount(pred.ravel(), minlength=npred + 1)[1:]
    ga = np.bincount(gt.ravel(), minlength=ngt + 1)[1:]
    pairs = pred.astype(np.int64) * (ngt + 1) + gt
    overlap = np.bincount(pairs.ravel(), minlength=(npred + 1) * (ngt + 1)).reshape(npred + 1, ngt + 1)[1:, 1:]
    union = pa[:, None] + ga[None, :] - overlap
    ious = np.divide(overlap, union, out=np.zeros_like(overlap, dtype=float), where=union > 0)
    candidates = [(ious[i, j], i, j) for i, j in zip(*np.where(ious >= threshold))]
    used_p, used_g, matched = set(), set(), []
    for iou, i, j in sorted(candidates, reverse=True):
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j); matched.append(float(iou))
    tp, fp, fn = len(matched), npred - len(matched), ngt - len(matched)
    inter = int(np.logical_and(pred > 0, gt > 0).sum())
    fg_union = int(np.logical_or(pred > 0, gt > 0).sum())
    p, r = tp / npred if npred else 0.0, tp / ngt if ngt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    mean_iou = float(np.mean(matched)) if matched else 0.0
    return {"pred_instances": npred, "gt_instances": ngt, "tp": tp, "fp": fp, "fn": fn,
            "instance_precision": p, "instance_recall": r, "instance_f1": f1,
            "mean_matched_iou": mean_iou,
            "panoptic_quality": sum(matched) / (tp + .5 * fp + .5 * fn) if tp + fp + fn else 0.0,
            "pixel_iou": inter / fg_union if fg_union else 1.0,
            "pixel_dice": 2 * inter / (int((pred > 0).sum()) + int((gt > 0).sum()))}


def boxes_from_labels(labels: np.ndarray) -> np.ndarray:
    boxes = []
    for lab in np.unique(labels):
        if lab == 0: continue
        yy, xx = np.where(labels == lab)
        boxes.append([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1])
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def box_metrics(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> dict:
    if len(pred) and len(gt):
        ix1 = np.maximum(pred[:, None, 0], gt[None, :, 0]); iy1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
        ix2 = np.minimum(pred[:, None, 2], gt[None, :, 2]); iy2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        pa = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        ga = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
        ious = inter / (pa[:, None] + ga[None, :] - inter)
    else:
        ious = np.zeros((len(pred), len(gt)))
    cand = [(ious[i, j], i, j) for i, j in zip(*np.where(ious >= threshold))]
    up, ug, matches = set(), set(), []
    for iou, i, j in sorted(cand, reverse=True):
        if i not in up and j not in ug:
            up.add(i); ug.add(j); matches.append(float(iou))
    tp, fp, fn = len(matches), len(pred) - len(matches), len(gt) - len(matches)
    p, r = tp / len(pred) if len(pred) else 0., tp / len(gt) if len(gt) else 0.
    return {"pred_instances": len(pred), "gt_instances": len(gt), "tp": tp, "fp": fp, "fn": fn,
            "instance_precision": p, "instance_recall": r,
            "instance_f1": 2*p*r/(p+r) if p+r else 0., "mean_matched_iou": np.mean(matches) if matches else 0.}


def run_orgaquant():
    sys.path.insert(0, str(OQ_ROOT))
    from keras_retinanet import models
    from keras_retinanet.utils.image import adjust_contrast, preprocess_image, read_image_bgr, resize_image
    model = models.load_model(str(OQ_MODEL), backbone_name="resnet50")
    result = {}
    for stem in STEMS:
        image = read_image_bgr(str(VAL / f"{stem}_img.jpg"))
        processed = preprocess_image(adjust_contrast(image, 1.5))
        processed, scale = resize_image(processed, min_side=1200, max_side=2048)
        boxes, scores, _ = model.predict_on_batch(np.expand_dims(processed, 0))
        keep = scores[0] >= 0.85
        kept = boxes[0][keep] / scale
        result[stem] = {"boxes": kept.tolist(), "scores": scores[0][keep].tolist()}
    (OUT / "orgaquant_predictions.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-orgaquant", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        if args.skip_orgaquant:
            raise RuntimeError("skipped by command line")
        oq = run_orgaquant()
    except Exception as exc:
        oq = None
        (OUT / "orgaquant_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        print(f"OrgaQuant unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
    rows = []
    for stem in STEMS:
        gt = load_label(VAL / f"{stem}_masks_organoid.png")
        oid = load_label(OID_OUT / f"{stem}_img_id-labeled.tif")
        os2 = load_label(OS2_OUT / f"{stem}_organoseg2_labels.tif")
        cp = load_label(CP_OUT / f"{stem}_img_cpsam_mask_16bit.png")
        for method, pred in (("OrganoID_default", oid), ("OrganoSeg2_default", os2),
                             ("CellposeSAM_default", cp)):
            rows.append({"image": stem, "method": method, "metric_type": "mask", **mask_metrics(pred, gt)})
        if oq is not None:
            rows.append({"image": stem, "method": "OrgaQuant_default", "metric_type": "box",
                         **box_metrics(np.asarray(oq[stem]["boxes"]), boxes_from_labels(gt))})
    keys = list(rows[0])
    for row in rows:
        for k in row:
            if k not in keys: keys.append(k)
    with (OUT / "per_image_comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summary = []
    for method in sorted({r["method"] for r in rows}):
        rr = [r for r in rows if r["method"] == method]
        sums = {k: sum(float(x[k]) for x in rr) for k in ("pred_instances", "gt_instances", "tp", "fp", "fn")}
        p = sums["tp"] / sums["pred_instances"] if sums["pred_instances"] else 0
        rec = sums["tp"] / sums["gt_instances"] if sums["gt_instances"] else 0
        item = {"method": method, "metric_type": rr[0]["metric_type"], **{k: int(v) for k,v in sums.items()},
                "instance_precision": p, "instance_recall": rec, "instance_f1": 2*p*rec/(p+rec) if p+rec else 0,
                "mean_matched_iou": float(np.mean([x["mean_matched_iou"] for x in rr]))}
        if rr[0]["metric_type"] == "mask":
            item.update(mean_pixel_iou=float(np.mean([x["pixel_iou"] for x in rr])),
                        mean_pixel_dice=float(np.mean([x["pixel_dice"] for x in rr])),
                        mean_panoptic_quality=float(np.mean([x["panoptic_quality"] for x in rr])))
        summary.append(item)
    with (OUT / "summary_comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in summary for k in r})); w.writeheader(); w.writerows(summary)
    (OUT / "comparison_results.json").write_text(json.dumps({"iou_match_threshold": .5, "rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
