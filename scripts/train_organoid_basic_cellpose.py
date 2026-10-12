#!/usr/bin/env python3
"""Train and evaluate a Cellpose model specialized for OrganoidBasic.

This script reads the existing OrganoidBasic image/mask pairs but never edits
them. Checkpoints and reports are written below analysis-outputs only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from cellpose import models, train


def pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    directory = root / split
    result = []
    for image in sorted(directory.glob("*_img.jpg")):
        mask = image.with_name(image.name.replace("_img.jpg", "_masks_organoid.png"))
        if mask.exists():
            result.append((image, mask))
    if not result:
        raise RuntimeError(f"No image/mask pairs found in {directory}")
    return result


def load_pair(image_path: Path, mask_path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        raise RuntimeError(f"Could not read {image_path}")
    if mask.ndim != 2:
        mask = mask[..., 0]
    # The labels are instance IDs, not merely a binary foreground mask.
    mask = mask.astype(np.int32)
    if image.shape[:2] != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return image, mask


def iou_scores(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    p = pred > 0
    t = truth > 0
    intersection = int(np.logical_and(p, t).sum())
    union = int(np.logical_or(p, t).sum())
    return {
        "iou": intersection / union if union else 0.0,
        "precision": intersection / int(p.sum()) if p.sum() else 0.0,
        "recall": intersection / int(t.sum()) if t.sum() else 0.0,
        "pred_area": int(p.sum()),
        "truth_area": int(t.sum()),
        "pred_instances": int(pred.max()),
    }


def evaluate(model_path: Path, validation: list[tuple[Path, Path]], size: int,
             diameters: list[float], cellprob: float, flow: float, limit: int | None) -> dict:
    use_gpu = bool(torch.cuda.is_available())
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))
    rows = []
    selected = validation[:limit] if limit else validation
    for diameter in diameters:
        scores = []
        for image_path, mask_path in selected:
            image, truth = load_pair(image_path, mask_path, size)
            masks, *_ = model.eval(
                image,
                diameter=diameter,
                channels=[0, 0],
                normalize=True,
                resample=False,
                cellprob_threshold=cellprob,
                flow_threshold=flow,
                min_size=15,
                do_3D=False,
            )
            scores.append(iou_scores(masks, truth))
        keys = ["iou", "precision", "recall", "pred_area", "truth_area", "pred_instances"]
        aggregate = {key: float(np.mean([row[key] for row in scores])) for key in keys}
        aggregate.update({"diameter": diameter, "n_images": len(scores)})
        rows.append(aggregate)
        print(json.dumps(aggregate), flush=True)
    return {"model": str(model_path), "image_size": size, "gpu": use_gpu, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/media/yinghan/My Passport/review/OrganoidBasic_v20211206")
    parser.add_argument("--output", default="analysis-outputs/organoid_basic_cellpose")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0,
                        help="Optional small subset for a smoke test; 0 uses all training pairs.")
    parser.add_argument("--evaluate-only", default="")
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_pairs = pairs(dataset, "train")
    val_pairs = pairs(dataset, "val")
    if args.train_limit:
        train_pairs = train_pairs[:args.train_limit]
    train_data, train_labels = zip(*(load_pair(i, m, args.size) for i, m in train_pairs))
    val_data, val_labels = zip(*(load_pair(i, m, args.size) for i, m in val_pairs))
    metadata = {
        "dataset": str(dataset),
        "train_images": len(train_pairs),
        "validation_images": len(val_pairs),
        "image_size": args.size,
        "epochs": args.epochs,
        "gpu": bool(torch.cuda.is_available()),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (output / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.evaluate_only:
        model_path = Path(args.evaluate_only).resolve()
    else:
        model = models.CellposeModel(gpu=bool(torch.cuda.is_available()), pretrained_model="cpsam")
        model_path, train_losses, test_losses = train.train_seg(
            model.net,
            train_data=list(train_data),
            train_labels=list(train_labels),
            test_data=list(val_data),
            test_labels=list(val_labels),
            load_files=False,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            n_epochs=args.epochs,
            weight_decay=0.1,
            normalize=True,
            compute_flows=False,
            save_path=str(output),
            save_every=max(args.epochs, 1),
            bsize=min(256, args.size),
            min_train_masks=1,
            model_name="organoid_basic_cpsam",
        )
        metadata["checkpoint"] = str(model_path)
        metadata["train_loss_last"] = float(train_losses[-1])
        metadata["test_loss_last"] = float(test_losses[-1]) if len(test_losses) else None
        (output / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report = evaluate(
        Path(model_path), val_pairs, args.size,
        diameters=[20, 30, 40, 55, 70, 90],
        cellprob=0.0,
        flow=0.4,
        limit=args.eval_limit or None,
    )
    report["checkpoint"] = str(model_path)
    (output / "validation_scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
