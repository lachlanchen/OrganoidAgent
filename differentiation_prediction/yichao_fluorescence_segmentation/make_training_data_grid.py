#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_fluorescence_segmentation.utils import (
    DEFAULT_OUTPUT_ROOT,
    gray_rgb,
    green_rgb,
    load_font,
    read_csv,
    read_gray_float,
    red_rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render exact B/F/F-mask training examples from the Yichao target manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_ROOT / "manifests" / "segmentation_targets_manifest.csv")
    parser.add_argument("--output", type=Path, default=Path("visualizations") / "yichao_training_b_f_mask_15x10.png")
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--groups", type=int, default=5, help="Number of B/F/mask triplets per row. 5 groups gives 15 columns.")
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="train")
    parser.add_argument("--mix-status", action="store_true", help="Sample positives, negatives, and overexposure-suppressed examples evenly when possible.")
    return parser.parse_args()


def resize(image: Image.Image, tile: int, *, mask: bool = False) -> Image.Image:
    resample = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    return image.resize((tile, tile), resample)


def mask_overlay(mask: np.ndarray, ignore: np.ndarray) -> Image.Image:
    base = Image.new("RGB", (mask.shape[1], mask.shape[0]), (0, 0, 0))
    pos = green_rgb(mask)
    ign = red_rgb(ignore * 0.75)
    base = Image.blend(base, pos, 0.95)
    base = Image.blend(base, ign, 0.55)
    return base


def choose_rows(rows: list[dict[str, str]], count: int, seed: int, mix_status: bool) -> list[dict[str, str]]:
    rng = random.Random(seed)
    if not mix_status:
        rows = list(rows)
        rng.shuffle(rows)
        return rows[:count]
    selected: list[dict[str, str]] = []
    statuses = ["positive", "negative", "overexposure_suppressed"]
    buckets = {status: [row for row in rows if row.get("target_status") == status] for status in statuses}
    per_status = max(1, count // len(statuses))
    for status in statuses:
        bucket = buckets[status]
        rng.shuffle(bucket)
        selected.extend(bucket[:per_status])
    remaining = [row for row in rows if row not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, count - len(selected))])
    rng.shuffle(selected)
    return selected[:count]


def render_grid(rows: list[dict[str, str]], output: Path, groups: int, row_count: int, tile: int) -> None:
    header_h = 34
    label_h = 34
    columns = groups * 3
    width = columns * tile
    height = header_h + row_count * (tile + label_h)
    canvas = Image.new("RGB", (width, height), (16, 19, 22))
    draw = ImageDraw.Draw(canvas)
    font = load_font(12)
    header_font = load_font(14)
    for group in range(groups):
        for offset, header in enumerate(("B", "F", "F-mask")):
            x = (group * 3 + offset) * tile + 6
            draw.text((x, 9), header, fill=(235, 240, 242), font=header_font)
    for index, row in enumerate(rows[: row_count * groups]):
        grid_row = index // groups
        group = index % groups
        x0 = group * 3 * tile
        y0 = header_h + grid_row * (tile + label_h)
        brightfield = gray_rgb(read_gray_float(Path(row["brightfield_crop_path"])))
        fluorescence = green_rgb(read_gray_float(Path(row["fluorescence_crop_path"])))
        positive = read_gray_float(Path(row["positive_mask_path"]))
        ignore = read_gray_float(Path(row["ignore_mask_path"]))
        mask = mask_overlay(positive, ignore)
        canvas.paste(resize(brightfield, tile), (x0, y0))
        canvas.paste(resize(fluorescence, tile), (x0 + tile, y0))
        canvas.paste(resize(mask, tile, mask=True), (x0 + tile * 2, y0))
        label = (
            f"{index:02d} {row.get('split')} {row.get('target_status')} "
            f"pos={float(row.get('target_positive_fraction', 0)):.3f} "
            f"{row.get('dataset')}"
        )
        draw.text((x0 + 4, y0 + tile + 5), label[:48], fill=(188, 196, 202), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    args = parse_args()
    rows = read_csv(args.manifest)
    if args.split != "all":
        rows = [row for row in rows if row.get("split") == args.split]
    if not rows:
        raise SystemExit(f"No rows available for split={args.split} in {args.manifest}")
    count = args.rows * args.groups
    selected = choose_rows(rows, count, args.seed, args.mix_status)
    render_grid(selected, args.output, args.groups, args.rows, args.tile)
    sidecar = args.output.with_suffix(".txt")
    with sidecar.open("w", encoding="utf-8") as handle:
        handle.write(f"manifest={args.manifest}\n")
        handle.write(f"split={args.split}\n")
        handle.write(f"rows={args.rows}\n")
        handle.write(f"groups={args.groups}\n")
        handle.write(f"tile={args.tile}\n")
        handle.write(f"seed={args.seed}\n")
        handle.write(f"mix_status={args.mix_status}\n")
        for index, row in enumerate(selected):
            handle.write(
                f"{index}\t{row.get('split')}\t{row.get('target_status')}\t"
                f"{row.get('target_positive_fraction')}\t{row.get('dataset')}\t{row.get('instance_id')}\n"
            )
    print(args.output)
    print(sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
