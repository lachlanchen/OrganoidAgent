#!/usr/bin/env python3
"""Train a pix2pix model for brightfield -> fluorescence translation.

Default setup:
- Training data: Data-Yichao-2
- Verification data (val/test split): Data-Yichao-1
- Input channel: c0 (brightfield)
- Target channel: c1 (fluorescence)

Outputs are written under results/ with checkpoints, logs, and exported predictions.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

CHANNEL_PATTERN = re.compile(r"^(?P<base>.+)_c(?P<channel>\d+)\.(?P<ext>jpe?g)$", re.IGNORECASE)


@dataclass(frozen=True)
class PairItem:
    input_path: Path
    target_path: Path
    key: str


class PairedYichaoDataset(Dataset):
    def __init__(self, pairs: Sequence[PairItem], image_size: int, random_flip: bool) -> None:
        self.pairs = list(pairs)
        self.image_size = image_size
        self.random_flip = random_flip

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_gray(self, path: Path) -> Image.Image:
        image = Image.open(path).convert("L")
        if self.image_size > 0 and image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        return image

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor * 2.0 - 1.0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        item = self.pairs[idx]
        input_img = self._load_gray(item.input_path)
        target_img = self._load_gray(item.target_path)

        if self.random_flip and random.random() < 0.5:
            input_img = input_img.transpose(Image.FLIP_LEFT_RIGHT)
            target_img = target_img.transpose(Image.FLIP_LEFT_RIGHT)

        return {
            "input": self._to_tensor(input_img),
            "target": self._to_tensor(target_img),
            "key": item.key,
        }


class UNetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        blocks: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not use_norm),
        ]
        if use_norm:
            blocks.append(nn.InstanceNorm2d(out_channels, affine=True))
        blocks.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            blocks.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        blocks: List[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            blocks.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return torch.cat((x, skip), dim=1)


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, use_norm=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.1)
        self.down5 = UNetDown(512, 512, dropout=0.1)
        self.down6 = UNetDown(512, 512, dropout=0.1)
        self.down7 = UNetDown(512, 512, dropout=0.1)
        self.down8 = UNetDown(512, 512, use_norm=False, dropout=0.1)

        self.up1 = UNetUp(512, 512, dropout=0.1)
        self.up2 = UNetUp(1024, 512, dropout=0.1)
        self.up3 = UNetUp(1024, 512, dropout=0.1)
        self.up4 = UNetUp(1024, 512, dropout=0.1)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()

        def block(in_c: int, out_c: int, normalize: bool = True) -> List[nn.Module]:
            layers: List[nn.Module] = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels * 2, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, condition: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat((condition, target), dim=1))


def build_pairs(root: Path, input_channel: int, target_channel: int) -> List[PairItem]:
    root = root.resolve()
    grouped: Dict[str, Dict[int, Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        match = CHANNEL_PATTERN.match(path.name)
        if not match:
            continue

        base = match.group("base")
        channel = int(match.group("channel"))
        key = f"{path.parent.relative_to(root)}/{base}"
        grouped.setdefault(key, {})[channel] = path

    pairs: List[PairItem] = []
    for key in sorted(grouped.keys()):
        channels = grouped[key]
        if input_channel in channels and target_channel in channels:
            pairs.append(PairItem(input_path=channels[input_channel], target_path=channels[target_channel], key=key))
    return pairs


def split_pairs(pairs: Sequence[PairItem], val_ratio: float, seed: int) -> Tuple[List[PairItem], List[PairItem]]:
    items = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(items)
    if len(items) <= 1:
        return items, []

    val_count = max(1, int(len(items) * val_ratio))
    val_count = min(val_count, len(items) - 1)
    val_items = items[:val_count]
    test_items = items[val_count:]
    return val_items, test_items


def denorm_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().clamp(-1, 1)
    arr = (arr + 1.0) * 0.5
    arr = arr.squeeze(0).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


def save_triplet_grid(inputs: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor, path: Path, limit: int = 4) -> None:
    rows: List[np.ndarray] = []
    count = min(limit, inputs.shape[0])
    for i in range(count):
        inp = denorm_to_uint8(inputs[i])
        prd = denorm_to_uint8(preds[i])
        tgt = denorm_to_uint8(targets[i])
        rows.append(np.concatenate([inp, prd, tgt], axis=1))

    if not rows:
        return

    canvas = np.concatenate(rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").save(path)


def batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred01 = (pred.detach() + 1.0) * 0.5
    tgt01 = (target.detach() + 1.0) * 0.5

    mae = torch.mean(torch.abs(pred01 - tgt01)).item()
    mse = torch.mean((pred01 - tgt01) ** 2).item()
    psnr = 20.0 * math.log10(1.0 / math.sqrt(max(mse, 1e-12)))
    return {"mae": mae, "mse": mse, "psnr": psnr}


def evaluate(
    generator: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    l1_loss: nn.Module,
    max_steps: int | None = None,
    sample_path: Path | None = None,
) -> Dict[str, float]:
    generator.eval()
    l1_total = 0.0
    mae_total = 0.0
    mse_total = 0.0
    psnr_total = 0.0
    count = 0

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            pred = generator(inp)

            l1 = l1_loss(pred, tgt).item()
            m = batch_metrics(pred, tgt)

            batch_size = inp.shape[0]
            l1_total += l1 * batch_size
            mae_total += m["mae"] * batch_size
            mse_total += m["mse"] * batch_size
            psnr_total += m["psnr"] * batch_size
            count += batch_size

            if step == 0 and sample_path is not None:
                save_triplet_grid(inp, pred, tgt, sample_path)

            if max_steps is not None and step + 1 >= max_steps:
                break

    if count == 0:
        return {"l1": float("nan"), "mae": float("nan"), "mse": float("nan"), "psnr": float("nan")}

    return {
        "l1": l1_total / count,
        "mae": mae_total / count,
        "mse": mse_total / count,
        "psnr": psnr_total / count,
    }


def save_config(path: Path, args: argparse.Namespace, counts: Dict[str, int]) -> None:
    serialized_args = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized_args[key] = str(value)
        else:
            serialized_args[key] = value

    payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "args": serialized_args,
        "counts": counts,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_pairs = build_pairs(args.train_root, args.input_channel, args.target_channel)
    verify_pairs = build_pairs(args.verify_root, args.input_channel, args.target_channel)
    val_pairs, test_pairs = split_pairs(verify_pairs, args.verify_val_ratio, args.seed)

    if not train_pairs:
        raise RuntimeError(f"No train pairs found in {args.train_root}")
    if not val_pairs:
        raise RuntimeError(f"No validation pairs found in {args.verify_root}")
    if not test_pairs:
        raise RuntimeError(f"No test pairs found in {args.verify_root}; decrease --verify-val-ratio")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_root / f"pix2pix_yichao_{timestamp}"
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    test_dir = run_dir / "test_predictions"
    for folder in [run_dir, checkpoints_dir, samples_dir, test_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    counts = {
        "train_pairs": len(train_pairs),
        "verify_pairs": len(verify_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
    }
    save_config(run_dir / "config.json", args, counts)

    train_ds = PairedYichaoDataset(train_pairs, args.image_size, random_flip=True)
    val_ds = PairedYichaoDataset(val_pairs, args.image_size, random_flip=False)
    test_ds = PairedYichaoDataset(test_pairs, args.image_size, random_flip=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    with torch.no_grad():
        probe = torch.zeros((1, 1, args.image_size, args.image_size), device=device)
        patch_shape = discriminator(probe, probe).shape[1:]

    metrics_csv = run_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "epoch",
                "train_g_loss",
                "train_d_loss",
                "train_l1",
                "val_l1",
                "val_mae",
                "val_psnr",
            ]
        )

    best_val_l1 = float("inf")
    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()

        g_loss_total = 0.0
        d_loss_total = 0.0
        l1_total = 0.0
        seen = 0

        for step, batch in enumerate(train_loader):
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            batch_size = inp.shape[0]

            valid = torch.ones((batch_size, *patch_shape), device=device)
            fake = torch.zeros((batch_size, *patch_shape), device=device)

            optimizer_g.zero_grad(set_to_none=True)
            fake_tgt = generator(inp)
            pred_fake = discriminator(inp, fake_tgt)
            g_gan = criterion_gan(pred_fake, valid)
            g_l1 = criterion_l1(fake_tgt, tgt)
            g_loss = g_gan + args.lambda_l1 * g_l1
            g_loss.backward()
            optimizer_g.step()

            optimizer_d.zero_grad(set_to_none=True)
            pred_real = discriminator(inp, tgt)
            d_real = criterion_gan(pred_real, valid)
            pred_fake_detached = discriminator(inp, fake_tgt.detach())
            d_fake = criterion_gan(pred_fake_detached, fake)
            d_loss = 0.5 * (d_real + d_fake)
            d_loss.backward()
            optimizer_d.step()

            g_loss_total += g_loss.item() * batch_size
            d_loss_total += d_loss.item() * batch_size
            l1_total += g_l1.item() * batch_size
            seen += batch_size

            if args.max_train_steps is not None and step + 1 >= args.max_train_steps:
                break

        train_metrics = {
            "g_loss": g_loss_total / max(seen, 1),
            "d_loss": d_loss_total / max(seen, 1),
            "l1": l1_total / max(seen, 1),
        }

        val_metrics = evaluate(
            generator,
            val_loader,
            device,
            criterion_l1,
            max_steps=args.max_eval_steps,
            sample_path=samples_dir / f"epoch_{epoch:03d}.png",
        )

        with metrics_csv.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    epoch,
                    f"{train_metrics['g_loss']:.6f}",
                    f"{train_metrics['d_loss']:.6f}",
                    f"{train_metrics['l1']:.6f}",
                    f"{val_metrics['l1']:.6f}",
                    f"{val_metrics['mae']:.6f}",
                    f"{val_metrics['psnr']:.6f}",
                ]
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"G={train_metrics['g_loss']:.4f} D={train_metrics['d_loss']:.4f} "
            f"train_l1={train_metrics['l1']:.4f} "
            f"val_l1={val_metrics['l1']:.4f} val_psnr={val_metrics['psnr']:.2f}"
        )

        latest_ckpt = checkpoints_dir / "latest.pt"
        torch.save(
            {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "val_l1": val_metrics["l1"],
            },
            latest_ckpt,
        )

        if val_metrics["l1"] < best_val_l1:
            best_val_l1 = val_metrics["l1"]
            torch.save(generator.state_dict(), checkpoints_dir / "best_generator.pt")
            torch.save(discriminator.state_dict(), checkpoints_dir / "best_discriminator.pt")

    print("Running final test evaluation using best generator...")
    best_generator = GeneratorUNet().to(device)
    best_generator.load_state_dict(torch.load(checkpoints_dir / "best_generator.pt", map_location=device))

    test_metrics = evaluate(
        best_generator,
        test_loader,
        device,
        criterion_l1,
        max_steps=args.max_eval_steps,
        sample_path=run_dir / "test_preview.png",
    )

    exported = 0
    best_generator.eval()
    with torch.no_grad():
        for batch in test_loader:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            pred = best_generator(inp)
            keys = batch["key"]

            for i in range(inp.shape[0]):
                key = str(keys[i]).replace("/", "__")
                save_triplet_grid(
                    inp[i : i + 1],
                    pred[i : i + 1],
                    tgt[i : i + 1],
                    test_dir / f"{key}.png",
                    limit=1,
                )
                exported += 1
                if args.test_save_limit > 0 and exported >= args.test_save_limit:
                    break
            if args.test_save_limit > 0 and exported >= args.test_save_limit:
                break

    summary = {
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "best_val_l1": best_val_l1,
        "test_metrics": test_metrics,
        "exported_test_predictions": exported,
        "run_dir": str(run_dir),
        "device": str(device),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Results: {run_dir}")
    print(f"Test L1: {test_metrics['l1']:.6f}, PSNR: {test_metrics['psnr']:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train pix2pix on Yichao brightfield/fluorescence pairs")
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path("/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-2/P11N&N39_Rep_DF_jpeg_all_by_object"),
        help="Training data root (Yichao-2)",
    )
    parser.add_argument(
        "--verify-root",
        type=Path,
        default=Path("/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-1/P11N&N39_Rep_DF_jpeg_all_by_object"),
        help="Verification data root (Yichao-1, split into val/test)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/home/lachlan/ProjectsLFS/OrganoidAgent/results"),
        help="Root directory for outputs",
    )
    parser.add_argument("--input-channel", type=int, default=0)
    parser.add_argument("--target-channel", type=int, default=1)
    parser.add_argument("--verify-val-ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-save-limit", type=int, default=200)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
