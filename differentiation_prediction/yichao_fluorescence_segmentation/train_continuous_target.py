#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_fluorescence_segmentation.datasets import (
    ContinuousFluorescenceTargetDataset,
    make_balanced_weights,
)
from differentiation_prediction.yichao_fluorescence_segmentation.models import GlobalGatedSegUNet
from differentiation_prediction.yichao_fluorescence_segmentation.utils import (
    DEFAULT_OUTPUT_ROOT,
    gray_rgb,
    green_rgb,
    heat_rgb,
    read_csv,
    save_grid,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train B-to-continuous-suppressed-F prediction for Yichao data.")
    parser.add_argument("--target-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("analysis-outputs") / "yichao_fluorescence_continuous" / "runs" / "soft_suppressed_unet_v1")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--include-organoid-mask", action="store_true", default=True)
    parser.add_argument("--no-organoid-mask", action="store_false", dest="include_organoid_mask")
    parser.add_argument("--include-distance", action="store_true", default=True)
    parser.add_argument("--no-distance", action="store_false", dest="include_distance")
    parser.add_argument("--target-scale", type=float, default=2.5)
    parser.add_argument("--soft-mask-dilate", type=int, default=9)
    parser.add_argument("--soft-mask-sigma", type=float, default=3.5)
    parser.add_argument("--soft-mask-floor", type=float, default=0.35)
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--lambda-mse", type=float, default=0.25)
    parser.add_argument("--lambda-soft-dice", type=float, default=0.50)
    parser.add_argument("--lambda-focal", type=float, default=1.0)
    parser.add_argument("--lambda-total", type=float, default=0.20)
    parser.add_argument("--lambda-bg", type=float, default=0.08)
    parser.add_argument("--background-weight", type=float, default=0.25)
    parser.add_argument("--signal-weight", type=float, default=4.0)
    parser.add_argument("--signal-power", type=float, default=0.5)
    parser.add_argument("--focal-alpha", type=float, default=0.85)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--bg-threshold", type=float, default=0.02)
    parser.add_argument("--metric-threshold", type=float, default=0.20)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--positive-sample-weight", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--panel-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--keep-periodic", type=int, default=8)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-patience-evals", type=int, default=25)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0005)
    parser.add_argument("--early-stop-min-epochs", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    parser.add_argument("--overfit-count", type=int, default=None)
    return parser.parse_args()


def make_loader(dataset: ContinuousFluorescenceTargetDataset, args: argparse.Namespace, shuffle: bool, sampler: WeightedRandomSampler | None = None) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": bool(shuffle and sampler is None),
        "sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def cosine_lr(epoch: int, args: argparse.Namespace) -> float:
    if args.epochs <= 1:
        return args.min_lr
    progress = min(max(epoch - 1, 0) / max(args.epochs - 1, 1), 1.0)
    return float(args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress)))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def optimizer_trainable_parameters(optimizer: torch.optim.Optimizer) -> list[torch.nn.Parameter]:
    return [
        param
        for group in optimizer.param_groups
        for param in group.get("params", [])
        if getattr(param, "requires_grad", False)
    ]


def continuous_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["logits"]
    pred = torch.sigmoid(outputs["logits"])
    target = batch["target"].to(pred.device)
    support = (target >= args.metric_threshold).float()
    weight = args.background_weight + args.signal_weight * torch.pow(target.clamp(0.0, 1.0), args.signal_power)
    smooth_l1 = (F.smooth_l1_loss(pred, target, reduction="none") * weight).sum() / weight.sum().clamp_min(1.0)
    mse = (((pred - target) ** 2) * weight).sum() / weight.sum().clamp_min(1.0)
    dice_num = 2.0 * (pred * support).sum(dim=(1, 2, 3)) + 1e-6
    dice_den = (pred + support).sum(dim=(1, 2, 3)) + 1e-6
    soft_dice = (1.0 - dice_num / dice_den).mean()
    bce = F.binary_cross_entropy_with_logits(logits, support, reduction="none")
    prob_for_focal = pred * support + (1.0 - pred) * (1.0 - support)
    alpha = args.focal_alpha * support + (1.0 - args.focal_alpha) * (1.0 - support)
    focal = (alpha * torch.pow((1.0 - prob_for_focal).clamp_min(1e-6), args.focal_gamma) * bce).mean()
    pred_mean = pred.mean(dim=(1, 2, 3))
    target_mean = target.mean(dim=(1, 2, 3))
    total = F.smooth_l1_loss(torch.log1p(pred_mean * 1000.0), torch.log1p(target_mean * 1000.0))
    background = target <= args.bg_threshold
    bg_loss = pred[background].mean() if bool(background.any()) else pred.mean() * 0.0
    loss = (
        args.lambda_l1 * smooth_l1
        + args.lambda_mse * mse
        + args.lambda_soft_dice * soft_dice
        + args.lambda_focal * focal
        + args.lambda_total * total
        + args.lambda_bg * bg_loss
    )
    return loss, {
        "loss_l1": float(smooth_l1.detach().cpu()),
        "loss_mse": float(mse.detach().cpu()),
        "loss_soft_dice": float(soft_dice.detach().cpu()),
        "loss_focal": float(focal.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
        "loss_bg": float(bg_loss.detach().cpu()),
    }


def save_panel(batch: dict[str, Any], pred: torch.Tensor, path: Path, max_items: int = 6) -> None:
    rows = []
    labels = []
    count = min(max_items, pred.shape[0])
    for idx in range(count):
        bf = batch["brightfield"][idx].squeeze(0).detach().cpu().numpy()
        fl = batch["fluorescence"][idx].squeeze(0).detach().cpu().numpy()
        target = batch["target"][idx].squeeze(0).detach().cpu().numpy()
        output = pred[idx].squeeze(0).detach().cpu().numpy()
        error = abs(output - target)
        rows.append([gray_rgb(bf), green_rgb(fl), gray_rgb(target), gray_rgb(output), heat_rgb(error)])
        labels.append(f"{batch['dataset'][idx]} | {batch['target_status'][idx]} | {str(batch['instance_id'][idx])[:90]}")
    save_grid(path, rows, ["B", "raw F", "target y", "pred y", "abs error"], labels, tile=180)


@torch.no_grad()
def evaluate(
    model: GlobalGatedSegUNet,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    panel_path: Path | None = None,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    seen = 0
    pixel_count = 0.0
    l1_sum = 0.0
    mse_sum = 0.0
    pred_sum = 0.0
    target_sum = 0.0
    pred_sq_sum = 0.0
    target_sq_sum = 0.0
    xy_sum = 0.0
    tp = fp = fn = 0.0
    first_panel = True
    for batch_index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        outputs = model(image)
        loss, _ = continuous_loss(outputs, batch, args)
        pred = torch.sigmoid(outputs["logits"]).detach()
        target = batch["target"].to(device, non_blocking=True)
        diff = pred - target
        batch_pixels = float(target.numel())
        pixel_count += batch_pixels
        l1_sum += float(diff.abs().sum().detach().cpu())
        mse_sum += float(diff.square().sum().detach().cpu())
        pred_sum += float(pred.sum().detach().cpu())
        target_sum += float(target.sum().detach().cpu())
        pred_sq_sum += float(pred.square().sum().detach().cpu())
        target_sq_sum += float(target.square().sum().detach().cpu())
        xy_sum += float((pred * target).sum().detach().cpu())
        pred_binary = pred >= args.metric_threshold
        target_binary = target >= args.metric_threshold
        tp += float((pred_binary & target_binary).sum().detach().cpu())
        fp += float((pred_binary & ~target_binary).sum().detach().cpu())
        fn += float((~pred_binary & target_binary).sum().detach().cpu())
        batch_size = image.shape[0]
        loss_sum += float(loss.detach().cpu()) * batch_size
        seen += batch_size
        if panel_path is not None and first_panel:
            save_panel(batch, pred.cpu(), panel_path)
            first_panel = False
        if args.limit_eval_batches is not None and batch_index + 1 >= args.limit_eval_batches:
            break
    pred_mean = pred_sum / max(pixel_count, 1.0)
    target_mean = target_sum / max(pixel_count, 1.0)
    cov = xy_sum / max(pixel_count, 1.0) - pred_mean * target_mean
    pred_var = max(pred_sq_sum / max(pixel_count, 1.0) - pred_mean * pred_mean, 0.0)
    target_var = max(target_sq_sum / max(pixel_count, 1.0) - target_mean * target_mean, 0.0)
    pearson = cov / max(math.sqrt(pred_var * target_var), 1e-12)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "loss": loss_sum / max(seen, 1),
        "mae": l1_sum / max(pixel_count, 1.0),
        "rmse": math.sqrt(mse_sum / max(pixel_count, 1.0)),
        "pearson": pearson,
        "pred_mean": pred_mean,
        "target_mean": target_mean,
        "total_intensity_ratio": pred_sum / max(target_sum, 1e-12),
        "threshold_precision": precision,
        "threshold_recall": recall,
        "threshold_f1": f1,
        "n": seen,
    }


def selection_score(metrics: dict[str, float]) -> float:
    return float(metrics["pearson"]) - 0.25 * abs(math.log(max(metrics["total_intensity_ratio"], 1e-6)))


def plot_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    val_rows = [row for row in rows if "val_loss" in row]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(epochs, [row.get("train_loss", math.nan) for row in rows], label="train loss")
    if val_rows:
        val_epochs = [int(row["epoch"]) for row in val_rows]
        axes[0, 0].plot(val_epochs, [row.get("val_loss", math.nan) for row in val_rows], "o-", label="val loss")
        axes[0, 1].plot(val_epochs, [row.get("val_pearson", math.nan) for row in val_rows], "o-", label="Pearson")
        axes[1, 0].plot(val_epochs, [row.get("val_mae", math.nan) for row in val_rows], "o-", label="MAE")
        axes[1, 0].plot(val_epochs, [row.get("val_rmse", math.nan) for row in val_rows], "o-", label="RMSE")
        axes[1, 1].plot(val_epochs, [row.get("val_threshold_precision", math.nan) for row in val_rows], "o-", label="precision@thr")
        axes[1, 1].plot(val_epochs, [row.get("val_threshold_recall", math.nan) for row in val_rows], "o-", label="recall@thr")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.legend()
        ax.set_xlabel("epoch")
    axes[0, 0].set_title("Loss")
    axes[0, 1].set_title("Continuous target correlation")
    axes[1, 0].set_title("Regression error")
    axes[1, 1].set_title("QA threshold metrics")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_periodic(path: Path, payload: dict[str, Any], keep: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    checkpoints = sorted(path.parent.glob("epoch_*.pt"))
    if keep > 0 and len(checkpoints) > keep:
        for old in checkpoints[: len(checkpoints) - keep]:
            old.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "predictions").mkdir(parents=True, exist_ok=True)
    (args.output_root / "plots").mkdir(parents=True, exist_ok=True)
    (args.output_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    manifest_path = args.target_root / "manifests" / "segmentation_targets_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing target manifest: {manifest_path}")
    rows = read_csv(manifest_path)
    if args.overfit_count is not None:
        train_rows = [row for row in rows if row["split"] == "train"][: args.overfit_count]
        val_rows = train_rows
        test_rows = train_rows
    else:
        train_rows = [row for row in rows if row["split"] == "train"]
        val_rows = [row for row in rows if row["split"] == "val"]
        test_rows = [row for row in rows if row["split"] == "test"]
    dataset_kwargs = {
        "image_size": args.image_size,
        "include_organoid_mask": args.include_organoid_mask,
        "include_distance": args.include_distance,
        "target_scale": args.target_scale,
        "soft_mask_dilate": args.soft_mask_dilate,
        "soft_mask_sigma": args.soft_mask_sigma,
        "soft_mask_floor": args.soft_mask_floor,
    }
    train_ds = ContinuousFluorescenceTargetDataset(train_rows, augment=args.overfit_count is None, **dataset_kwargs)
    val_ds = ContinuousFluorescenceTargetDataset(val_rows, **dataset_kwargs)
    test_ds = ContinuousFluorescenceTargetDataset(test_rows, **dataset_kwargs)
    sampler = None
    if args.balanced_sampler:
        weights = make_balanced_weights(train_rows, args.positive_sample_weight)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = make_loader(train_ds, args, shuffle=True, sampler=sampler)
    val_loader = make_loader(val_ds, args, shuffle=False)
    test_loader = make_loader(test_ds, args, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    in_channels = 1 + int(args.include_organoid_mask) + int(args.include_distance)
    model = GlobalGatedSegUNet(in_channels=in_channels, base_channels=args.base_channels, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    grad_clip_params = optimizer_trainable_parameters(optimizer)
    metrics: list[dict[str, Any]] = []
    best_score = -1e18
    best_epoch = 0
    start_epoch = 1
    last_path = args.output_root / "last_model.pt"
    metrics_path = args.output_root / "metrics.jsonl"
    write_json(
        args.output_root / "run_config.json",
        {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "device": str(device),
            "train_count": len(train_ds),
            "val_count": len(val_ds),
            "test_count": len(test_ds),
            "input_channels": in_channels,
            "target": "one_channel_soft_suppressed_fluorescence_clipped_0_1",
        },
    )
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        metrics = list(checkpoint.get("metrics", []))
        best_score = float(checkpoint.get("best_score", best_score))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        with metrics_path.open("w", encoding="utf-8") as handle:
            for row in metrics:
                handle.write(json.dumps(row) + "\n")
    elif metrics_path.exists():
        metrics_path.unlink()
    stale_evals = 0
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        set_lr(optimizer, cosine_lr(epoch, args))
        model.train()
        loss_sum = 0.0
        seen = 0
        component_sums: dict[str, float] = {}
        grad_norm_value = 0.0
        for batch_index, batch in enumerate(train_loader):
            image = batch["image"].to(device, non_blocking=True)
            outputs = model(image)
            loss, components = continuous_loss(outputs, batch, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(grad_clip_params, 3.0, foreach=False)
            optimizer.step()
            grad_norm_value = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
            batch_size = image.shape[0]
            loss_sum += float(loss.detach().cpu()) * batch_size
            seen += batch_size
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.0) + value * batch_size
            if args.limit_train_batches is not None and batch_index + 1 >= args.limit_train_batches:
                break
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "lr": cosine_lr(epoch, args),
            "grad_norm": grad_norm_value,
        }
        row.update({f"train_{key}": value / max(seen, 1) for key, value in component_sums.items()})
        should_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if should_eval:
            panel_path = args.output_root / "predictions" / f"val_epoch_{epoch:04d}.png" if (epoch == 1 or epoch % args.panel_every == 0) else None
            val_metrics = evaluate(model, val_loader, device, args, panel_path=panel_path)
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            score = selection_score(val_metrics)
            row["val_selection_score"] = score
            if score > best_score + args.early_stop_min_delta:
                best_score = score
                best_epoch = epoch
                stale_evals = 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "score": best_score, "metrics": row}, args.output_root / "best_model.pt")
                if panel_path is not None:
                    shutil.copy2(panel_path, args.output_root / "predictions" / "val_best.png")
            else:
                stale_evals += 1
            row["early_stop_stale_evals"] = stale_evals
            if args.early_stop and epoch >= args.early_stop_min_epochs and stale_evals >= args.early_stop_patience_evals:
                row["early_stop_triggered"] = True
                stopped_early = True
        metrics.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": metrics, "best_score": best_score, "best_epoch": best_epoch}
        torch.save(payload, last_path)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_periodic(args.output_root / "checkpoints" / f"epoch_{epoch:04d}.pt", payload, args.keep_periodic)
        plot_metrics(metrics, args.output_root / "plots" / "training_metrics.png")
        print(json.dumps(row), flush=True)
        if stopped_early:
            write_json(args.output_root / "early_stop_summary.json", {"triggered": True, "epoch": epoch, "best_epoch": best_epoch, "best_score": best_score, "stale_evals": stale_evals})
            break
    if not (args.output_root / "best_model.pt").exists():
        torch.save({"model": model.state_dict(), "epoch": metrics[-1]["epoch"], "score": best_score, "metrics": metrics[-1]}, args.output_root / "best_model.pt")
    best = torch.load(args.output_root / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(model, test_loader, device, args, panel_path=args.output_root / "predictions" / "test_best.png")
    write_json(args.output_root / "test_metrics.json", {**test_metrics, "best_epoch": int(best.get("epoch", 0)), "stopped_early": stopped_early})
    print(json.dumps({"stage": "continuous_target_finished", "test": test_metrics, "best_epoch": int(best.get("epoch", 0)), "stopped_early": stopped_early}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
