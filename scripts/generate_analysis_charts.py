#!/usr/bin/env python3
"""Generate reproducible longitudinal charts from OrganoidAgent result tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="results.json or results.csv")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--metrics", default="area,growth_rate")
    p.add_argument("--x-field", default="day")
    p.add_argument("--group-field", default="group")
    return p.parse_args()


def load_rows(path: Path) -> tuple[list[dict], dict]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return list(payload.get("rows", payload.get("samples", []))), payload
        return list(payload), {}
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle)), {}


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()
    rows, metadata = load_rows(Path(args.input))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Import only here so the API can still start when chart dependencies are optional.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    charts = []
    for metric in [x.strip() for x in args.metrics.split(",") if x.strip()]:
        if metric in {"viability", "viability_percent"}:
            values = []
            for row in rows:
                value = number(row.get(metric))
                filename = str(row.get("filename") or "").strip()
                if value is None or not filename:
                    continue
                values.append((filename, value, str(row.get("level") or "")))
            if not values:
                continue
            values.sort(key=lambda item: item[1], reverse=True)
            labels = [item[0] for item in values]
            scores = [item[1] for item in values]
            level_colors = {
                "High viability": "#2e8b57",
                "Intermediate viability": "#e6a23c",
                "Low viability": "#c94c4c",
            }
            colors = [level_colors.get(item[2], "#4c78a8") for item in values]
            fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
            bars = ax.bar(labels, scores, color=colors)
            ax.set_xlabel("Image")
            ax.set_ylabel("Viability (%)" if metric == "viability_percent" else "Viability")
            ax.set_title("Predicted viability by image")
            ax.set_ylim(0, 100 if metric == "viability_percent" else 1)
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=45)
            label_format = "{:.1f}%" if metric == "viability_percent" else "{:.3f}"
            ax.bar_label(bars, labels=[label_format.format(value) for value in scores], padding=3, fontsize=8)
            fig.tight_layout()
            filename = f"{metric}_bar.png"
            fig.savefig(output / filename)
            plt.close(fig)
            charts.append({
                "metric": metric,
                "path": str(output / filename),
                "filename": filename,
                "categories": labels,
            })
            continue
        grouped = {}
        for row in rows:
            x = number(row.get(args.x_field))
            y = number(row.get(metric))
            if x is None or y is None:
                continue
            group = str(row.get(args.group_field, "all"))
            grouped.setdefault(group, []).append((x, y))
        if not grouped:
            continue
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
        matrix = []
        for group, points in sorted(grouped.items()):
            points.sort()
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", linewidth=1, alpha=0.35, label=str(group))
            matrix.append({x: y for x, y in points})
        all_x = sorted({x for values in matrix for x in values})
        means, stds = [], []
        for x in all_x:
            values = [values[x] for values in matrix if x in values]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        ax.plot(all_x, means, color="black", linewidth=2.5, marker="o", label="Mean")
        ax.fill_between(all_x, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color="black", alpha=0.12, label="Mean ± SD")
        units = metadata.get("units", "")
        ylabel = f"{metric} ({units})" if units and metric == "area" else metric
        ax.set_xlabel(args.x_field)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} across {args.x_field}")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=7, ncol=2)
        fig.tight_layout()
        filename = f"{metric}_line.png"
        fig.savefig(output / filename)
        plt.close(fig)
        charts.append({"metric": metric, "path": str(output / filename), "filename": filename, "groups": len(grouped), "timepoints": all_x})
    (output / "charts.json").write_text(json.dumps({"charts": charts}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
