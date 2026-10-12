from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"F:\review\OrganoidAgent-main")
VAL = Path(r"F:\review\OrganoidBasic_v20211206\val")
BASE = ROOT / "analysis-outputs" / "external_method_comparison_3_images"
OUT = BASE / "overlay_comparisons"
OID = BASE / "organoid_default"
OS2 = BASE / "organoseg2_default"
CP = ROOT / "analysis-outputs" / "morphology_jobs" / "morphology-20261010T152202-2ac33ac5" / "runs"
STEMS = [
    "20210727_16294_ckunspgrq2th20z824ozpde5m",
    "20210727_18955_ckw3jwvx34z2310aoet4o6hwk",
    "20210727_20499_ckw3l33aa8yqg119087nsbx46",
]


def labels(path):
    a = np.asarray(Image.open(path))
    return a[..., 0] if a.ndim == 3 else a


def color_for(label):
    # Stable, bright instance color without relying on random state.
    return np.array([(label * 67) % 205 + 50, (label * 131) % 205 + 50,
                     (label * 193) % 205 + 50], dtype=np.uint8)


def overlay(rgb, lab, alpha=0.38):
    result = rgb.astype(np.float32).copy()
    colored = np.zeros_like(rgb)
    for value in np.unique(lab):
        if value:
            colored[lab == value] = color_for(int(value))
    fg = lab > 0
    result[fg] = (1 - alpha) * result[fg] + alpha * colored[fg]
    boundaries = np.zeros(lab.shape, np.uint8)
    for value in np.unique(lab):
        if value:
            mask = (lab == value).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(boundaries, contours, -1, 255, 2)
    result[boundaries > 0] = (255, 40, 210)
    return result.astype(np.uint8)


def titled(image, title, width=460):
    h = round(image.height * width / image.width)
    image = image.resize((width, h), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, h + 46), "white")
    panel.paste(image, (0, 46))
    ImageDraw.Draw(panel).text((12, 12), title, fill="black", font=ImageFont.load_default(size=22))
    return panel


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem in STEMS:
        rgb = np.asarray(Image.open(VAL / f"{stem}_img.jpg").convert("RGB"))
        maps = {
            "GT": labels(VAL / f"{stem}_masks_organoid.png"),
            "OrganoID": labels(OID / f"{stem}_img_id-labeled.tif"),
            "OrganoSeg2": labels(OS2 / f"{stem}_organoseg2_labels.tif"),
            "CellposeSAM": labels(CP / f"{stem}_img_cpsam_mask_16bit.png"),
        }
        panels = [titled(Image.fromarray(rgb), "Original")]
        for method, lab in maps.items():
            rendered = Image.fromarray(overlay(rgb, lab))
            rendered.save(OUT / f"{stem}_{method.lower()}_overlay.png")
            panels.append(titled(rendered, f"{method} (n={int(lab.max())})"))
        width = sum(p.width for p in panels)
        height = max(p.height for p in panels)
        sheet = Image.new("RGB", (width, height), (238, 238, 242))
        x = 0
        for panel in panels:
            sheet.paste(panel, (x, 0)); x += panel.width
        sheet.save(OUT / f"{stem}_comparison.png", quality=95)
        rows.append(sheet.resize((1500, round(sheet.height * 1500 / sheet.width)), Image.Resampling.LANCZOS))
    contact = Image.new("RGB", (1500, sum(x.height for x in rows) + 20 * (len(rows)-1)), "white")
    y = 0
    for row in rows:
        contact.paste(row, (0, y)); y += row.height + 20
    contact.save(OUT / "all_three_comparison.png", quality=95)


if __name__ == "__main__":
    main()
