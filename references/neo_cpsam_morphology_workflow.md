# NEO CellposeSAM morphology workflow

## Scope

This workflow analyzes selected JPEG images from `datasets/07_NEO`. It uses the
pretrained `cpsam` model in Cellpose 4.0.5 and calls `model.eval(image)` without
overriding inference parameters. GPU selection changes only the execution device.

## Outputs

For every image the runner writes a 16-bit instance mask, a colored instance map,
an overlay, and image-level metrics. `objects.csv` contains one row per instance;
`results.csv` contains image-level aggregates; `results.json` and `report.md`
record provenance and interpretation notes.

Lengths and areas are reported in pixels and pixels squared. They must not be
reported as physical units unless a valid pixel calibration is added later.

## Morphology definitions

The core OrganoID-compatible profile consists of area, perimeter, circularity,
solidity, and eccentricity:

- `area_px`: foreground pixel count.
- `perimeter_px`: closed external-contour arc length.
- `circularity = 4 pi area / perimeter^2`.
- `solidity = area / convex_hull_area`.
- `eccentricity = sqrt(1 - minor_axis^2 / major_axis^2)`.

Run-level summaries include instance count and total/mean projected area. This
matches the area-based longitudinal analysis in OrganoSeg2 Figure 5. No volume is
inferred from a 2D mask. Without spatial calibration, area is reported in px²;
with a validated calibration it may be converted to µm².

The extended OrganoSeg2-style region profile adds centroid, bounding box, extent,
major/minor axes, aspect ratio, equivalent diameter, maximum Feret diameter,
orientation, intensity mean/standard deviation, and image-border contact. The
workflow also exports mean boundary radius, radius variation, non-circularity,
and ellipse-relative non-smoothness for comparison with published organoid shape
analysis workflows.

Border-touching objects are retained but explicitly flagged. Overlays should be
reviewed before biological interpretation because pretrained segmentation is not
a substitute for dataset-specific ground-truth validation.

## Method references

- Matthews et al. (2022), OrganoID: https://doi.org/10.1371/journal.pcbi.1010584
- Wells et al., OrganoSeg2: https://doi.org/10.1038/s41598-026-37526-7
- OrganoSeg2 source: https://github.com/JanesLab/OrganoSeg2
- Cellpose settings: https://cellpose.readthedocs.io/en/latest/settings.html
