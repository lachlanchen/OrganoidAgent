# Yichao Instance Pairs

This pipeline segments every Yichao brightfield frame, saves per-image segmentation intermediates, and exports paired instance crops for later dataset packing.

Brightfield / fluorescence mapping used here:

- `c1` = brightfield
- `c0` = fluorescence

Default output root:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs`

Main scripts:

- `run_yichao_instance_pair_extraction.py`
- `build_yichao_instance_pair_database.py`

Output layout:

- `images/<dataset>/<object>/<image_stem>/`
- `instances/<dataset>/<object>/<image_stem>/instance_0001/`
- `manifests/image_records.csv`
- `manifests/instance_records.csv`
- `manifests/summary.json`
- `database/instance_pairs.sqlite`

Each image folder contains:

- `brightfield_input.jpg`
- `fluorescence_reference.jpg`
- `debug_signal.png`
- `support.png`
- `multiscale_mask_16bit.png`
- `multiscale_instance_rgb.png`
- `multiscale_overlay_on_brightfield.png`
- `multiscale_overlay_on_fluorescence.png`
- `comparison_panel.png`
- `image_record.json`

Each instance folder contains:

- `brightfield_crop.png`
- `fluorescence_crop.png`
- `mask_crop.png`
- `overlay_on_brightfield_crop.png`
- `overlay_on_fluorescence_crop.png`
- `instance_rgb_crop.png`
- `instance_record.json`

The segmentation policy is intentionally simple for Yichao:

- run multiscale Cellpose on the brightfield image
- merge overlapping Cellpose masks across diameters
- only use the threshold/signal recovery branch as a fallback when Cellpose finds no candidates

Example:

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
bash analysis-tools/yichao_instance_pairs/run_yichao_instance_pair_extraction.sh --gpu true
bash analysis-tools/yichao_instance_pairs/build_yichao_instance_pair_database.sh
```
