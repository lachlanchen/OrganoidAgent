# Yichao Instance Pairs 256

This folder contains dedicated scripts to turn the finished segmented Yichao instance database into a resized paired dataset for brightfield-to-fluorescence learning.

Source dataset:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs/database/instance_pairs.sqlite`

Default output dataset:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs_resized_256`

The prepared dataset layout is:

- `brightfield_256/<dataset>/<pair_index>.png`
- `fluorescence_256/<dataset>/<pair_index>.png`
- `metadata/pairs_manifest.csv`
- `metadata/summary.json`
- `preview/random_pairs_*.png`

Notes:

- Brightfield is `c1`.
- Fluorescence is `c0`.
- The resize target is `256x256`.
- Pair indices are global, zero-padded, and consistent across brightfield and fluorescence.
- Original provenance is preserved in the CSV manifest.

Example commands:

```bash
python /home/lachlan/ProjectsLFS/OrganoidAgent/differentiation_prediction/yichao_instance_pairs_256/prepare_dataset.py
python /home/lachlan/ProjectsLFS/OrganoidAgent/differentiation_prediction/yichao_instance_pairs_256/preview_random_pairs.py
```
