# BioAgentUtils

Utilities for local microscopy data processing.

## LIF to JPEG Export

`lif_to_jpeg.py` exports Leica `.lif` images to JPEG files.
By default it exports all time points and all Z planes for each selected image/channel.

Install dependency:

```bash
pip install readlif
```

Example:

```bash
python BioAgentUtils/lif_to_jpeg.py \
  "Yichao-data-1/P11N&N39_Rep_DF.lif" \
  --output-dir "Yichao-data-1/P11N&N39_Rep_DF_jpeg"
```

Optional flags:

- `--image-indices 0 1` export specific image indices.
- `--channels 0` export specific channel(s).
- `--z-indices 0 1 2` export specific Z planes.
- `--time-indices 0 5 10` export specific time points.
- `--first-plane-only` export only `t=0, z=0` (legacy single-plane mode).
- `--overwrite` replace existing JPEGs.
- `--quality 95` set JPEG quality.
