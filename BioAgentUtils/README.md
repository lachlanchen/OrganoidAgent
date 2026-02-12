# BioAgentUtils

Utilities for local microscopy data processing.

## LIF to JPEG Export

`lif_to_jpeg.py` exports Leica `.lif` images to JPEG files.

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
- `--overwrite` replace existing JPEGs.
- `--quality 95` set JPEG quality.
