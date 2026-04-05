# Yichao Dataset Structure for Brightfield-to-Fluorescence Pix2pix

This note documents the usable paired data in:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-1`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-2`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-3`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-4`

The intended supervised task is:

- input: `c0` brightfield
- target: `c1` fluorescence

The repository already assumes this mapping in:

- `BioAgentUtils/prepare_yichao_pairs_to_npy.py`
- `BioAgentUtils/train_pix2pix_yichao.py`


## Important Split Note

The original dynamic source file was:

- `Data-Yichao-3/N39_TriRep_DF.lif`

That original Leica LIF was **not** physically rewritten into two new LIF files. Instead, the extracted JPEG exports were reorganized as:

- `Data-Yichao-3`: Day 2 + Day 3 extracted subset
- `Data-Yichao-4`: Day 4 extracted subset

For convenience, the same original raw source files are mirrored in both folders:

- `N39_TriRep_DF.lif`
- `N39_TriRep_DF_2.lif`

So:

- the JPEG exports are split
- the raw `.lif` file itself is still the same original source
- `Data-Yichao-3` and `Data-Yichao-4` must not be double-counted as independent raw LIF acquisitions


## Core Unit of Usable Data

For pix2pix, the useful supervised unit is:

- one paired 2D image plane at fixed `series/position`, `z`, and `t`
- brightfield `c0` paired with fluorescence `c1`

For each LIF series:

- usable paired samples = `z_count * time_count`
- total exported JPEG files = `z_count * time_count * 2`

because each `(t, z)` plane is exported for both channels.


## Important Overlap Warning

`Data-Yichao-1/P11N&N39_Rep_DF.lif` is not an independent evaluation-only dataset.

Its 5 static MUC2 samples are byte-identical to the first 5 static samples inside:

- `Data-Yichao-2/P11N&N39_Rep_DF.lif`

So:

- train on `Data-Yichao-2` and test on `Data-Yichao-1` is a leaky split
- the current repo defaults are fine for smoke tests, but not for a valid final benchmark


## File-Level Summary

### Data-Yichao-1

LIF file:

- `Data-Yichao-1/P11N&N39_Rep_DF.lif`

Contents:

- 5 series
- all are static single-plane samples
- names:
  - `N39_TriRep_MUC2_mNeon_20X_1`
  - `N39_TriRep_MUC2_mNeon_20X_2`
  - `N39_TriRep_MUC2_mNeon_20X_3`
  - `N39_TriReP_MUC2_mNeon_20X_4`
  - `N39_TriRep_MUC2_mNeon_20X_5`

Acquisition structure:

- XY size: `1024 x 1024`
- channels: `2`
- z-depth per sample: `1`
- timepoints per sample: `1`
- usable paired samples per series: `1`
- total usable paired samples: `5`
- approximate pixel size: `0.303 um/pixel`

Interpretation:

- 5 distinct static fields of view
- no z-stack
- no time-lapse
- all 5 are duplicated in `Data-Yichao-2`


### Data-Yichao-2

LIF file:

- `Data-Yichao-2/P11N&N39_Rep_DF.lif`

Contents:

- 11 series total
- 5 static MUC2 series, duplicated from `Data-Yichao-1`
- 6 dynamic Day-2 positions

Series breakdown:

| Series group | Count | XY | Z | T | Usable pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static MUC2 | 5 | 1024x1024 | 1 | 1 | 5 |
| `N39_TriRep_DF_D2/Position001` | 1 | 1024x1024 | 11 | 16 | 176 |
| `N39_TriRep_DF_D2/Position002` | 1 | 1024x1024 | 9 | 16 | 144 |
| `N39_TriRep_DF_D2/Position003` | 1 | 1024x1024 | 11 | 16 | 176 |
| `N39_TriRep_DF_D2/Position004` | 1 | 1024x1024 | 9 | 16 | 144 |
| `N39_TriRep_DF_D2/Position005` | 1 | 1024x1024 | 9 | 16 | 144 |
| `N39_TriRep_DF_D2/Position006` | 1 | 1024x1024 | 11 | 16 | 176 |

Acquisition structure for the dynamic Day-2 part:

- channels: `2`
- approximate pixel size: `0.568 um/pixel`
- z step magnitude: about `2.469 um`
- time step: about `3622 s` or `60.4 min`

Totals:

- total usable paired samples including duplicated static images: `965`
- unique dynamic paired samples beyond `Data-Yichao-1`: `960`

Interpretation:

- this is a mixed file
- it contains both the static MUC2 images and the dynamic Day-2 stacks
- for model development, the unique part is mainly the 6 Day-2 positions


### Data-Yichao-3

Current folder role:

- extracted subset from the original `N39_TriRep_DF.lif`
- now holds Day 2 and Day 3 only

Files present:

- `Data-Yichao-3/N39_TriRep_DF.lif`
- `Data-Yichao-3/N39_TriRep_DF_2.lif`
- `Data-Yichao-3/N39_TriRep_DF_jpeg_all`
- `Data-Yichao-3/N39_TriRep_DF_jpeg_all_by_object`

`N39_TriRep_DF_2.lif`:

- empty file
- size `0` bytes
- unusable

Current extracted subset:

| Day | Positions | Usable pairs | Total JPEG files |
| --- | ---: | ---: | ---: |
| Day 2 | 3 | 2296 | 4592 |
| Day 3 | 5 | 4459 | 8918 |

Current totals in `Data-Yichao-3`:

- positions: `8`
- usable paired samples: `6755`
- total JPEG files in `N39_TriRep_DF_jpeg_all`: `13510`

Interpretation:

- this is the Day 2 + Day 3 extracted monitoring subset
- it is the larger side of the split
- it comes from the same original raw LIF source that also underlies `Data-Yichao-4`


### Data-Yichao-4

Current folder role:

- extracted subset from the original `N39_TriRep_DF.lif`
- now holds Day 4 only

Files present:

- `Data-Yichao-4/N39_TriRep_DF.lif`
- `Data-Yichao-4/N39_TriRep_DF_2.lif`
- `Data-Yichao-4/N39_TriRep_DF_jpeg_all`
- `Data-Yichao-4/N39_TriRep_DF_jpeg_all_by_object`

Current extracted subset:

| Day | Positions | Usable pairs | Total JPEG files |
| --- | ---: | ---: | ---: |
| Day 4 | 5 | 3264 | 6528 |

Current totals in `Data-Yichao-4`:

- positions: `5`
- usable paired samples: `3264`
- total JPEG files in `N39_TriRep_DF_jpeg_all`: `6528`

Interpretation:

- this is the Day 4 extracted monitoring subset
- it is useful as a held-out day-shift set or as an additional training block
- it comes from the same original raw LIF source that also underlies `Data-Yichao-3`


### Original Dynamic Source Behind Data-Yichao-3 and Data-Yichao-4

The original unsplit dynamic source `N39_TriRep_DF.lif` contained 13 dynamic series total:

- 3 positions on Day 2
- 5 positions on Day 3
- 5 positions on Day 4

Original source breakdown:

| Series | XY | Z | T | Usable pairs |
| --- | ---: | ---: | ---: | ---: |
| `Experiment_1 Day_2/Position001` | 512x512 | 26 | 41 | 1066 |
| `Experiment_1 Day_2/Position002` | 512x512 | 10 | 41 | 410 |
| `Experiment_1 Day_2/Position003` | 512x512 | 20 | 41 | 820 |
| `Experiment_1 Day_3/Position001` | 512x512 | 16 | 49 | 784 |
| `Experiment_1 Day_3/Position002` | 512x512 | 24 | 49 | 1176 |
| `Experiment_1 Day_3/Position003` | 512x512 | 8 | 49 | 392 |
| `Experiment_1 Day_3/Position004` | 512x512 | 11 | 49 | 539 |
| `Experiment_1 Day_3/Position005` | 512x512 | 32 | 49 | 1568 |
| `Experiment_1 Day_4/Position001` | 512x512 | 18 | 32 | 576 |
| `Experiment_1 Day_4/Position002` | 512x512 | 25 | 32 | 800 |
| `Experiment_1 Day_4/Position003` | 512x512 | 18 | 32 | 576 |
| `Experiment_1 Day_4/Position004` | 512x512 | 23 | 32 | 736 |
| `Experiment_1 Day_4/Position005` | 512x512 | 18 | 32 | 576 |

Original day-level totals:

| Day | Positions | Usable pairs |
| --- | ---: | ---: |
| Day 2 | 3 | 2296 |
| Day 3 | 5 | 4459 |
| Day 4 | 5 | 3264 |

Acquisition structure:

- channels: `2`
- approximate pixel size: `1.137 um/pixel`
- Day 2 z step: about `2.000 um`
- Day 3 z step: about `1.608 um`
- Day 4 z step: about `2.000 um`
- time step: about `1800-1805 s` or about `30 min`

Original total usable paired samples from this single source:

- `10019`


## What Each Folder Means

### `N39_TriRep_DF_jpeg_all`

This is the flat export directory.

Each file is one 2D plane from one position at one timepoint and one z-depth.

Example:

```text
00_Experiment_1_Day_3_Position002_t017_z006_c1.jpg
```

Meaning:

- `00`: series index inside the LIF export
- `Experiment_1_Day_3_Position002`: one monitored position / field of view
- `t017`: timepoint 17
- `z006`: z-plane 6
- `c1`: channel 1, treated here as fluorescence


### `N39_TriRep_DF_jpeg_all_by_object`

This is the same export regrouped by LIF series name.

Here, “object” is better read as:

- one imaging position
- one fixed field of view
- one monitored sample stack

So one folder under `*_by_object` corresponds to one monitored position.


### `Data-Yichao-2/P11N&N39_Rep_DF_jpeg_all_by_object`

This folder mixes two different data types:

- dynamic Day-2 monitoring positions: `N39_TriRep_DF_D2_Position001..006`
- static single-image MUC2 samples: `N39_TriRep_MUC2_mNeon_20X_1..5`

So not every folder there is a time-lapse sequence.


### `Data-Yichao-1/P11N&N39_Rep_DF_jpeg`

This is a legacy export for the 5 static MUC2 samples.

It predates the fuller `t000_z000` naming style and is effectively:

- one c0/c1 pair per sample
- no time dimension
- no z-stack

The cleaner equivalent export is:

- `Data-Yichao-1/P11N&N39_Rep_DF_jpeg_all`


## Unique Usable Data Across All Yichao LIFs

If duplicated content is removed:

- `Data-Yichao-1`: `5` paired samples, but all duplicated in `Data-Yichao-2`
- `Data-Yichao-2`: `960` unique dynamic paired samples
- original `N39_TriRep_DF.lif` source: `10019` unique dynamic paired samples

Because `Data-Yichao-3` and `Data-Yichao-4` are only a folder-level split of that same original source:

- `Data-Yichao-3`: `6755` extracted pairs
- `Data-Yichao-4`: `3264` extracted pairs
- `6755 + 3264 = 10019`

Total unique paired planes across the underlying non-empty source LIF files:

- `10984`

Unique series/positions across the underlying sources:

- 5 static MUC2 series
- 6 dynamic Yichao-2 Day-2 positions
- 13 dynamic positions from the original `N39_TriRep_DF.lif`
- total unique series/positions: `24`


## What "Replication" Means Here

The filenames contain strings like `TriRep`, but the LIF metadata examined here does not cleanly encode biological replicate labels.

Safe interpretation:

- one LIF series is one field of view / position / sample stack
- positions within the same day should be treated as distinct samples
- z-planes are repeated observations across depth
- timepoints are repeated observations across time
- do not assume that `TriRep` provides a machine-readable replicate grouping for evaluation


## Implications for Pix2pix Training

Recommended supervised sample definition:

- one paired plane at fixed `position`, `t`, and `z`

Recommended split rule:

- split by position, not by random planes

Why:

- adjacent z-slices are strongly correlated
- adjacent timepoints are strongly correlated
- random plane splitting would leak almost-identical data between train and validation/test


### Practical baseline

The cleanest current baseline is:

- train first on `Data-Yichao-3`
- use `Data-Yichao-4` as a held-out day-shift evaluation set, or as a second-stage training extension

This now has a clearer meaning than the old unsplit layout:

- `Data-Yichao-3`: Day 2 + Day 3
- `Data-Yichao-4`: Day 4


### Should Yichao-2 be mixed with Data-Yichao-3/4?

Not immediately.

Yichao-2 dynamic data and the `N39_TriRep_DF` source are at different spatial sampling:

- Yichao-2 dynamic: `1024 x 1024`, about `0.568 um/pixel`
- original `N39_TriRep_DF` source: `512 x 512`, about `1.137 um/pixel`

That means:

- the physical field of view and effective scale differ
- mixing them without normalization adds a domain shift

Best practice:

- first build a clean baseline on the split `Data-Yichao-3` / `Data-Yichao-4`
- then add Yichao-2 dynamic positions only after deciding how to normalize physical scale


## Bottom Line

If the goal is to learn fluorescence from brightfield with pix2pix:

- `Data-Yichao-1` is static only and not an independent test set
- `Data-Yichao-2` contains useful dynamic Day-2 data plus duplicated static content
- `Data-Yichao-3` now holds the Day 2 + Day 3 extracted subset from the original `N39_TriRep_DF.lif`
- `Data-Yichao-4` now holds the Day 4 extracted subset from that same original source
- the raw `N39_TriRep_DF.lif` mirrored in both folders is still one shared source file, not two separate acquisitions
