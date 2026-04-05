# Yichao Dataset Structure for Brightfield-to-Fluorescence Pix2pix

This note documents the usable paired data in:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-1`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-2`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-3`

The intended supervised task is:

- input: `c0` brightfield
- target: `c1` fluorescence

The repository already assumes this mapping in:

- `BioAgentUtils/prepare_yichao_pairs_to_npy.py`
- `BioAgentUtils/train_pix2pix_yichao.py`


## Core Unit of Usable Data

For pix2pix, the useful supervised unit is:

- one paired 2D image plane at fixed `series/position`, `z`, and `t`
- brightfield `c0` paired with fluorescence `c1`

For each LIF series:

- number of usable paired samples = `z_count * time_count`
- total exported JPEG files = `z_count * time_count * 2` because there are 2 channels


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

LIF files:

- `Data-Yichao-3/N39_TriRep_DF.lif`
- `Data-Yichao-3/N39_TriRep_DF_2.lif`

`N39_TriRep_DF_2.lif`:

- empty file
- size `0` bytes
- unusable

`N39_TriRep_DF.lif` contents:

- 13 dynamic series total
- 3 positions on Day 2
- 5 positions on Day 3
- 5 positions on Day 4

Series breakdown:

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

Day-level totals:

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

Totals:

- total usable paired samples: `10019`

Interpretation:

- this is the largest and most useful dataset for supervised brightfield-to-fluorescence learning
- it is internally structured as time-lapse z-stacks across multiple days and positions


## Unique Usable Data Across All Yichao LIFs

If duplicated content is removed:

- `Data-Yichao-1`: `5` paired samples, but all duplicated in `Data-Yichao-2`
- `Data-Yichao-2`: `960` unique dynamic paired samples
- `Data-Yichao-3`: `10019` unique dynamic paired samples

Total unique paired planes across non-empty LIF files:

- `10984`

Unique series/positions:

- 5 static MUC2 series
- 6 dynamic Yichao-2 Day-2 positions
- 13 dynamic Yichao-3 positions
- total unique series/positions: `24`


## What "Replication" Means Here

The filenames contain strings like `TriRep`, but the LIF metadata examined here does not cleanly encode biological replicate labels.

Safe interpretation:

- one LIF series is one field of view / position / sample stack
- positions within the same day should be treated as distinct samples
- do not assume that `TriRep` provides a machine-readable replicate grouping for evaluation


## Implications for Pix2pix Training

### Recommended first training setup

Use `Data-Yichao-3` first.

Reasons:

- it is the largest dataset
- it has the richest z and time structure
- it avoids the Y1/Y2 duplication issue
- it is internally consistent in image size: `512 x 512`

Recommended supervised sample definition:

- one paired plane at fixed `position`, `t`, and `z`

Recommended split rule:

- split by position, not by random planes

Why:

- adjacent z-slices are strongly correlated
- adjacent timepoints are strongly correlated
- random plane splitting would leak almost-identical data between train and validation/test


### Recommended data split strategy

For a valid benchmark:

- keep all planes from a given position in exactly one split
- ideally keep positions from each day represented across train/val/test

Example:

- train: most positions from Day 2, Day 3, Day 4
- val: one held-out position
- test: one or more different held-out positions

A stricter version is:

- leave out entire positions for testing
- optionally leave out one full day for domain-shift testing


### Should Yichao-2 be mixed with Yichao-3?

Not immediately.

Yichao-2 dynamic data and Yichao-3 dynamic data are at different spatial sampling:

- Yichao-2 dynamic: `1024 x 1024`, about `0.568 um/pixel`
- Yichao-3 dynamic: `512 x 512`, about `1.137 um/pixel`

This means:

- the physical field of view and effective scale differ
- mixing them without normalization adds a domain shift

Best practice:

- first train a clean baseline on `Data-Yichao-3` only
- then optionally add Yichao-2 dynamic positions after resampling to a common physical scale


### How the current repo scripts use the data

`BioAgentUtils/prepare_yichao_pairs_to_npy.py`

- scans JPEG exports and pairs `c0` with `c1`
- default train root is `Data-Yichao-2`
- default test root is `Data-Yichao-1`
- optional "first pair per top-level object" mode exists for quick smoke tests

`BioAgentUtils/train_pix2pix_yichao.py`

- also pairs `c0` with `c1`
- default train root is `Data-Yichao-2`
- default verify root is `Data-Yichao-1`
- can crop large images into tiles
- can reduce data to the first pair per top-level object

These defaults are acceptable for quick debugging, but they should not be used as the final scientific split because Y1 is duplicated inside Y2.


## Practical Recommendation

For a real pix2pix experiment:

1. Build a manifest from `Data-Yichao-3` with columns like:
   - `lif_file`
   - `series_name`
   - `day`
   - `position`
   - `z`
   - `t`
   - `input_path`
   - `target_path`
   - `split`
2. Split at the position level.
3. Train the first model on Y3 only.
4. Add Y2 dynamic data only after deciding how to normalize spatial scale.
5. Do not use `Data-Yichao-1` as a standalone held-out benchmark against training on `Data-Yichao-2`.


## Bottom Line

If the goal is to learn fluorescence from brightfield with pix2pix:

- the best current training asset is `Data-Yichao-3/N39_TriRep_DF.lif`
- `Data-Yichao-2` is useful, but only its dynamic Day-2 positions are unique
- `Data-Yichao-1` is mainly useful as documentation of the static MUC2 subset, not as an independent test set
