# Yichao B2F Strong Training Strategy

## Current diagnosis

The previous future-expression prediction is not reliable yet. The first problem to fix is not the future model; it is to make same-time brightfield-to-fluorescence (B2F) reconstruction strong and stable.

The old B2F run had several avoidable weaknesses:

- It trained only 40 epochs.
- It used 256 x 256 crops only, which can compress thin peripheral fluorescent cell structures.
- It selected `best_model.pt` mainly by validation expression AUROC. That is unstable because expression examples are rare.
- Validation/test splits are highly imbalanced. The test split has only 14 expression-labeled B2F crops in the previous run.
- The fluorescence target is sparse relative to background, so ordinary image losses are dominated by easy dark/background pixels.
- The future task inherits additional problems: approximate track linking, rare future-expression examples, and uncertain target policy (`last_future` may not be the biologically best target).

## Correct priority

1. Train B2F properly first.
2. Use the B2F encoder as pretraining for future prediction.
3. Then train a joint temporal model for future expression.

Do not scale the future model before the B2F model and temporal labels are credible.

## Strong B2F changes

New code:

```text
/home/lachlan/ProjectsLFS/OrganoidAgent/differentiation_prediction/yichao_future_expression/train_b2f_strong.py
/home/lachlan/ProjectsLFS/OrganoidAgent/differentiation_prediction/yichao_future_expression/resume_strong_b2f_tmux.sh
```

Main changes:

- Reads original instance crops by default, not only the saved 256 crops.
- Defaults to 384 x 384 training.
- Uses a larger residual GroupNorm U-Net with squeeze-excitation.
- Uses pixel imbalance-aware training for sparse fluorescence.
- Uses continuous-intensity fluorescence loss instead of turning the target into a low-threshold binary mask.
- Uses Charbonnier intensity loss, SSIM-like structure loss, and Sobel edge loss.
- Uses scalar auxiliary targets for expression status, peak fluorescence, and total fluorescence.
- Supports image-level balanced sampling for rare expression examples.
- Selects the best checkpoint by reconstruction/signal quality, not AUROC alone.
- Saves `last_model.pt` every epoch and sparse periodic checkpoints every 20 epochs.
- Keeps only the most recent periodic checkpoints to avoid disk overload.

## Long-run command

The intended long run is:

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid
python -u -m differentiation_prediction.yichao_future_expression.train_b2f_strong \
  --output-root /home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_future_expression/stage1_b2f_strong_384_long \
  --image-size 384 \
  --path-mode original_crop \
  --epochs 1000 \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --base-channels 48 \
  --dropout 0.05 \
  --lr 1.5e-4 \
  --min-lr 1e-6 \
  --amp \
  --channels-last \
  --balanced-sampler \
  --eval-every 5 \
  --panel-every 20 \
  --save-every 20 \
  --keep-periodic 8 \
  --resume
```

The tmux helper starts this command:

```bash
/home/lachlan/ProjectsLFS/OrganoidAgent/differentiation_prediction/yichao_future_expression/resume_strong_b2f_tmux.sh
```

## May 11 progress check and correction

The first long 384 run reached epoch 47, then stopped with:

```text
TypeError: silu() keywords must be strings
```

The run also showed flat validation metrics from epoch 1 through epoch 45. The validation panels showed that the model was producing a broad green brightfield-like reconstruction rather than sparse true fluorescence. That means the issue was not simply insufficient epoch count.

The fix is:

- Remove the channels-last training path from the tmux command because it likely triggered the PyTorch/SiLU runtime path.
- Replace the strong model's SiLU activations with GELU.
- Stop using low-threshold binary fluorescence BCE as the main pixel signal. A low threshold made most organoid pixels count as fluorescent and encouraged broad green predictions.
- Use continuous target-intensity weighting: bright fluorescence pixels get larger weight, but dark/background pixels still pull the prediction down.
- Reduce scalar auxiliary weight so image reconstruction drives the run.

## What success should look like

B2F should not be judged only by image MAE. A mostly dark prediction can get deceptively good MAE. The useful checks are:

- `val_masked_mae`: fluorescence error inside the organoid mask.
- `val_signal_mae`: error on true fluorescent signal pixels.
- `val_signal_f1`: whether the model localizes fluorescent pixels.
- `val_peak_pearson`: whether image-level fluorescence intensity is ranked correctly.
- Saved validation panels every 20 epochs.
- Held-out `test_best.png` after training.

## Next future model after B2F works

After B2F is credible, use its encoder as initialization for a future model:

```text
B1...Bk -> temporal encoder -> future heads
```

Recommended future heads:

- Current B2F reconstruction: `Bk -> Fk`
- Short-horizon fluorescence: `Bk -> F(k+1)` and `Bk -> F(k+2)`
- Peak future fluorescence: `B1...Bk -> F_peak`
- Future expression probability
- First-expression timing

Use horizon-specific targets rather than only `last_future`. The model should identify the earliest day `k` where morphology becomes predictive, not just produce one final-frame guess.
