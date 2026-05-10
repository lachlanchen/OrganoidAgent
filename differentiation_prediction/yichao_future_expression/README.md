# Yichao Future Expression Pipeline

This folder trains real models from the projected Yichao organoid instance database:

- Input database: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_projected_instance_pairs/database/projected_instance_pairs.sqlite`
- Output root: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_future_expression`
- Brightfield channel: projected `c1`
- Fluorescence channel: projected `c0`
- Crop policy: complete/non-edge projected organoid instances only, then resize square crops to 256x256.

## Stages

1. `build_projected_dataset.py`
   Materializes resized projected brightfield/fluorescence/mask crops, computes fluorescence labels, links time-series instances into approximate tracks, and writes:
   - `manifests/projected_instances_manifest.csv`
   - `manifests/tracks.csv`
   - `manifests/future_samples.csv`
   - `future_expression.sqlite`

2. `train_b2f.py`
   Trains a real multitask U-Net for same-time brightfield-to-fluorescence feasibility. It saves checkpoints, metrics, prediction panels, and test metrics under `stage1_b2f/`.

3. `analyze_features.py`
   Trains a small explicit-feature model and permutation-importance analysis for morphology features such as area, diameter, circularity, support ratio, edge strength, and time index. It saves feature importance plots under `stage1_feature_analysis/`.

4. `train_future_expression.py`
   Trains a sequence model for early prediction: `B(D1...Dk) -> future fluorescence`. By default the dataset builder excludes prefixes that are already fluorescence-positive, so this stage tests whether pre-expression brightfield morphology predicts later expression.

## Run

Start or resume the full GPU run in tmux:

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
bash differentiation_prediction/yichao_future_expression/resume_future_expression_tmux.sh
```

Attach:

```bash
tmux attach -t yichao_future_expression
```

Monitor logs:

```bash
tail -f /home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_future_expression/logs/full_pipeline_*.log
```

## Main Artifacts

- B2F predictions: `analysis-outputs/yichao_future_expression/stage1_b2f/predictions/`
- B2F metrics: `analysis-outputs/yichao_future_expression/stage1_b2f/test_metrics.json`
- Feature explanations: `analysis-outputs/yichao_future_expression/stage1_feature_analysis/feature_importance.csv`
- Future prediction metrics: `analysis-outputs/yichao_future_expression/stage2_future_expression/test_metrics.json`
- Future prediction table: `analysis-outputs/yichao_future_expression/stage2_future_expression/test_predictions.csv`
- Future feature ablation: `analysis-outputs/yichao_future_expression/stage2_future_expression/feature_ablation_importance.csv`
