# Accuracy Optimization Guide

## Reality Check First

- Reaching **80%** on strict 4-class EEG emotion recognition is usually difficult.
- You can still improve your current results with stronger protocol and ensembling.
- Best practice is to optimize aggressively while reporting limitations honestly.

## Fast Baseline

Train with default settings:

```bash
python3 run.py train
```

Evaluate single-model pair:

```bash
python3 run.py evaluate
```

## Fold Ensemble (Recommended)

After training, evaluate top fold ensembles:

```bash
python3 run.py evaluate-ensemble --top-k 3 --split cross_subject
```

Try different `top-k` values:

```bash
python3 run.py evaluate-ensemble --top-k 2 --split cross_subject
python3 run.py evaluate-ensemble --top-k 4 --split cross_subject
python3 run.py evaluate-ensemble --top-k 5 --split cross_subject
```

Pick the best by `quadrant_from_binary.balanced_accuracy` and `macro_f1`.

## Automated Experiment Search

Run the full experiment plan:

```bash
python3 run.py experiments --plan training/experiments_plan.json --eval-split cross_subject
```

This command trains multiple configurations, ensemble-evaluates each one, and writes:

- `artifacts/experiment_leaderboard.json`

Use the top leaderboard run as your final reported model path.

## Practical Tuning Tips

1. Prioritize `group_kfold_subject` and `cross_subject` over easy splits.
2. Compare `balanced`, `robust`, and `aggressive` presets.
3. Keep `epochs` high enough for scheduler + early stopping to work.
4. Use ensemble scoring, not single fold scoring, for final selection.
5. Report both accuracy and class-balanced metrics.

## Current Best Reproducible Run

```bash
python3 training/train_eegnet_binary.py \
  --task valence --split cross_trial --epochs 80 --batch-size 64 \
  --preset custom --loss cross_entropy --no-class-weights \
  --lr 1e-3 --weight-decay 1e-4 --patience 12 \
  --seed 999 --output-dir artifacts/acc_valence_s999

python3 training/train_eegnet_binary.py \
  --task arousal --split cross_trial --epochs 80 --batch-size 64 \
  --preset custom --loss cross_entropy --no-class-weights \
  --lr 1e-3 --weight-decay 1e-4 --patience 12 \
  --seed 999 --output-dir artifacts/acc_arousal_s999
```

Window-level (`cross_trial`, seed 999):
- Valence: accuracy `68.61%`, balanced accuracy `53.38%`, macro F1 `0.5345`
- Arousal: accuracy `57.17%`, balanced accuracy `58.38%`, macro F1 `0.5534`

Trial-level aggregation:

```bash
python3 run.py evaluate-trial \
  --valence-model artifacts/acc_valence_s999/valence/cross_trial/eegnet_valence_best.pth \
  --arousal-model artifacts/acc_arousal_s999/arousal/cross_trial/eegnet_arousal_best.pth \
  --valence-scaler artifacts/acc_valence_s999/valence/cross_trial/valence_scaler.pkl \
  --arousal-scaler artifacts/acc_arousal_s999/arousal/cross_trial/arousal_scaler.pkl \
  --split cross_trial \
  --aggregation vote \
  --report-out artifacts/trial_eval_seed999_vote.json
```

Observed trial-level results:
- `mean_prob`: valence `72.83%`, arousal `61.96%`, quadrant `44.57%`
- `vote`: valence `75.00%`, arousal `61.96%`, quadrant `44.57%`

## What to Report in College Submission

- Best run + evaluation protocol.
- Mean and std where available.
- Confusion matrix behavior (which classes are hardest).
- Honest limitation: strict 4-class EEG task remains challenging.

## Future Work (Important)

1. LOSO-first ranking and confidence intervals.
2. Subject adaptation or domain generalization methods.
3. Better artifact handling and richer frequency features.
4. Probability calibration (temperature scaling).
5. Multi-stage approach: optimize binary tasks, then map to quadrants with calibration.
