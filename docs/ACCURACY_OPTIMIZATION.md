# Accuracy Optimization Guide

## Reality Check First

- Reaching **80%** on strict 4-class EEG emotion recognition is usually difficult.
- You can still improve your current results with stronger protocol and ensembling.
- Best practice is to optimize aggressively while reporting limitations honestly.

## Fast Baseline

Train with default settings:

```bash
python run.py train
```

Evaluate single-model pair:

```bash
python run.py evaluate
```

## Fold Ensemble (Recommended)

After training, evaluate top fold ensembles:

```bash
python run.py evaluate-ensemble --top-k 3 --split cross_subject
```

Try different `top-k` values:

```bash
python run.py evaluate-ensemble --top-k 2 --split cross_subject
python run.py evaluate-ensemble --top-k 4 --split cross_subject
python run.py evaluate-ensemble --top-k 5 --split cross_subject
```

Pick the best by `quadrant_from_binary.balanced_accuracy` and `macro_f1`.

## Automated Experiment Search

Run the full experiment plan:

```bash
python run.py experiments --plan training/experiments_plan.json --eval-split cross_subject
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
