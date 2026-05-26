#!/usr/bin/env bash
set -euo pipefail

# One-command sweep focused on improving binary accuracy quickly.
# Defaults target cross-trial (easier split) for higher raw accuracy.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SPLIT="${SPLIT:-cross_trial}"
SEEDS="${SEEDS:-42 77 123 999 2026}"
TOP_K="${TOP_K:-3}"
OUT_ROOT="${OUT_ROOT:-artifacts/fast_sweep}"

EPOCHS_MAIN="${EPOCHS_MAIN:-80}"
EPOCHS_AROUSAL="${EPOCHS_AROUSAL:-110}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_AROUSAL_BOOST="${RUN_AROUSAL_BOOST:-1}"

echo "Project root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "Split: $SPLIT"
echo "Seeds: $SEEDS"
echo "Output root: $OUT_ROOT"
echo

mkdir -p "$OUT_ROOT"

for seed in $SEEDS; do
  out_dir="$OUT_ROOT/seed_${seed}"
  echo "=== Training main run for seed=${seed} -> ${out_dir} ==="
  "$PYTHON_BIN" training/train_eegnet_binary.py \
    --task both \
    --split "$SPLIT" \
    --epochs "$EPOCHS_MAIN" \
    --batch-size "$BATCH_SIZE" \
    --preset custom \
    --loss cross_entropy \
    --no-class-weights \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --patience 12 \
    --seed "$seed" \
    --output-dir "$out_dir"
done

if [[ "$RUN_AROUSAL_BOOST" == "1" ]]; then
  for seed in $SEEDS; do
    out_dir="$OUT_ROOT/arousal_boost_seed_${seed}"
    echo "=== Training arousal boost run for seed=${seed} -> ${out_dir} ==="
    "$PYTHON_BIN" training/train_eegnet_binary.py \
      --task arousal \
      --split "$SPLIT" \
      --epochs "$EPOCHS_AROUSAL" \
      --batch-size "$BATCH_SIZE" \
      --preset custom \
      --loss focal \
      --focal-gamma 1.5 \
      --lr 5e-4 \
      --weight-decay 5e-4 \
      --patience 18 \
      --seed "$seed" \
      --output-dir "$out_dir"
  done
fi

TOP_ENV="$OUT_ROOT/topk_paths.env"

echo "=== Ranking reports and selecting top-${TOP_K} checkpoints per task ==="
"$PYTHON_BIN" - "$OUT_ROOT" "$SPLIT" "$TOP_K" "$TOP_ENV" <<'PY'
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
split = sys.argv[2]
top_k = int(sys.argv[3])
env_path = Path(sys.argv[4])

pattern = f"*_{split}_report.json"
report_paths = sorted(out_root.rglob(pattern))
if not report_paths:
    raise SystemExit(f"No reports found under {out_root} matching {pattern}")

rows = []
for report_path in report_paths:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    task = payload.get("config", {}).get("resolved_task")
    if task not in {"valence", "arousal"}:
        continue
    summary = payload.get("summary", {})
    folds = payload.get("folds", [])
    if not folds:
        continue
    fold = folds[0]
    rows.append(
        {
            "task": task,
            "report_path": str(report_path),
            "accuracy": float(summary.get("accuracy_mean", 0.0)),
            "balanced_accuracy": float(summary.get("balanced_accuracy_mean", 0.0)),
            "macro_f1": float(summary.get("macro_f1_mean", 0.0)),
            "checkpoint_path": str(fold.get("checkpoint_path", "")),
            "scaler_path": str(fold.get("scaler_path", "")),
        }
    )

if not rows:
    raise SystemExit("No usable report rows found.")

for task in ("valence", "arousal"):
    ranked = sorted(
        [r for r in rows if r["task"] == task],
        key=lambda r: r["accuracy"],
        reverse=True,
    )
    print(f"\nTop {len(ranked)} {task} runs by accuracy:")
    for idx, row in enumerate(ranked, start=1):
        print(
            f"  {idx:02d}. acc={row['accuracy']:.4f} "
            f"bal_acc={row['balanced_accuracy']:.4f} "
            f"f1={row['macro_f1']:.4f} "
            f"path={row['report_path']}"
        )

def csv_for(task: str, field: str) -> str:
    ranked = sorted(
        [r for r in rows if r["task"] == task],
        key=lambda r: r["accuracy"],
        reverse=True,
    )[: max(1, top_k)]
    vals = [r[field] for r in ranked if r[field]]
    if not vals:
        raise SystemExit(f"No values found for task={task}, field={field}")
    return ",".join(vals)

valence_models = csv_for("valence", "checkpoint_path")
valence_scalers = csv_for("valence", "scaler_path")
arousal_models = csv_for("arousal", "checkpoint_path")
arousal_scalers = csv_for("arousal", "scaler_path")

env_path.write_text(
    "\n".join(
        [
            f'VALENCE_MODELS_CSV="{valence_models}"',
            f'VALENCE_SCALERS_CSV="{valence_scalers}"',
            f'AROUSAL_MODELS_CSV="{arousal_models}"',
            f'AROUSAL_SCALERS_CSV="{arousal_scalers}"',
            "",
        ]
    ),
    encoding="utf-8",
)
print(f"\nWrote top-k CSV paths to: {env_path}")
PY

source "$TOP_ENV"

ENSEMBLE_OUT="$OUT_ROOT/ensemble_top${TOP_K}_${SPLIT}.json"
echo "=== Running top-${TOP_K} seed ensemble evaluation ==="
"$PYTHON_BIN" training/evaluate_binary_pair_ensemble.py \
  --split "$SPLIT" \
  --valence-models "$VALENCE_MODELS_CSV" \
  --valence-scalers "$VALENCE_SCALERS_CSV" \
  --arousal-models "$AROUSAL_MODELS_CSV" \
  --arousal-scalers "$AROUSAL_SCALERS_CSV" \
  > "$ENSEMBLE_OUT"

echo "Saved ensemble report: $ENSEMBLE_OUT"
echo
echo "Done. Next step: open the report and use the best run for submission."
