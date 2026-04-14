#!/usr/bin/env bash
# Full 30-run CV sweep: 2 models × 5 folds × 3 seeds.
# Run from repo root.
set -euo pipefail

MODELS=("compositional_mlp" "set_transformer")
FOLDS=(0 1 2 3 4)
SEEDS=(42 123 2024)

for model in "${MODELS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "=== ${model}  fold=${fold}  seed=${seed} ==="
      python scripts/train.py \
        --config "configs/${model}.yaml" \
        --fold "${fold}" \
        --seed "${seed}" \
        --output-dir "artifacts/${model}"
    done
  done
done

echo
echo "sweep done — now run:"
echo "  python scripts/aggregate_cv.py --artifacts-dir artifacts"
