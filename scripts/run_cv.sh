#!/usr/bin/env bash
# CV sweep for the three diversification variants of SetTransformer.
# (The base ``set_transformer`` and the ``compositional_mlp`` baseline have
# already been run; their checkpoints live under ``artifacts/`` — we do NOT
# rerun them here.)
#
# 3 models × 5 folds × 3 seeds = 45 runs.
#
# A run is considered complete if ``metrics.json`` exists in its run dir
# (train.py writes it only at the very end of a successful fit). Any run
# with an existing ``metrics.json`` is skipped so accidental re-invocations
# don't overwrite previous work.
#
# Run from repo root.
set -euo pipefail

MODELS=("set_transformer_wide" "set_transformer_deep" "set_transformer_highdrop")
FOLDS=(0 1 2 3 4)
SEEDS=(42 123 2024)

for model in "${MODELS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_dir="artifacts/${model}/fold_${fold}_seed_${seed}"
      if [[ -f "${run_dir}/metrics.json" ]]; then
        echo "=== ${model}  fold=${fold}  seed=${seed}  [SKIP: already trained] ==="
        continue
      fi
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
