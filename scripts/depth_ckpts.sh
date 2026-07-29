#!/usr/bin/env bash
# Re-run the baseline depth sweep WITH checkpoints, so the effective-depth instrument can
# be applied across nominal depths. The original H1 runs predate checkpoint saving.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for L in 16 12 4 2; do
  name="ck_swiglu_L${L}_s0"
  [ -f "runs/${name}.done" ] && { echo "skip $name"; continue; }
  echo "=== $name $(date +%H:%M) ==="
  if uv run python -m arbor.train --name "$name" --unit swiglu --n-layer $L \
      --dim 512 --n-head 8 --ctx 1024 --tokens 100000000 --batch-tokens 131072 \
      --micro-batch 16 --seed 0 --eval-every 50 --eval-iters 40 > "runs/${name}.out" 2>&1; then
    grep -E "^FINAL" "runs/${name}.out"; touch "runs/${name}.done"
  else
    echo "!! $name FAILED"; tail -8 "runs/${name}.out"
  fi
done
echo "depth checkpoints complete"
