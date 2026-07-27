#!/usr/bin/env bash
# ARBOR phase-1 sweep.
#
# H1 (headline): depth-compression ratio. Train both families across layer counts at
#   per-layer-matched FLOPs, then read iso-loss crossings.
# H2 + controls: at fixed depth, isolate what the tree is actually contributing.
#
# Sized to coexist with Aether's ollama models on a 12 GB card (~2-3 GB peak).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TOKENS=${TOKENS:-80000000}
DIM=${DIM:-384}
HEAD=${HEAD:-6}
CTX=${CTX:-512}
MB=${MB:-4}
BT=${BT:-65536}
SEED=${SEED:-0}
COMMON="--dim $DIM --n-head $HEAD --ctx $CTX --tokens $TOKENS --batch-tokens $BT --seed $SEED --eval-every 50 --eval-iters 40"

run () {  # run <name> <args...>
  local name=$1; shift
  if [ -f "runs/${name}.done" ]; then echo "skip $name (done)"; return; fi
  for mb in $MB 2 1; do
    echo "=== $name (micro-batch $mb) ==="
    if uv run python -m arbor.train --name "$name" --micro-batch "$mb" $COMMON "$@" 2>&1 | tail -40; then
      touch "runs/${name}.done"; return
    fi
    echo "!! $name failed at micro-batch $mb (likely VRAM contention), retrying smaller"
  done
  echo "!! $name FAILED"
}

mkdir -p runs

# ---- H1: depth sweep -------------------------------------------------------------
for L in 2 4 6 8 12 16; do
  run "h1_swiglu_L${L}_s${SEED}" --unit swiglu --n-layer $L --ffn-mult 4.0
done
for L in 2 4 6 8; do
  run "h1_arbor_L${L}_s${SEED}" --unit arbor --n-layer $L --tree-depth 3 \
      --match-flops-to "swiglu:${L}:${DIM}:4.0"
done

# ---- H2 + controls at fixed depth ------------------------------------------------
D=8
run "h2_flat_L${D}_s${SEED}"     --unit flat  --n-layer $D --tree-depth 3 \
    --no-mult --no-junction-act --no-learned-merge --match-flops-to "swiglu:${D}:${DIM}:4.0"
run "h2_additive_L${D}_s${SEED}" --unit arbor --n-layer $D --tree-depth 3 --no-mult \
    --match-flops-to "swiglu:${D}:${DIM}:4.0"
run "h2_notree_L${D}_s${SEED}"   --unit notree --n-layer $D --match-flops-to "swiglu:${D}:${DIM}:4.0"
for TD in 1 2 4 5; do
  run "h2_arbor_td${TD}_L${D}_s${SEED}" --unit arbor --n-layer $D --tree-depth $TD \
      --match-flops-to "swiglu:${D}:${DIM}:4.0"
done

echo "sweep complete"
