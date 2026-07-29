#!/usr/bin/env bash
# ARBOR phase-1 sweep.
#
# H1 (headline, runs FIRST): depth-compression ratio rho. Train both families across
#   layer counts at per-layer-matched FLOPs, then read iso-loss crossings.
# H2: at fixed depth, isolate what the tree is actually contributing -- including the
#   controls that would explain a win away.
#
# Defaults assume a dedicated GPU (~11 GB). For a shared card set DIM=384 CTX=512 MB=4.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TOKENS=${TOKENS:-100000000}
DIM=${DIM:-512}
HEAD=${HEAD:-8}
CTX=${CTX:-1024}
MB=${MB:-16}
BT=${BT:-131072}
SEED=${SEED:-0}
COMMON="--dim $DIM --n-head $HEAD --ctx $CTX --tokens $TOKENS --batch-tokens $BT --seed $SEED --eval-every 50 --eval-iters 40"

run () {  # run <name> <args...>
  local name=$1; shift
  if [ -f "runs/${name}.done" ]; then echo "skip $name (done)"; return; fi
  for mb in $MB 8 4 2; do
    echo "=== $name (micro-batch $mb) $(date +%H:%M) ==="
    if uv run python -m arbor.train --name "$name" --micro-batch "$mb" $COMMON "$@" 2>&1 \
        | stdbuf -oL grep -E "^step|^FINAL|^FLOP-matched|Error"; then
      touch "runs/${name}.done"
      if [ "$mb" != "$MB" ]; then
        # MICROBATCH FALLBACK: the objective is unchanged (batch-tokens is fixed, only
        # grad-accum moves) but bf16 accumulation over different groupings is NOT
        # numerically identical -- measured at 0.0174 nats on one L=12 run, ~9x sigma.
        # Mark it so it can never silently enter a curve alongside primary-micro-batch runs.
        echo "$mb" > "runs/${name}.NONCOMPARABLE"
        echo "!! $name ran at micro-batch $mb (primary $MB) -- marked NONCOMPARABLE"
      fi
      return
    fi
    echo "!! $name failed at micro-batch $mb, retrying smaller"
  done
  echo "!! $name FAILED"
}

mkdir -p runs

# ---- H1: depth sweep (the headline) ----------------------------------------------
# Interleaved so a partial sweep still yields a usable curve for both arms.
for L in 8 4 16 2 6 12; do
  run "h1_swiglu_L${L}_s${SEED}" --unit swiglu --n-layer $L --ffn-mult 4.0
  case $L in 8|4|2|6) run "h1_arbor_L${L}_s${SEED}" --unit arbor --n-layer $L --tree-depth 3 \
      --match-flops-to "swiglu:${L}:${DIM}:4.0" ;; esac
done

# ---- H2: what is the tree contributing, at fixed depth ---------------------------
D=8
run "h2_flat_L${D}_s${SEED}"     --unit flat  --n-layer $D --tree-depth 3 \
    --no-mult --no-junction-act --no-learned-merge --match-flops-to "swiglu:${D}:${DIM}:4.0"
run "h2_additive_L${D}_s${SEED}" --unit arbor --n-layer $D --tree-depth 3 --no-mult \
    --match-flops-to "swiglu:${D}:${DIM}:4.0"
run "h2_notree_L${D}_s${SEED}"   --unit notree --n-layer $D --match-flops-to "swiglu:${D}:${DIM}:4.0"
for TD in 2 4 5; do
  run "h2_arbor_td${TD}_L${D}_s${SEED}" --unit arbor --n-layer $D --tree-depth $TD \
      --match-flops-to "swiglu:${D}:${DIM}:4.0"
done

echo "sweep complete"
