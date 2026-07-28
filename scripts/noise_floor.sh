#!/usr/bin/env bash
# Seed noise floor for the L=8 baseline.
#
# Two independent reviews converged on this being the single most important missing
# measurement. Experiment 1's conclusions rest on 0.031-0.041 nat differences measured on
# ONE seed. Until sigma is known, we cannot say whether that is a result or noise -- and
# the 2026 reproduction study ("Most Transformer Modifications Still Do Not Transfer")
# finds most published transformer modifications land inside 3-seed noise.
#
# This is run BEFORE any experiment-2/3 arm, and the pass threshold is fixed from it
# before looking at any new architecture's number.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TOKENS=${TOKENS:-100000000}
for SEED in 1 2 3; do
  name="nf_swiglu_L8_s${SEED}"
  [ -f "runs/${name}.done" ] && { echo "skip $name"; continue; }
  echo "=== $name $(date +%H:%M) ==="
  uv run python -m arbor.train --name "$name" --unit swiglu --n-layer 8 \
    --dim 512 --n-head 8 --ctx 1024 --tokens $TOKENS --batch-tokens 131072 \
    --micro-batch 16 --seed $SEED --eval-every 50 --eval-iters 40 2>&1 \
    | stdbuf -oL grep -E "^step|^FINAL" && touch "runs/${name}.done"
done

echo "--- noise floor ---"
uv run python - <<'EOF'
import json, glob, statistics
v=[]
for f in sorted(glob.glob('runs/nf_swiglu_L8_s*.jsonl'))+['runs/h1_swiglu_L8_s0.jsonl']:
    st=[json.loads(l) for l in open(f) if l.strip()]
    st=[x for x in st if x['type']=='step']
    if st: v.append((f.split('/')[-1], min(x['val_loss'] for x in st)))
for n,x in v: print(f"  {n:28s} {x:.4f}")
if len(v)>1:
    xs=[x for _,x in v]
    sd=statistics.stdev(xs)
    print(f"\n  n={len(xs)} mean={statistics.mean(xs):.4f} sigma={sd:.4f}")
    print(f"  3-sigma pass threshold: any arm must beat {statistics.mean(xs)-3*sd:.4f}")
    print(f"  ARBOR L=8 was {4.0108:.4f} (deficit {4.0108-statistics.mean(xs):+.4f}, "
          f"{(4.0108-statistics.mean(xs))/sd:+.1f} sigma)")
EOF
