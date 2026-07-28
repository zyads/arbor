#!/usr/bin/env bash
# Phase 2: seed noise floor + the re-run that actually tests ARBOR's hypothesis.
#
# Three independent reviews converged on two blocking problems with Experiment 1:
#
#  (a) NO NOISE FLOOR. Every conclusion rests on 0.031-0.041 nat differences from ONE
#      seed. An independent short-budget measurement put sigma at ~0.014 nats, making
#      those gaps only 2.3-3.0 sigma. Until sigma is measured at the real token budget,
#      rho is not a number. And the comparison is a DIFFERENCE of two random variables,
#      so ARBOR's sigma is needed too, not just the baseline's.
#
#  (b) THE MECHANISM NEVER RAN. mu (the NMDA coincidence term, the entire point of the
#      architecture) initialised at exactly 0 and only random-walked on gradient noise:
#      measured mean|mu| = 0.00153*sqrt(step), reaching ~1% of the junction signal. What
#      actually trained was a flat ADDITIVE tree -- algebraically Wu et al.'s dendrite,
#      the very thing ARBOR was built to move beyond. Re-run with mu_init=1.0 (~20% of
#      the junction signal at init).
#
# Also fixed since Experiment 1: solve_branch_mult had a snapping bug that handed ARBOR a
# ~2% FLOP and ~3.8% param deficit in every run. Now M=5632 (+1.6%) instead of 5120 (-2.15%).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TOKENS=${TOKENS:-100000000}
COMMON="--dim 512 --n-head 8 --ctx 1024 --tokens $TOKENS --batch-tokens 131072 \
        --micro-batch 16 --eval-every 50 --eval-iters 40"

run () {
  local name=$1; shift
  [ -f "runs/${name}.done" ] && { echo "skip $name"; return; }
  echo "=== $name $(date +%H:%M) ==="
  # Full output to a per-run log. Piping straight into grep swallows tracebacks -- an
  # earlier launch OOMed on every run and looked like a silent instant success/failure
  # because only ^step|^FINAL lines could ever reach the log.
  local lg="runs/${name}.out"
  if uv run python -m arbor.train --name "$name" $COMMON "$@" > "$lg" 2>&1; then
    grep -E "^step|^FINAL|^FLOP-matched" "$lg" | tail -6
    touch "runs/${name}.done"
  else
    echo "!! $name FAILED -- last 12 lines of $lg:"; tail -12 "$lg"
  fi
}

# (a) baseline noise floor -- seeds 1,2,3 (seed 0 already exists as h1_swiglu_L8_s0)
for SEED in 1 2 3; do
  run "nf_swiglu_L8_s${SEED}" --unit swiglu --n-layer 8 --seed $SEED
done

# (b) ARBOR with the multiplicative term ACTUALLY ACTIVE, 3 seeds for its own sigma
for SEED in 0 1 2; do
  run "mu_arbor_L8_s${SEED}" --unit arbor --n-layer 8 --tree-depth 3 --mu-init 1.0 \
      --match-flops-to swiglu:8:512:4.0 --seed $SEED
done

echo "--- phase 2 summary ---"
uv run python - <<'PY'
import json, glob, statistics
def best(f):
    st=[json.loads(l) for l in open(f) if l.strip()]
    st=[x for x in st if x['type']=='step']
    return min(x['val_loss'] for x in st) if st else None
grp={'baseline (swiglu L8)': sorted(glob.glob('runs/nf_swiglu_L8_s*.jsonl'))+['runs/h1_swiglu_L8_s0.jsonl'],
     'ARBOR mu=1.0 L8':      sorted(glob.glob('runs/mu_arbor_L8_s*.jsonl')),
     'ARBOR mu=0 L8 (exp1)': ['runs/h1_arbor_L8_s0.jsonl']}
stats={}
for name, fs in grp.items():
    vs=[best(f) for f in fs if glob.glob(f)]
    vs=[v for v in vs if v is not None]
    if not vs: continue
    m=statistics.mean(vs); sd=statistics.stdev(vs) if len(vs)>1 else float('nan')
    stats[name]=(m,sd,len(vs))
    print(f"  {name:24s} n={len(vs)} mean={m:.4f} sigma={sd:.4f}  {[round(v,4) for v in vs]}")
if 'baseline (swiglu L8)' in stats and 'ARBOR mu=1.0 L8' in stats:
    bm,bs,bn=stats['baseline (swiglu L8)']; am,asd,an=stats['ARBOR mu=1.0 L8']
    pooled=((bs**2+asd**2)/2)**0.5
    print(f"\n  ARBOR(mu=1.0) - baseline = {am-bm:+.4f} nats = {(am-bm)/pooled:+.2f} pooled sigma")
    print(f"  => {'SIGNIFICANT' if abs(am-bm)>3*pooled else 'INSIDE 3-SIGMA NOISE'}")
PY
