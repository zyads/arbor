#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "scripts/depth_ckpts.sh" >/dev/null; do sleep 60; done
# effective depth across nominal depths -- the actual test of the saturation hypothesis
uv run python -m arbor.effective_depth \
  ck_swiglu_L2_s0 ck_swiglu_L4_s0 nf_swiglu_L8_s1 ck_swiglu_L12_s0 ck_swiglu_L16_s0 \
  mu_arbor_L8_s0 h2_notree_L8_s0 --iters 4 --micro-batch 8
