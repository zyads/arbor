# Experiment 1 — ARBOR: does dendritic depth substitute for network depth?

**Answer: no, in the strongest possible sense. Multiplicative dendritic depth is not neutral —
it is actively harmful, monotonically, and the model cannot learn its way out of it.**

Run 2026-07-27/28 on one RTX 4070. `d`=512, 8 heads, ctx 1024, 100M FineWeb-Edu tokens per run,
Muon+AdamW, bf16. Every arm sees byte-identical tokens in byte-identical order; the only thing that
differs between arms is the feedforward unit.

> This document was rewritten after an independent audit found that the *first* version of this
> experiment never activated its own mechanism. That history is kept in §5, because it is the most
> useful part of the record.

## 1. Headline

| variant | val loss | n | σ | vs baseline |
|---|---|---|---|---|
| **SwiGLU baseline** | **3.9799** | 4 | 0.0020 | — |
| ARBOR, `notree` (no tree, no width bottleneck) | 3.9962 | 1 | — | +8σ |
| ARBOR, `flat` (Wu et al.'s `Σσ(w·x)`) | 4.0058 | 1 | — | +13σ |
| ARBOR, `additive` (μ frozen at 0) | 4.0081 | 1 | — | +14σ |
| ARBOR, μ=0 (multiplication inert) | 4.0108 | 1 | — | +15σ |
| **ARBOR, μ=1.0 (multiplication ON)** | **4.2109** | 3 | 0.0101 | **+31.6 pooled σ** |

**Turning the multiplicative NMDA mechanism on costs 0.200 nats (27 pooled σ) versus leaving it
off.** The direction is unambiguous across two independent settings of μ:

- more multiplication → much worse
- less multiplication → less bad
- none at all (ordinary SwiGLU) → best

## 2. The mechanism does not self-tune

μ initialised at 1.0 finished at **0.94–1.02** across every layer and tree level (std ≈ 0.03). The
optimizer had 100M tokens to notice μ was costing it 0.2 nats and dial it toward zero. It did not
move it at all. From the other end, μ initialised at 0.0 random-walked *up* to ~0.02
(`mean|μ| = 0.00153·√step`, the signature of gradient noise, not learning).

So μ is **nearly unidentifiable from the gradient** — it does not self-tune from either direction —
yet its value materially determines the quality of the optimum reached. The network re-equilibrates
every other parameter around whatever μ it is given, settling into a worse local optimum with a
locally flat gradient in μ. It stays stuck wherever it is put.

This is the more interesting half of the result. An architecture knob that (a) strongly affects
final quality and (b) receives no useful gradient is a trap: it looks tuned and is not.

## 3. Depth compression (H1)

Baseline depth curve — smooth, monotone, diminishing returns:

| layers | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| val | 4.3965 | 4.0530 | 4.0065 | 3.9773 | 3.9312 | 3.9155 |

ARBOR (μ=0, tree depth 3), FLOP-matched per layer: L2 4.4279, L4 4.0945, L6 4.0447, L8 4.0108.

**ρ = 0.83** (range 0.64–0.94 depending on interpolation; 0.638 under a power-law fit). The
hypothesis needed ρ ≥ 4 — a 4-layer ARBOR matching a 16-layer transformer. That is a 0.179-nat gap,
roughly **90σ**. H1 is not close, and no defect found in the audit moves it.

Note ρ *fell* when the L=6 baseline point was added, because the true curve is flatter between L=4
and L=8 than linear interpolation assumes. **Denser baseline sampling makes an iso-loss
layer-equivalence metric harsher, not kinder** — worth knowing, since sparse baselines
systematically flatter any proposed architecture.

## 4. Secondary findings

**Noise floor.** Baseline σ = **0.0020** over 4 seeds at 100M tokens (total range 0.0049). ARBOR's
σ = **0.0101** over 3 seeds — **5× noisier**. The comparison is a difference of two random
variables, so both are required; assuming the baseline's σ applies to a new architecture is not
safe.

**Seeding does not make training reproducible.** Re-running an identical config with an identical
seed (`swiglu` L=16, seed 0) reproduces step 0 to four decimals — 10.9227 both times, so init and
data order are deterministic — then diverges to 0.0226 nats apart by 6.7M tokens and settles 0.0039
apart at 100M. The cause is non-deterministic reduction order in the GPU matmul/attention kernels,
compounding through training. So the σ reported above is **total run-to-run variance, not seed
variance**, and it is irreducible by seeding: every comparison needs repeats regardless. A useful
corollary is that same-seed repeats are a free way to measure the noise floor.

**Early-training noise is ~6× late-training noise** (0.0226 at 6.7M vs 0.0039 at 100M for the same
config). This independently corroborates §5.3: the "crossover" sat in the region where run-to-run
scatter is largest, and an architecture compared at 10M tokens can differ from *itself* by 0.02 nats.

**The bottleneck, not the tree, explains most of the residual gap.** All dendritic variants funnel
5632 branches into 704 somas before writing to the residual stream, where SwiGLU writes from 2048.
`notree` (L=0, no width reduction) is the best of the family at 3.9962, recovering roughly half the
gap to baseline. The DESIGN.md sold that 3.2× traffic reduction as a *bandwidth win*; measured, it
is mostly a capacity cost.

**Freezing μ at 0 vs letting it learn from 0** differ by 0.0027 nats (1.1σ) — confirming
behaviourally what instrumentation showed: with μ initialised at zero, the multiplicative machinery
is inert, and what trains is algebraically Wu et al.'s flat additive dendrite.

## 5. The methodological record (the useful part)

The first version of this experiment produced a plausible, well-presented, **wrong** analysis. What
went wrong, and what caught it:

1. **The mechanism never ran.** μ initialised at exactly 0 and never left gradient noise, so the
   "degree-2³ multiplicative unit" was a flat additive tree. Nominal unit depth 5; measured
   junction input/output correlation 0.99, i.e. effective depth ~2. *The independent variable was
   never varied.* Caught by instrumenting a re-run, not by reading the code.
2. **A rounding bug biased every run against the new architecture.** `solve_branch_mult` bisected a
   continuous variable then snapped to a grid, so it did not minimise |flops−target|. Every ARBOR
   run got −2.15% FLOPs and −3.8% params when a strictly closer grid point existed (~12% of the
   measured gap). The repo's own `match_flops(tol=0.02)` check would have caught it and was never
   called.
3. **An "exciting finding" that was noise.** A 2M-token smoke test showed ARBOR ahead by 0.19 nats;
   it was on the wrong side of a crossover. The crossover itself was then oversold — the published
   table subsampled every third eval and omitted step 0, where ARBOR was already behind. The
   "ahead" region's within-run scatter (0.034) was the size of the effect. ARBOR's FFN output std
   at init is 5.2× the baseline's, which alone predicts faster early progress.
4. **Single-seed conclusions.** Everything rested on 0.03-nat differences with no measured σ.
5. **A log filter that could only show good news.** `train.py | grep -E "^step|^FINAL"` — a Python
   traceback matches neither, so seven consecutive OOM crashes looked like silent completions.
6. **16 trainings, zero checkpoints saved.** The defect in (1) could therefore only be found by
   retraining from scratch, and no post-hoc analysis of any run was possible.
7. **An OOM-resilience "feature" silently broke comparability.** `scripts/sweep.sh` retries a failed
   run at a smaller micro-batch. It fired once, on the L=12 baseline, which therefore ran at
   `micro_batch=8` while every other point on the depth curve ran at 16. The objective is identical
   (batch-tokens is fixed; only grad-accum changes) but bf16 accumulation over different groupings is
   not numerically identical: re-running L=12 at micro_batch 16 moved the final loss by **0.0174
   nats**, ~9x the measured sigma. A genuinely identical re-run (L=16, same micro-batch) differs by
   only 0.0039. So one point of the published depth curve was measured under different numerics than
   the other five, with nothing marking it. Fallbacks must record that they fired and mark the run
   non-comparable.

**Every one of these produced a more publishable-looking result than the truth.** That is the
pattern worth internalising: the failure modes are not random, they are biased toward apparent
success. A 2026 multi-seed reproduction study finds most published transformer modifications land
inside 3-seed noise, with several well-known ones negative on reproduction. This experiment is a
worked example of how that happens without anyone intending it.

## 6. What the biology claim survives

The Aizenbud/Beniaguev/Segev result — that a human cortical neuron needs a ~7-layer network to
emulate, driven by NMDA nonlinearity — is not in dispute. What is tested here is the *engineering
inference* drawn from it ("future networks might benefit from more sophisticated neurons"). In this
form that inference is false: the more faithfully the NMDA coincidence mechanism is implemented, the
worse the language model, by a wide margin.

A coherent reading across the literatures: intra-neuron multiplicative depth does not buy
language-model quality (this work); compute-matched dendrites give no expressivity gain (Wu et al.,
*Patterns* 2026); and the transformer-like computation in biology may live in the **neuron–astrocyte
loop** rather than the dendritic tree (Kozachkov, Kastanenka & Krotov, *PNAS* 2023). Biology's
per-unit depth is plausibly an answer to *wiring cost* — space, energy, slow axons — a constraint a
GPU does not have.

## 7. Threats to validity

- **μ=1.0 is one setting.** Both measured endpoints (0 and 1) point toward zero, but a small
  optimum (μ≈0.1) is not excluded. What is excluded is that substantial multiplicativity helps.
- **One scale** (`d`=512) and **one token budget** (100M), well short of compute-optimal.
- **One tree topology** (reducing butterfly, L=3). The tree-depth sweep was cut.
- **Wall-clock is not comparable** across arms: ARBOR ran with gradient checkpointing and the
  baseline did not, and the first baseline run had a cold GPU. ρ is measured in layers, not seconds,
  so this does not affect it.
- The `notree`/`flat`/`additive` controls are single-seed.

Reproduce: `bash scripts/sweep.sh && bash scripts/noise_floor.sh && uv run python -m arbor.analyze`
