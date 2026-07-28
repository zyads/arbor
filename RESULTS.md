# Experiment 1 — ARBOR: does dendritic depth substitute for network depth?

**Answer: no. ρ = 0.86. H1 falsified.**

Run 2026-07-27 on one RTX 4070. `d`=512, 8 heads, ctx 1024, 100M FineWeb-Edu tokens per run,
Muon+AdamW, bf16, one seed. Every arm sees byte-identical tokens in byte-identical order; the
only thing that differs between arms is the feedforward unit.

## H1 — depth compression

Baseline (SwiGLU GPT) validation loss vs depth:

| layers | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| val loss | 4.3965 | 4.0530 | 3.9773 | 3.9155 |

Smooth, monotone, diminishing returns — the curve the ρ interpolation rests on is well behaved.

ARBOR (dendritic unit, tree depth 3), FLOP-matched per layer:

| ARBOR | val | baseline layers needed to match | **ρ** |
|---|---|---|---|
| L=2 | 4.4279 | < 2 | < 1.00 |
| L=4 | 4.0945 | 3.76 | **0.94** |
| L=8 | 4.0108 | 6.23 | **0.78** |

**Mean ρ = 0.86.** The hypothesis needed ρ ≥ 4 ("a 4-layer ARBOR matches a 16-layer transformer").
It got 0.86 — a dendritic layer is worth *less* than the plain transformer layer it replaces.

**ρ degrades with depth (0.94 → 0.78).** The raw loss deficit is roughly constant (0.031 / 0.041 /
0.034 nats at L=2/4/8), but the baseline's depth curve flattens, so a constant deficit costs more
layer-equivalents the deeper you go. The trend runs *opposite* to the hypothesis: ARBOR was meant
to let you go shallower, and instead its layer-equivalent value decays as depth grows. There is no
regime where the trade turns favourable.

On top of that it is **66% slower in wall-clock** (27.2 vs 16.4 min at L=8) for identical peak VRAM.
It loses on every axis measured.

## The finding that actually matters: a crossover

At L=8, gap = ARBOR − baseline (negative = ARBOR ahead):

| tokens | 20M | 40M | 59M | 79M | 98M |
|---|---|---|---|---|---|
| gap | **−0.026** | **−0.040** | +0.025 | +0.032 | **+0.034** |

**The dendritic advantage is an early-training artifact that reverses at ~50M tokens, and the
deficit is still widening at the end of the run.**

This is the most transferable result here. A short experiment — which is what most small-scale
architecture papers run — lands squarely in the misleading region and yields an exciting positive.
Our own 2M-token smoke test showed ARBOR ahead by 0.19 nats. It was noise from the wrong side of
the crossover. This plausibly explains why the dendritic-ANN literature is so mixed: Wu et al.
(*Patterns* 2026) found a null under proper compute matching while smaller/shorter studies report
gains.

**Any architecture comparison that stops before the crossover is measuring optimisation speed, not
model quality.**

## Why it failed — two explanations, in order of strength

**1. The unit was the wrong place for higher-order interactions (strongest).**
Ramsauer et al. showed softmax attention *is* the retrieval step of a modern Hopfield network.
Krotov & Hopfield's Dense Associative Memory results say capacity is set by the interaction order
of the energy function: `x²` → ~0.14N, `xⁿ` → ~N^(n−1), `exp(x)` → **exponential**. Higher-order
multiplicative interactions buy enormous power — *in the retrieval operation* — and softmax already
sits at the top of that ladder.

ARBOR put a degree-2³ polynomial unit in the **feedforward** path, where the capacity theorem says
nothing, next to an operator with infinite interaction order. There was never headroom there.
(Corollary: every linear-attention variant is a demotion on this ladder, `exp` → polynomial, which
reframes their quality gap as a capacity theorem rather than an engineering shortfall.)

**2. Write-bandwidth compression (weaker, still plausible).**
The reducing tree takes 5120 branches down to 640 somas, so ARBOR writes into the residual stream
through a 640-dim port where the baseline uses 2048. Early in training the FFN has little to say and
a narrow port is free; late in training it binds. This fits the crossover timing. The queued H2
ablations discriminate: if `notree` (L=0, no compression) beats full ARBOR, this explanation carries
weight.

## What the biology claim actually survives

The Aizenbud/Beniaguev/Segev result — that a human cortical neuron needs a ~7-layer network to
emulate, driven by NMDA nonlinearity — is not in dispute. What this experiment tests is the
*engineering inference* drawn from it ("future networks might benefit from more sophisticated
neurons"), and at least in this form that inference does not hold.

A coherent reading of all three literatures: intra-neuron depth doesn't buy language-model quality
(this work); properly compute-matched dendrites give no expressivity gain (Wu et al.); and the
transformer-like computation in biology may live in the **neuron–astrocyte loop** rather than the
dendritic tree (Kozachkov, Kastanenka & Krotov, PNAS 2023). Biology's per-unit depth is plausibly an
answer to *wiring cost* — space, energy, slow axons — which is a constraint a GPU does not have.

## Threats to validity

- **One seed.** The 0.031–0.041 nat deficits are consistent across three depths, which argues
  against noise, but no error bars are reported and none should be inferred.
- **Undertrained.** 100M tokens for ~50M params is well short of compute-optimal. The crossover
  finding suggests longer runs would *widen* the deficit, not close it, but that is extrapolation.
- **One scale.** `d`=512 only.
- **Wall-clock is implementation-dependent.** The 66% penalty reflects unfused elementwise tree ops
  under `torch.compile`; a hand-written fused kernel would narrow it. It would not change ρ, which
  is measured in layers, not seconds.
- **One tree topology.** Reducing butterfly, `L`=3. The H2 sweep over `L ∈ {2,4,5}` and the
  width-preserving variant are not yet complete.

## Status

H1 complete and falsified. H2 ablations (`flat` / `additive` / `notree` / tree-depth sweep) running;
they cannot change the H1 verdict, only the diagnosis of *why*.

Reproduce: `bash scripts/sweep.sh && uv run python -m arbor.analyze`
