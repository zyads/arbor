# ARBOR — shallow networks of deep neurons

## The claim

The neocortex is a **~6-layer network of ~7-layer-deep units**.
A language model is a **~100-layer network of 2-layer units**.

Nobody has tested the cortical configuration. ARBOR does.

> **Headline hypothesis (H1): a 4-layer ARBOR matches a 16-layer transformer at equal quality.**

### Why this is the claim worth making

Network depth is the only resource in a transformer you cannot buy your way out of. Width
parallelises across cores; batch parallelises across the batch; **layers are sequential**. `L` layers
means `L` dependent kernel launches, each round-tripping the full activation tensor through HBM,
before a single token comes out. That serial depth is the latency floor of every forward pass, and
it is the binding constraint on local inference and on every token-serial reasoning/agent workload —
precisely the regimes where more GPUs do not help.

A dendritic tree is the opposite shape: all `M` branches integrate **in parallel**, and the tree
merges them in `log₂` depth with fused elementwise ops and no HBM round-trip. So ARBOR trades
*sequential network depth* for *parallel unit depth*. If the trade is favourable, iso-quality
latency drops by the depth-compression ratio.

And the claim is **scale-free**. If a 4-layer model matches a 16-layer one at 50 M params, the same
architectural fact applies at 1 T. That is the one kind of result a single person with a single
consumer GPU can produce that does not care how many GPUs anyone else has.

**Primary metric is therefore not val loss. It is the depth-compression ratio**
`ρ = (baseline layers needed to match) / (ARBOR layers)`, measured by training both families across
a depth sweep and reading off iso-loss crossings. `ρ = 1` falsifies the project. `ρ ≥ 2` is a real
result. `ρ ≥ 4` is a big one.

### Secondary thesis (H2)

At matched FLOPs *and* matched wall-clock, ARBOR reaches lower validation loss than a strong SwiGLU
GPT baseline — capability per FLOP, per parameter, and per byte moved, the three things that bind on
a 12 GB consumer GPU.

This is a research repo with falsifiable hypotheses and a compute-matched harness, not a
"biologically inspired" gesture. The harness is built to let the hypotheses *lose*.

---

## 1. What the research actually says

### 1.1 A single neuron is deep (Beniaguev, London & Segev, *Neuron* 2021)

They fit a temporal-convolutional DNN to the millisecond-resolution input→output map of a
biophysically detailed L5 cortical pyramidal cell. Result: **7 layers, 128 channels, 153 ms of
history** were needed. Strip the NMDA conductances and the required depth collapses to ~1–2 layers.
So the depth is not an artifact of the fitting — it is *caused by the NMDA nonlinearity in the
dendrites*.

### 1.2 Human neurons are deeper than rat neurons (Aizenbud, Yoeli, Beniaguev, de Kock, London & Segev, *PNAS* 2026)

They define a **Functional Complexity Index (FCI)**: how deep an ANN must be to reproduce a cell's
I/O. Human cortical neurons score consistently higher than rat. The causes they isolate:

- expanded dendritic surface + richer branching (more, and more separated, integration sites)
- higher density and stronger nonlinearity of **NMDA-receptor** signalling

And a detail worth stealing: **complexity peaks in L2/3 in humans, but in L5 in rats** — i.e. the
brain allocates per-unit complexity *non-uniformly across depth*. That is a free architectural
hyperparameter nobody in ML tunes.

### 1.3 The inconvenient counter-result (Wu, Zhao, Yu, Ma et al. + Poirazi, *Patterns* 2026)

This is the paper that keeps the project honest. They replaced point neurons with dendritic neurons:

```
d̂ⱼ = wⱼᵀx + bⱼ ,  dⱼ = σ(d̂ⱼ)          (branches)
h  = Σⱼ₌₁ᴷ dⱼ                            (soma)
```

Normalising for compute (K branches per neuron ⇒ D/K neurons), they found **little to no accuracy
difference vs point neurons** across ResNets, transformers and 1-D conv speech models, dense and 85%
sparse. Their conclusion: dendrites do **not** buy expressivity. What they buy is **communication
efficiency** — the same accuracy with `Ψ = D_dendritic/D_point` fewer neurons, hence far fewer
activations shuttled to off-chip memory. (Their framing: a 32-bit add costs ~20 fJ; fetching the
operands from memory costs ~1.3 nJ — ~64,000×.)

Chavlis & Poirazi separately report marked *parameter* efficiency from dendritic sampling, which
they attribute to sparse connectivity rather than the nonlinearity itself.

### 1.4 The gap this repo attacks

Look carefully at what Wu et al. actually tested: `h = Σⱼ σ(wⱼᵀx)`. That is

- **2 layers deep**, not 7;
- the soma sum is **unweighted and linear** — no free parameters, no nonlinearity at the junction;
- **purely additive** — no multiplicative interaction anywhere.

But §1.1 says the depth in a real neuron comes *specifically from NMDA*, and NMDA is not a
saturating sum — it is a **coincidence detector**: the channel conducts only when glutamate is bound
**AND** the local membrane is already depolarised. That is a *product*, not a sum. It is the
mechanism behind "the neuron distinguishes not just how many signals arrived, but which combination
arrived."

> **So the null result in §1.3 is a result about flat, additive dendrites. The deep, multiplicative,
> hierarchical unit that §1.1 and §1.2 actually describe has not been tested in a language model.**

That is the hole ARBOR is built to fill.

### 1.5 Where the existing ML primitives sit

SwiGLU — `W_down(SiLU(W_gate x) ⊙ (W_up x))` — is already **a one-level dendritic tree with a single
multiplicative junction**. The field converged on a *fragment* of the biology by pure empiricism and
then stopped. ARBOR's question is whether the rest of the tree is worth anything.

Nearby prior art, and how ARBOR differs:
| Work | Multiplicative? | Hierarchical? | Where the tree lives |
|---|---|---|---|
| GLU / SwiGLU | yes, 1 level | no | feature dim |
| Wu et al. dendrites | no | 1 level, unweighted | feature dim |
| Chavlis & Poirazi | no | 1 level | feature dim, sparse sampling |
| Wave-Attractor-Tree (2026) | GLU at merges | yes | **sequence** dim |
| Numenta Active Dendrites | gating by context | no | feature dim |
| **ARBOR** | **yes, at every level** | **yes, L levels** | **feature dim, inside one FFN** |

---

## 2. The ARBOR unit

Let `d` = model width, `M` = number of **branches** (synaptic integration sites), `L` = **tree
depth**, `N = M / 2^L` = number of **somas**.

**Stage 1 — synaptic integration.** The only large GEMM on the way in:

```
s = W_in x            W_in ∈ R^{M×d}
```

**Stage 2 — NMDA branch nonlinearity.** Supralinear near threshold, saturating above, with *learned
per-branch* gain and threshold:

```
b = s · sigmoid(α ⊙ s − θ)        α, θ ∈ R^M   (2 params per branch)
```

**Stage 3 — dendritic tree, `L` levels of multiplicative junctions.** At level ℓ each surviving
channel is paired with its butterfly partner (`i ⊕ 2^ℓ`), so the receptive field doubles per level
and reaches all-to-all in `log₂M`. Each junction is a *learned, weighted, coincidence-detecting*
merge:

```
z = γ·u + δ·v + μ·(u ⊙ v)         3 params per junction
u ← z · sigmoid(α'·z − θ')        junction NMDA spike
```

`μ` is the NMDA term: it fires on **conjunctions** of the two subtrees, not on their total. The tree
halves the width each level, so total junction cost is `3M(1 − 2^{-L}) ≈ 3M` params and `O(M)`
FLOPs — **negligible against the GEMMs**. This is the "free depth": `L` extra nonlinear layers with
no wiring cost, which is exactly the trick a dendrite plays.

**Stage 4 — soma:**

```
y = W_out h           W_out ∈ R^{d×N},  N = M/2^L
```

**Stage 5 (optional) — dendritic time constants.** Per-branch causal depthwise conv over the
sequence (kernel 3–4) between stages 1 and 2, standing in for the 153 ms of history each real
synapse integrates over. Cheap (`k·M` params). **Must be granted to the baseline too in any
ablation** — short convs are independently known to help, and letting only ARBOR have one would be
cheating.

### 2.1 Why this should be cheaper, concretely

| | SwiGLU | ARBOR |
|---|---|---|
| params | `3·d·d_ff` | `d·M + d·M/2^L + ~3M` |
| in-GEMM | `2·d·d_ff` | `d·M` |
| out-GEMM | `d·d_ff` | `d·M/2^L` |
| activation written before the down-proj | `d_ff` floats | `M/2^L` floats |
| nonlinear depth of the unit | 2 | **L+2** |

Set `M = 2·d_ff` (identical in-GEMM cost to gate+up). ARBOR's out-GEMM is then `2^{L-1}×` smaller
than SwiGLU's, total ≈ `2·d·d_ff` vs `3·d·d_ff` — **1.5× cheaper at equal branch count**, or 1.5×
more branches at equal budget, *plus* L free nonlinear levels. The tensor crossing the memory bus
before the down-projection shrinks by `2^L`, which is Wu et al.'s communication win made concrete on
a bandwidth-starved 4070 (504 GB/s).

**Falsifiable claim:** at matched FLOPs/token *and* matched wall-clock on one RTX 4070, ARBOR
reaches lower validation loss than a strong SwiGLU GPT baseline. If it doesn't, §1.3 generalises to
the multiplicative case and the answer is "no" — which is a real finding and gets written up as one.

### 2.2 Known failure modes (stated up front)

1. **Variance explosion.** A chain of L products is a degree-`2^L` polynomial. Mitigation: init
   `μ = 0` (tree starts purely additive and additively-equivalent to a wide MLP), `γ = δ = 0.5`,
   RMSNorm at the soma, and per-level variance tracking in the logs.
2. **Memory-bound elementwise ops.** The tree is `O(M)` FLOPs but touches `O(M)` memory L times. If
   the levels aren't fused into one kernel the "free" depth costs real wall-clock. `torch.compile`
   fusion is not optional; wall-clock is a reported metric precisely so this can't be hidden.
3. **The halving is a bottleneck.** The reducing tree destroys information by construction. That's
   the trade being tested, not a bug — but the width-preserving butterfly variant is implemented as
   a control.
4. **Deep units may just be harder to optimise.** Muon + careful init, and a `μ`-warmup schedule if
   the additive→multiplicative transition proves unstable.

---

## 3. Experimental protocol

Non-negotiable, because this is exactly where architecture papers cheat:

- **One data pipeline**, byte-identical tokens and order for every variant; fixed held-out val split.
- **Matched budgets, three ways** — params, FLOPs/token, and wall-clock. Report all three. A win on
  only one is reported as a win on only one.
- **Strong baseline.** Modern GPT: RMSNorm, RoPE, QK-norm, SwiGLU, Muon+AdamW. Beating a weak
  baseline proves nothing.
- **Controls, not just the hero model:**
  - `flat-dendrite` — Wu et al.'s `Σσ(wx)`, to reproduce their null in *our* setting first;
  - `additive-tree` — ARBOR with `μ ≡ 0` frozen, isolating hierarchy from multiplicativity;
  - `no-tree` — `L=0`, isolating the branch nonlinearity alone;
  - conv given to *both* arms.
- **Seeds.** ≥3 per config; report spread, not the best run.
- **Every run logs** params, FLOPs/token, tokens seen, wall-clock, peak VRAM, activation bytes.

**Phase 1 scale (fits the 4070):** ~30–60 M params, `d`=512, 8–10 layers, ctx 1024, FineWeb-Edu,
1–3 B tokens, bf16. ~4 GB VRAM is currently free (ollama holds 8 GB) — see README for freeing it.

## 4. Roadmap beyond the FFN

Phase 1 is the FFN only, because it's the clean, isolable test. If it wins:

2. **Dendritic value path in attention** — the same tree on `V`, so heads combine conjunctively.
3. **Heterogeneous depth across layers** — per §1.2, humans peak at L2/3, rats at L5. Sweep `L` as a
   function of network depth instead of holding it constant.
4. **Sparse somas (k-WTA)** — cortex is sparse; top-k somas turns the communication win into an
   inference-time bandwidth win as well.
5. **Fewer, fatter layers** — the real payoff. If the unit is 7 layers deep, does a 4-layer ARBOR
   match a 12-layer GPT? That's where the memory-bandwidth and latency win becomes large.

## 5. Explicitly rejected

Spiking neurons, STDP / Hebbian "biologically plausible" learning rules, Hodgkin–Huxley or
multi-compartment biophysical simulation, and neuromorphic encodings. There is no evidence any of
these beat gradient descent at language-model scale, and they would burn the entire compute budget
on fidelity to mechanisms that exist because neurons are wet, not because they compute well.
**We steal the topology and the nonlinearity. We keep backprop.**

## 6. References

- Beniaguev, Segev & London. *Single cortical neurons as deep artificial neural networks.* Neuron, 2021. https://www.cell.com/neuron/fulltext/S0896-6273(21)00501-8
- Aizenbud, Yoeli, Beniaguev, de Kock, London & Segev. *Dendritic morphology and synaptic nonlinearities enhance functional complexity in human cortical neurons.* PNAS, 2026. https://www.pnas.org/doi/10.1073/pnas.2533168123
- Wu, Zhao, Yu, Ma, Gao, Yip, Tang, Pan, Poirazi & Huang. *Dendritic nonlinearities mitigate communication costs.* Patterns 7(6), 2026. https://doi.org/10.1016/j.patter.2026.101520
- Beniaguev et al. *Single biological neurons as temporally precise spatio-temporal pattern recognizers.* arXiv:2309.15090
- Poirazi & Mel. Two-layer model of a pyramidal neuron. (dendritic subunit hypothesis)
