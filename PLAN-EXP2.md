# Experiment 2 (draft v2) — Is the attention value-read low-rank bottlenecked?

**Status: draft plan, revised after independent pre-registration review. Not yet implemented.**

> **v1 of this plan was wrong in its central argument and its novelty claim. Both are corrected
> below, and the correction changed what the experiment is.** See §0.

---

## 0. What v1 got wrong

**The load-bearing argument was false.** v1 claimed that reading `p` with several value matrices
gains nothing, since `p V_j = (pX) W_j` makes every readout a linear image of one vector `pX` — and
concluded that any useful extra readout must therefore be *nonlinear* in `p`.

The algebra is right. The inference is wrong. `pX ∈ R^d` with `d`=512, but a head reads a
**`d_head`=64**-dimensional linear image of it. The head is discarding 448 of 512 available *linear*
dimensions. A second **linear** value projection buys up to 64 genuinely new dimensions, at exactly
the FLOP and parameter cost the nonlinear version wanted to spend. The argument would only hold if
`d_v` were already `d`; it is `d/H`. This is the standard low-rank attention bottleneck
(Bhojanapalli et al., ICML 2020).

**Consequence:** "Moment Attention" is **wide-value attention with the second value matrix frozen to
the elementwise square.** The whole delta from the cheapest linear alternative is one `x⊙x`. v1
never ran that alternative.

**The operator is not novel.** `σ = sqrt(Σ_t α_t h_t⊙h_t − μ⊙μ)`, concatenated with `μ` and fed to a
linear layer under the same softmax weights, is **Okabe, Koshinaka & Shinoda, "Attentive Statistics
Pooling for Deep Speaker Embedding," Interspeech 2018** (arXiv:1803.10963, eqs. 4–6) — equation for
equation, including the "the mean loses information the spread retains" motivation. Standard
speaker-verification practice since 2018 (x-vector → ASP → ECAPA-TDNN). Independently rediscovered
as variance pooling for MIL (Carmichael et al., MICCAI 2022, arXiv:2206.08885). Related: TCPNet
(NeurIPS 2021), Moment Channel Attention (AAAI 2024), Token Statistics Transformer (ICLR 2025),
LASER (arXiv:2411.03493 — a cumulant-generating-function readout, the closest LM-side analogue).

What remains open is a **placement** claim, not a mechanism claim: nobody appears to have put this
operator inside the token mixer of a causal LM, per token, per head, every layer. That is also weak
*negative* evidence — the speech community has had it for eight years and never moved it there.

## 1. The revised question

Not "does a nonlinear readout beat the mean" but:

> **Is the attention value/output read low-rank bottlenecked at `d_head`, and if so, does a
> *nonlinear* readout buy anything a *wider linear* read does not?**

Both outcomes are informative, which v1 could not claim:

- wide-value > baseline ⇒ the value read is bottlenecked; a same-cost linear fix exists. Useful.
- moment > wide-value ⇒ nonlinearity in `p` genuinely adds something. The interesting outcome.
- neither > baseline ⇒ the read is not the constraint. Kills a framing this project has been
  leaning on since Experiment 1.

## 2. Arms

All at `d`=512, 8 layers, ctx 1024, FLOP- and param-matched via `accounting.py`.

| arm | readout | notes |
|---|---|---|
| `baseline` | `E_p[v]`, `d_v = d_head` | reference |
| `baseline-wide` | `E_p[v]`, extra params in `d_ff` | param-matched control |
| **`wide-value`** | `E_p[v]`, `d_v = 2·d_head`, `W_o: 2d→d` | **primary comparator** — purely linear |
| `moment-raw` | `concat[E_p[v], sqrt(E_p[v⊙v])]` | the challenger |
| `more-heads` | 2× heads, `d_head/2` | secondary; changes #distributions, a different axis |
| `nonlinear-value` | `concat[E_p[v], p·SiLU(V)]` | if this ties `moment-raw`, "second moment" is decoration |

**Gate: if `moment-raw ≤ wide-value`, stop.** That single comparison decides the experiment, and no
control in v1 would have detected it.

## 3. Protocol — noise floor first

Experiment 1 measured effects of 0.031–0.041 nats on **one seed**. Screening at that effect size
without a noise floor is measuring nothing.

1. **Run 3 baseline seeds. Publish σ and fix the pass threshold before looking at any other arm.**
   ~50 min, and it retroactively bounds how much Experiment 1's ρ=0.83 can be trusted. Worth doing
   regardless of this experiment.
2. 3 seeds × 6 arms × ~19 min ≈ **5.7 h**. Within budget.
3. **Run past the crossover.** Experiment 1's central lesson: our own 2M-token smoke test showed a
   0.19-nat "win" that reversed at 50M tokens. No verdict before 100M.

## 4. Implementation — corrected against measurements on this machine

torch 2.11.0+cu128, RTX 4070:

- **Use two SDPA calls sharing `q,k`, not a widened `v`.** Verified: FlashAttention rejects
  `v` with a different last dim (`Flash attention requires q,k,v to have the same last dimension`)
  and silently falls back. Measured fwd+bwd on the isolated op: baseline flash **2.13 ms**,
  wide-`v` **4.93 ms**, two flash calls **4.37 ms**. v1's "preferred" path was the slower one and
  its claim that the flash kernel is retained was false.
- **Use the raw second moment `sqrt(E_p[v⊙v])`, not the centered std.** The centered version suffers
  catastrophic cancellation exactly where the mechanism is supposed to matter: at peaked attention
  (entropy 0.8 nats) **3.2% of `E[v²]−m²` entries go negative in bf16**, median relative error 1.4%.
  It also has a gradient pathology — with `ε=1e-6`, `d(sqrt)/d(var)` is 500 at var≈0 vs 0.5 at
  var≈1, a 1000× amplification, on top of `clamp_min(0)` zeroing gradients on the negative entries.
  The raw moment is always ≥0, needs no clamp or ε, and is information-equivalent given `m`.
- Keep any centered variant, if run at all, in fp32 for the subtraction.

## 5. Cost — corrected

Recomputed with this repo's `accounting.py` at `d`=512/L=8/ctx=1024: baseline 1.270e8 fwd
FLOPs/token; extra AV pass + doubled `W_o` add 8.39e6 ⇒ **1.066× including lm_head, 1.111× excluding
it** — not v1's claimed 1.15–1.25×. Measured end-to-end (compiled, bf16, B=8, T=1024, fwd+bwd):
baseline **162.7 ms/step**, two-flash-call **183.1 ms (1.13×)**; params 59.32M → 61.42M (+3.5%).
v1's wall-clock figure was accidentally near-right for the wrong reason — it overestimated FLOPs by
~2× and omitted the lost flash kernel.

## 6. Controls dropped or demoted from v1

- **`shuffled-spread` — dropped from the first pass.** It has a causality trap: shuffling `p` across
  a row leaks future positions unless shuffled within each variable-length causal prefix, and doing
  so changes the concentration statistics of `s`, so it is not an information-only ablation. Largely
  redundant once `wide-value` exists.
- **`more-heads` — demoted to secondary.** It changes the number of *distributions*, a different
  axis. To isolate "must the readout be nonlinear in `p`," hold `p` fixed and widen the read: that
  is `wide-value`.
- **Synthetic probe — demoted to a 10-minute correctness filter.** v1 claimed "standard single-head
  attention cannot represent this." The baseline has 8 heads and 8 layers and the tokens are
  role-tagged by construction, so head 1 keys on role A, head 2 on role B, and the FFN computes
  `|A−B|`. The baseline will pass. Only a *failure* is informative. Run `baseline` and `wide-value`
  on it too, and delete the claim that passing establishes the mechanism.

## 7. Honest prior

**~20%** that `moment-raw` beats both matched baselines by more than 3-seed noise; **~12%** that it
also beats `wide-value` — and that second number is the one that should govern the decision.

Arguments against: the operator has been available and universally used in speech since 2018 without
anyone moving it into a token mixer; the diagonal readout recovers per-axis *magnitudes* only, not
directions (recovering A and B from a two-item retrieval needs the full `O(d²)` attention-weighted
covariance); measured, **76–92% of the aggregate spread signal is linearly predictable from `‖m‖`**,
which the model already has; and the genuinely new content is one hardcoded quadratic feature
competing against a *learnable* linear alternative of identical cost.

The honest expected value of this experiment now sits mostly in the **`wide-value` arm** — a
same-cost, purely linear question about whether attention's value read is rank-bottlenecked — rather
than in the moment readout that motivated it.
