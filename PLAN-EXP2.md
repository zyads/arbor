# Experiment 2 (draft) — Moment Attention: reading more than the mean

**Status: draft plan. Not yet implemented, not yet run.**

## The claim

> Attention computes a full distribution `p` over `n` memories — a rich object — and then destroys
> it by taking a weighted average. One vector out, regardless of context length. **We propose
> reading the second moment of the same distribution, which is nearly free and carries information
> the mean provably cannot.**

## Why this slot and not another

Experiment 1 established (see [RESULTS.md](RESULTS.md)) that the feedforward unit has no headroom
for higher-order interactions, because softmax attention already sits at the top of the Dense
Associative Memory capacity ladder (`exp` separation ⇒ exponential capacity).

But **capacity theory bounds how many memories you can store. It says nothing about how much you
can read out per operation.** That is the unclosed gap, and it is where this experiment goes.

Concretely, a head returns

```
o = Σᵢ pᵢ vᵢ = E_p[v]
```

a point in the convex hull of the value vectors. Two consequences:

1. A head that matches **two** things returns their midpoint — which is neither.
2. That midpoint is **indistinguishable** from confidently retrieving a third thing that happens to
   sit at the midpoint.

The read bandwidth of a layer is `H · d_head = d` dimensions, **independent of `n`**. As context
grows, the fraction of available information a layer can retrieve per token falls as `1/n`. Five
years of attention research has made this rank-1 read *cheaper*; essentially none has made it
*wider*.

### Why the obvious fix fails (and why nobody does it)

"Read `p` with several value matrices" does nothing:

```
p V_j = p (X W_j) = (p X) W_j
```

Every readout is a **linear image of the same single vector** `pX`. Zero information gained. So any
useful extra readout must be **nonlinear in `p`**. The cheapest such object is the second moment.

## The mechanism

```
mean    o = E_p[v]                                    (standard attention)
spread  s = sqrt( E_p[v⊙v] − E_p[v]⊙E_p[v] + ε )      (per-dim std of the retrieved set)
out     y = W_o · concat[o, s]                        W_o : 2d → d
```

`E_p[v⊙v] = p(V⊙V)` is **not** a linear image of `pV`, so `s` carries genuinely new information:
*how dispersed was what I retrieved*, per dimension. That is exactly the quantity that disambiguates
"retrieved A and B" from "retrieved the midpoint C".

Interpretation: the head reports **what it found** *and* **how spread out what it found was**.

### Implementation

The expensive part of attention is computing `p` (the QK matmul + softmax); reading `V` is cheap. So
the second moment reuses `p` for free:

- **Preferred:** one `scaled_dot_product_attention` call with `v = concat[V, V⊙V]` of width
  `2·d_head` (PyTorch SDPA permits `v`'s last dim to differ from `q`/`k`). One QK pass, doubled AV.
- **Fallback:** two SDPA calls sharing `q,k` — `SDPA(q,k,V)` and `SDPA(q,k,V⊙V)`.

Either way the flash kernel is retained; no `O(n²)` materialisation of `p`.

### Cost

| | baseline | moment attention |
|---|---|---|
| QK matmul | `2·d·n/2` | unchanged |
| AV matmul | `2·d·n/2` | **2×** |
| out projection | `d²` | `2d²` |
| extra elementwise | — | `V⊙V`, subtract, sqrt: `O(n·d)` |

Roughly **1.15–1.25× total model FLOPs**, absorbed by trimming `d_ff` or head count to hit exact
FLOP parity via the existing `accounting.py` matcher.

## Falsification

Beat a **FLOP-matched *and* param-matched** SwiGLU GPT baseline on val loss at `d`=512, 8 layers,
100M+ tokens, ≥3 seeds. If it does not clear both matched baselines, it is dead and gets written up
as a negative result the same way ARBOR was.

**Run past the crossover.** Experiment 1's central lesson: our own 2M-token smoke test showed a
0.19-nat "win" that reversed at 50M. No verdict before 100M tokens.

## Controls — this is where a fake win would come from

1. **Param-matched baseline.** Moment attention adds `d²` per layer via the wider output
   projection. A baseline with matched params (wider `d_ff`) is mandatory; without it any gain is
   attributable to parameter count.
2. **`shuffled-spread` (the decisive control).** Feed `W_o` a spread vector computed from a
   *shuffled* `p`, so the extra parameters and the extra tensor are present but the information is
   destroyed. If performance holds up, the gain was parameters, not moments. **This is the control
   that decides the experiment.**
3. **`detached-spread`.** Block gradients through `s`. Separates "the spread is a useful feature"
   from "the spread is a useful gradient path".
4. **`raw-moment`.** Use `E_p[v⊙v]` uncentered instead of the centered std, to check the centering
   and sqrt matter.
5. **`more-heads` control.** Spend the same extra FLOPs on additional standard heads instead. This
   is the honest competitor: more heads is the existing way to buy more reads. Moment attention only
   wins if it beats *this*.

## Diagnostic: a synthetic probe that isolates the mechanism

Independent of the LM benchmark, build a task where the mechanism is provably needed:

> Sequence contains tokens tagged with two roles. The target is a function of **whether the two
> retrieved values were far apart or close**, with the *mean held constant by construction*.

Standard single-head attention cannot represent this (the mean is invariant by design); moment
attention should solve it with one head. A clean pass/fail here would establish the mechanism even
if the LM gain is small — and a failure here would kill the idea before spending GPU-days on
language modelling.

Run this **first**. It is minutes, not hours.

## Prior art (checked)

- **Moment Channel Attention (MCA)** — channel attention in CNNs; different operation, different domain.
- **Token Statistics Transformer** — uses second-moment statistics *instead of* pairwise similarity,
  as a linear-attention variant. Opposite goal: it removes softmax attention; we keep it and add a
  readout.
- **Multi-head attention** — the existing way to get more reads, at the cost of recomputing `p` per
  head. Included above as control #5.

No prior work found that keeps softmax attention and adds a second-moment readout of the *same*
distribution. Novelty is plausible but unverified; a proper literature search is a prerequisite
before any write-up.

## Order of work

1. Synthetic probe (minutes) — mechanism works at all, or stop.
2. Implement `MomentAttention` in `arbor/model.py`; extend `accounting.py` for exact FLOP matching.
3. Smoke run vs baseline, confirm wall-clock overhead is near the analytic estimate.
4. Full matched run at `d`=512 / L=8 / 100M tokens: baseline, param-matched baseline, moment,
   shuffled-spread, more-heads.
5. Seeds ×3 on whatever survives step 4.

## Honest prior

Lower than ARBOR's was, and ARBOR failed. The strongest argument against: the model can already
approximate a spread readout by allocating two heads to the same target and letting the FFN subtract
them — so the mechanism may be *available* already, just not cheap. If that's what's happening, the
`more-heads` control will match moment attention and the idea is dead. I estimate **~30%** that it
clears all five controls.
