# ARBOR

**Shallow networks of deep neurons.**

The neocortex is a ~6-layer network of ~7-layer-deep units. A language model is a ~100-layer
network of 2-layer units. Nobody has tested the cortical configuration. This repo does, on one
RTX 4070.

Headline hypothesis: **a 4-layer ARBOR matches a 16-layer transformer.** Network depth is
sequential — it is the latency floor of every forward pass and the one resource more GPUs cannot
buy down. A dendritic tree is internally parallel and log-depth. If the trade is favourable,
iso-quality inference latency drops by the depth-compression ratio, at *any* scale.

Full derivation, prior art, and the honest counter-evidence: **[DESIGN.md](DESIGN.md)**.

---

## What this is not

It is not a plan to out-train a frontier model on a consumer GPU. That is arithmetically
excluded: Kimi K2 is ~3×10²⁴ training FLOPs; a 4070 delivers ~10¹³ FLOP/s, i.e. **~9,500 years**.
One month of this GPU buys a ~450M-param model. No architecture is worth a 10⁵× compute multiplier
— the entire transformer era's algorithmic progress is worth maybe 10–100×.

What a single GPU *can* produce is a **scale-free architectural result**: if the depth trade holds
at 50M params it holds at 1T, and it applies to whoever does have the GPUs. So the product here is
the measurement, not the model. The harness is built to let the hypothesis lose.

## The unit

```
s = W_in x                                  synaptic integration      (the one big GEMM)
b = s · σ(α·s − θ)                          NMDA branch nonlinearity
repeat L times:                             the dendritic tree
    u, v = b.chunk(2)                       butterfly pairing
    z    = γ·u + δ·v + μ·(u ⊙ v)            ← μ is NMDA coincidence detection
    b    = g · z · σ(α'·z − θ')             junction NMDA spike
y = W_out b                                 soma
```

`chunk(2)` is a decimation-in-frequency butterfly, so after `L` levels each soma aggregates `2^L`
branches spread at stride `N` across the full width. The tree costs `O(M)` params and FLOPs against
the GEMMs' `O(M·d)` — **~1% of the unit's FLOPs for `L` extra nonlinear levels.** That is the trick
a dendrite plays: depth without wiring cost.

At FLOP-parity with SwiGLU (`d`=512, 8 layers, `L`=3): 5120 branches, 640 somas, unit depth 5 vs 2,
and **3.2× fewer floats crossing HBM into the output projection.**

## Status

| | |
|---|---|
| Model + baseline + controls | done |
| Compute-matched accounting (params / FLOPs / bytes / wall-clock / VRAM) | done |
| Data pipeline (FineWeb-Edu, 1.45B+ tokens) | done |
| H1 depth sweep | **running** |
| H2 ablations | queued |

**First signal only** (2.1M tokens, `d`=384, 4 layers, matched FLOPs, identical data and order):

| | val loss | wall-clock | peak VRAM |
|---|---|---|---|
| SwiGLU baseline | 7.489 | 30.4 s | 0.90 GB |
| ARBOR (L=3) | **7.302** | 43.3 s | 0.90 GB |

Better per token, **42% slower per second** — so at matched wall-clock the baseline sees ~1.4×
more tokens. 2M tokens is barely past unigram statistics; this is a reason to run the experiment,
not a result. The wall-clock deficit is the unfused-elementwise-tree problem and is the main
engineering risk.

## Run it

```bash
uv sync
uv run python -m arbor.data --files 4        # ~1.5B tokens of FineWeb-Edu
bash scripts/sweep.sh                        # H1 depth sweep + H2 ablations
uv run python -m arbor.analyze               # verdict, incl. depth-compression ratio rho
```

`rho < 1.15` falsifies the project. `rho >= 2` is a real result. `rho >= 4` is a big one.

Sized to coexist with other GPU tenants (~2–3 GB peak); `scripts/sweep.sh` retries at smaller
micro-batch if it loses a VRAM race. For clean numbers give it the card to itself —
`ollama stop <model>` frees whatever the local models are holding.

## Layout

```
arbor/model.py        SwiGLU baseline, DendriticUnit, transformer (identical except the unit)
arbor/accounting.py   params / FLOPs / bytes-moved -- audited in one place, since this is
                      the easiest thing to get wrong in an architecture comparison
arbor/data.py         FineWeb-Edu -> uint16 shards; one pipeline, byte-identical for all variants
arbor/train.py        Muon + AdamW, chunked LM head, JSONL logging of every budget metric
arbor/analyze.py      iso-loss interpolation -> rho; refuses to extrapolate
scripts/sweep.sh      the experiment
```

## Controls

An architecture win is easy to fake, so the sweep runs the alternatives that would explain a win
away:

- `flat` — Wu et al.'s `Σ σ(w·x)`, the published *null* result, reproduced in our setting first
- `additive` — ARBOR with `μ ≡ 0`: isolates hierarchy from multiplicativity
- `notree` — `L=0`: isolates the branch nonlinearity alone
- `tree-depth` sweep `L ∈ {1,2,3,4,5}`
- the optional per-branch temporal conv is granted to **both** arms or neither

## References

See [DESIGN.md](DESIGN.md#6-references) — Beniaguev/London/Segev (Neuron 2021), Aizenbud et al.
(PNAS 2026), Wu et al. (Patterns 2026).
