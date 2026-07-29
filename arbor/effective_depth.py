"""Effective depth: how much of a network's nominal depth is doing work?

Experiment 1 shipped a unit advertising 5 nonlinear levels whose junctions sat at 0.99
correlation with their inputs -- effective depth ~2. Nobody caught it until the whole
experiment had run and been written up. The same pathology is documented at network scale
(deep pre-LN blocks drift toward the identity; ~40% of a large model's deep layers can be
deleted with minor damage), yet no architecture paper reports it.

So: we count layers, but we do not measure whether they compute. This module measures it.

Two instruments, deliberately independent:

  1. GEOMETRIC (cheap, no data labels): per sub-block, cos(h_in, h_out) and the relative
     update norm ||f(LN(h))|| / ||h||. A block at cosine 1.0 is a rounding error.

  2. CAUSAL (expensive, decisive): lesion each sub-block (zero its contribution) and
     measure the val-loss increase. This is the ground truth the geometric proxy must be
     validated against -- a small update can still be load-bearing if it is well-aimed.

`effective_depth` is then the count of sub-blocks whose lesion delta exceeds the marginal
value of one layer read off the measured depth curve. That threshold is not arbitrary: it
is "does this block earn its keep relative to what one more layer buys in this family."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import TokenLoader
from .model import ARBOR, Config

RUNS = Path(__file__).resolve().parent.parent / "runs"


def load_run(name: str, device="cuda"):
    ck = torch.load(RUNS / f"{name}.pt", map_location=device, weights_only=False)
    cfg = Config(**{k: v for k, v in ck["config"].items()
                    if k in Config.__dataclass_fields__})
    model = ARBOR(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def geometric_profile(model, loader, iters=8, device="cuda"):
    """Per sub-block: cos(input, output) and relative update magnitude."""
    stats = []
    handles = []

    def mk(idx, kind):
        def hook(mod, inp, out):
            h = inp[0]                      # the normed input is inside; use residual stream
            stats.append((idx, kind, h, out))
        return hook

    # Instrument by re-running the block math explicitly rather than hooking, so we see
    # the true residual stream on both sides of the add.
    prof = []
    for _ in range(iters):
        x, y = loader.next(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.wte(x)
            T = x.shape[1]
            from .model import rope_cache
            cos_s, sin_s = rope_cache(T, model.cfg.dim // model.cfg.n_head,
                                      model.cfg.rope_base, device)
            for i, blk in enumerate(model.blocks):
                for kind, norm, fn in (("attn", blk.n1, lambda z: blk.attn(z, cos_s, sin_s)),
                                       ("ffn", blk.n2, blk.ffn)):
                    upd = fn(norm(h))
                    hn = h + upd
                    c = F.cosine_similarity(h.float().flatten(0, 1),
                                            hn.float().flatten(0, 1), dim=-1).mean()
                    r = (upd.float().norm(dim=-1) / h.float().norm(dim=-1).clamp_min(1e-6)).mean()
                    prof.append((i, kind, float(c), float(r)))
                    h = hn
    # average across iters
    agg = {}
    for i, kind, c, r in prof:
        agg.setdefault((i, kind), []).append((c, r))
    return [{"block": i, "kind": k,
             "cos_in_out": sum(a for a, _ in v) / len(v),
             "rel_update": sum(b for _, b in v) / len(v)}
            for (i, k), v in sorted(agg.items())]


@torch.no_grad()
def _val_loss(model, loader, iters, device="cuda"):
    loader.reset()
    tot = 0.0
    for _ in range(iters):
        x, y = loader.next(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        tot += loss.item()
    return tot / iters


@torch.no_grad()
def lesion_profile(model, loader, iters=8, device="cuda"):
    """Zero each sub-block's contribution in turn; report the val-loss increase."""
    base = _val_loss(model, loader, iters, device)
    out = []
    for i, blk in enumerate(model.blocks):
        for kind in ("attn", "ffn"):
            mod = blk.attn if kind == "attn" else blk.ffn
            orig = mod.forward
            mod.forward = (lambda *a, **k: torch.zeros_like(a[0])) if kind == "ffn" else \
                          (lambda z, *a, **k: torch.zeros_like(z))
            try:
                d = _val_loss(model, loader, iters, device) - base
            finally:
                mod.forward = orig
            out.append({"block": i, "kind": kind, "lesion_delta": d})
    return base, out


def marginal_layer_value(curve: dict[int, float], n_layer: int) -> float:
    """Nats bought by one more layer at this depth, from the measured depth curve.

    At the DEEPEST measured point there is no forward difference, so fall back to the
    backward one. Returning nan here silently made every `delta > threshold` comparison
    False and reported 0/32 effective sub-blocks for the 16-layer model -- a bug that
    looked exactly like a dramatic finding.
    """
    ks = sorted(curve)
    if len(ks) < 2:
        return float("nan")
    lo = max([k for k in ks if k <= n_layer], default=ks[0])
    hi = min([k for k in ks if k > lo], default=None)
    if hi is None:                      # at or past the deepest point: backward difference
        hi, lo = ks[-1], ks[-2]
    return (curve[lo] - curve[hi]) / (hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run names with saved .pt checkpoints")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--no-lesion", action="store_true", help="geometric profile only (fast)")
    a = ap.parse_args()

    # measured baseline depth curve, for the "does this block earn a layer's keep" threshold
    curve = {}
    for p in list(RUNS.glob("ck_swiglu_L*.jsonl")) + list(RUNS.glob("h1_swiglu_L*.jsonl")):
        if (RUNS / f"{p.stem}.NONCOMPARABLE").exists():
            continue
        st = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        steps = [x for x in st if x["type"] == "step"]
        if steps and any(x["type"] == "final" for x in st):
            n = st[0]["config"]["n_layer"]
            # prefer the ck_* (clean, uniform micro-batch) series where both exist
            if n not in curve or p.stem.startswith("ck_"):
                curve[n] = min(x["val_loss"] for x in steps)

    for name in a.runs:
        model, cfg = load_run(name)
        val = TokenLoader("val.bin", a.micro_batch, cfg.ctx)
        print(f"\n=== {name}  ({cfg.unit}, {cfg.n_layer} layers) ===")

        geo = geometric_profile(model, val, a.iters)
        print(f"  {'block':>6s} {'kind':>5s} {'cos(in,out)':>12s} {'rel update':>11s}", end="")
        les = None
        if not a.no_lesion:
            base, les = lesion_profile(model, val, a.iters)
            lk = {(d["block"], d["kind"]): d["lesion_delta"] for d in les}
            print(f" {'lesion Δ':>10s}", end="")
        print()
        for g in geo:
            row = f"  {g['block']:6d} {g['kind']:>5s} {g['cos_in_out']:12.4f} {g['rel_update']:11.4f}"
            if les is not None:
                row += f" {lk[(g['block'], g['kind'])]:10.4f}"
            print(row)

        if les is not None and len(curve) >= 2:
            mlv = marginal_layer_value(curve, cfg.n_layer)
            # a layer is 2 sub-blocks; a sub-block earns its keep at half a layer's value
            thr = mlv / 2
            import math
            if math.isnan(thr):
                raise SystemExit("threshold is nan -- depth curve too sparse to calibrate")
            n_eff = sum(1 for d in les if d["lesion_delta"] > thr)
            print(f"\n  baseline val {base:.4f} | marginal value of one layer here: "
                  f"{mlv:.4f} nats | sub-block threshold {thr:.4f}")
            print(f"  EFFECTIVE DEPTH: {n_eff}/{2*cfg.n_layer} sub-blocks earn their keep "
                  f"({n_eff/(2*cfg.n_layer)*100:.0f}% of nominal)")
            json.dump({"run": name, "geometric": geo, "lesion": les,
                       "base_val": base, "marginal_layer_value": mlv,
                       "effective_subblocks": n_eff, "nominal_subblocks": 2 * cfg.n_layer},
                      open(RUNS / f"{name}.effdepth.json", "w"), indent=1)


if __name__ == "__main__":
    main()
