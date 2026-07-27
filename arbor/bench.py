"""Measure what H1 is actually about: sequential-depth latency.

The depth-compression claim only pays off if fewer, deeper layers really are faster per
token. Network depth is serial (L dependent kernel launches, each round-tripping the
residual stream through HBM); the dendritic tree is internally parallel and log-depth.
This script measures that instead of assuming it.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from .accounting import flops_per_token, solve_branch_mult, summarize
from .model import ARBOR, Config


@torch.no_grad()
def latency(model, batch: int, seq: int, iters: int = 50, warmup: int = 10, device="cuda"):
    x = torch.randint(0, 50000, (batch, seq), device=device)
    for _ in range(warmup):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--tree-depth", type=int, default=3)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 4, 6, 8, 12, 16])
    ap.add_argument("--decode-batch", type=int, default=1, help="batch for the serial-decode measurement")
    ap.add_argument("--prefill-batch", type=int, default=8)
    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    a = ap.parse_args()

    rows = []
    for unit in ("swiglu", "arbor"):
        for L in a.layers:
            cfg = Config(unit=unit, n_layer=L, dim=a.dim, n_head=a.n_head, ctx=a.ctx,
                         tree_depth=a.tree_depth)
            if unit == "arbor":
                tgt = flops_per_token(Config(unit="swiglu", n_layer=L, dim=a.dim,
                                             n_head=a.n_head, ctx=a.ctx))["total"]
                cfg.branch_mult = solve_branch_mult(cfg, tgt)
                cfg.__post_init__()
            m = ARBOR(cfg).to("cuda").eval()
            if a.compile:
                m = torch.compile(m)
            try:
                dec = latency(m, a.decode_batch, 1, iters=100)      # one token: pure serial depth
                pre = latency(m, a.prefill_batch, a.ctx, iters=20)  # throughput-bound
            except torch.OutOfMemoryError:
                print(f"OOM at {unit} L={L}, skipping"); continue
            s = summarize(cfg)
            rows.append({
                "unit": unit, "n_layer": L,
                "params_m": s["params_total"] / 1e6,
                "gflops_per_token": s["flops_per_token_fwd"] / 1e9,
                "decode_ms": dec * 1e3,
                "prefill_tok_per_s": a.prefill_batch * a.ctx / pre,
                "unit_depth": s["unit_nonlinear_depth"],
                "downproj_floats": s["down_proj_input_floats"],
            })
            print(f"{unit:7s} L={L:2d} decode {dec*1e3:6.2f} ms  "
                  f"prefill {a.prefill_batch*a.ctx/pre/1e3:7.1f} k tok/s  "
                  f"{s['params_total']/1e6:5.1f}M params", flush=True)
            del m
            torch.cuda.empty_cache()

    with open("runs/latency.json", "w") as f:
        json.dump(rows, f, indent=1)

    print("\nserial-depth cost (single-token decode, the latency floor):")
    for u in ("swiglu", "arbor"):
        r = [x for x in rows if x["unit"] == u]
        if len(r) >= 2:
            per_layer = (r[-1]["decode_ms"] - r[0]["decode_ms"]) / (r[-1]["n_layer"] - r[0]["n_layer"])
            print(f"  {u:7s} {per_layer:.3f} ms per layer")
    print("\n-> combine with rho from `python -m arbor.analyze`: iso-quality latency win")
    print("   = (rho * arbor_layers * arbor_ms_per_layer) vs (arbor_layers * arbor_ms_per_layer)")


if __name__ == "__main__":
    main()
