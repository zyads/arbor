"""Training loop with strict, comparable accounting.

Every run writes a JSONL of (step, tokens, train loss, val loss, wall-clock, peak VRAM,
tokens/s) plus a header with the analytic params/FLOPs. Variants see byte-identical data
in byte-identical order, so only the unit differs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn as nn

from .accounting import flops_per_token, param_breakdown, solve_branch_mult, summarize
from .data import TokenLoader
from .model import ARBOR, Config

RUNS = Path(__file__).resolve().parent.parent / "runs"


# --------------------------------------------------------------------------------------
# Muon (Keller Jordan): orthogonalized momentum SGD for 2-D hidden weights
# --------------------------------------------------------------------------------------

@torch.compile(dynamic=False)
def _newtonschulz5(G, steps: int = 5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.mT if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p.grad)
                buf = st["buf"]
                buf.lerp_(p.grad, 1 - g["momentum"])
                upd = p.grad.lerp_(buf, g["momentum"]) if g["nesterov"] else buf
                upd = _newtonschulz5(upd, g["ns_steps"])
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(upd.reshape(p.shape).to(p.dtype), alpha=-g["lr"] * scale)


def build_optimizers(model: ARBOR, lr_muon: float, lr_adam: float, wd: float):
    muon_p, adam_decay, adam_plain = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "wte" not in name and "lm_head" not in name:
            muon_p.append(p)                  # hidden matrices
        elif p.ndim >= 2:
            adam_decay.append(p)              # embedding, depthwise conv
        else:
            adam_plain.append(p)              # norms, NMDA gains/thresholds, junction params
    opts = [
        Muon(muon_p, lr=lr_muon),
        torch.optim.AdamW(
            [{"params": adam_decay, "weight_decay": wd},
             {"params": adam_plain, "weight_decay": 0.0}],
            lr=lr_adam, betas=(0.9, 0.95), eps=1e-10),
    ]
    return opts, dict(muon=len(muon_p), adam_decay=len(adam_decay), adam_plain=len(adam_plain))


def lr_scale(step, total, warmup, final_frac=0.0):
    """Warmup then linear decay (WSD-style trapezoid tail)."""
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return 1.0 - (1.0 - final_frac) * t


# --------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader: TokenLoader, iters: int, device="cuda"):
    model.eval()
    loader.reset()
    tot = 0.0
    for _ in range(iters):
        x, y = loader.next(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--unit", type=str, default="swiglu", choices=["swiglu", "arbor", "flat", "notree"])
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--ffn-mult", type=float, default=4.0)
    ap.add_argument("--branch-mult", type=float, default=8.0)
    ap.add_argument("--tree-depth", type=int, default=3)
    ap.add_argument("--conv-k", type=int, default=0)
    ap.add_argument("--no-mult", action="store_true", help="freeze the NMDA coincidence term to 0")
    ap.add_argument("--no-junction-act", action="store_true")
    ap.add_argument("--no-learned-merge", action="store_true")
    # budget
    ap.add_argument("--tokens", type=int, default=200_000_000)
    ap.add_argument("--batch-tokens", type=int, default=131_072, help="tokens per optimizer step")
    ap.add_argument("--micro-batch", type=int, default=16, help="sequences per forward")
    ap.add_argument("--match-flops-to", type=str, default="", help="'unit:n_layer:dim:ffn_mult' to FLOP-match against")
    # optim
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-adam", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = Config(
        n_layer=a.n_layer, dim=a.dim, n_head=a.n_head, ctx=a.ctx, unit=a.unit,
        ffn_mult=a.ffn_mult, branch_mult=a.branch_mult, tree_depth=a.tree_depth,
        conv_k=a.conv_k, multiplicative=not a.no_mult,
        junction_act=not a.no_junction_act, learned_merge=not a.no_learned_merge,
    )

    if a.match_flops_to:
        u, nl, dm, fm = a.match_flops_to.split(":")
        tgt = Config(unit=u, n_layer=int(nl), dim=int(dm), ffn_mult=float(fm),
                     n_head=a.n_head, ctx=a.ctx)
        target = flops_per_token(tgt)["total"]
        cfg.branch_mult = solve_branch_mult(cfg, target)
        cfg.__post_init__()
        got = flops_per_token(cfg)["total"]
        print(f"FLOP-matched to {a.match_flops_to}: target {target/1e6:.2f}M -> "
              f"{got/1e6:.2f}M ({100*(got-target)/target:+.2f}%), M={cfg.n_branches} N={cfg.n_somas}")

    device = "cuda"
    model = ARBOR(cfg).to(device)
    if a.compile:
        model = torch.compile(model)

    opts, counts = build_optimizers(model, a.lr_muon, a.lr_adam, a.wd)

    accum = a.batch_tokens // (a.micro_batch * a.ctx)
    assert accum >= 1, "batch-tokens too small for micro-batch*ctx"
    total_steps = a.tokens // a.batch_tokens

    train = TokenLoader("train_*.bin", a.micro_batch, a.ctx)
    val = TokenLoader("val.bin", a.micro_batch, a.ctx)

    summ = summarize(cfg)
    pb = param_breakdown(cfg)
    fpt = flops_per_token(cfg)
    RUNS.mkdir(exist_ok=True)
    logf = open(RUNS / f"{a.name}.jsonl", "w")
    header = {
        "type": "header", "name": a.name, "argv": vars(a), "config": cfg.to_dict(),
        "summary": summ, "params": pb, "flops_per_token": fpt,
        "param_groups": counts, "total_steps": total_steps, "grad_accum": accum,
        "train_tokens_available": train.total_tokens,
        "train_flops_estimate": 3 * fpt["total"] * a.tokens,
    }
    logf.write(json.dumps(header) + "\n"); logf.flush()
    print(json.dumps({k: header[k] for k in ("name", "summary", "total_steps", "grad_accum")}, indent=1))
    if train.total_tokens < a.tokens:
        print(f"WARNING: only {train.total_tokens/1e6:.0f}M train tokens available for a "
              f"{a.tokens/1e6:.0f}M-token run -- will repeat data {a.tokens/train.total_tokens:.2f}x")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tokens_seen = 0
    best_val = float("inf")

    for step in range(total_steps):
        s = lr_scale(step, total_steps, a.warmup)
        for opt, base in zip(opts, (a.lr_muon, a.lr_adam)):
            for g in opt.param_groups:
                g["lr"] = base * s

        tl = 0.0
        for _ in range(accum):
            x, y = train.next(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / accum).backward()
            tl += loss.item() / accum
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        tokens_seen += a.batch_tokens

        if step % a.eval_every == 0 or step == total_steps - 1:
            torch.cuda.synchronize()
            el = time.perf_counter() - t0
            vl = evaluate(model, val, a.eval_iters, device)
            best_val = min(best_val, vl)
            rec = {
                "type": "step", "step": step, "tokens": tokens_seen,
                "train_loss": tl, "val_loss": vl, "best_val": best_val,
                "elapsed_s": el, "tokens_per_s": tokens_seen / el,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
                "lr_scale": s,
                "train_flops": 3 * fpt["total"] * tokens_seen,
            }
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(f"step {step:5d}/{total_steps} tok {tokens_seen/1e6:7.1f}M "
                  f"train {tl:.4f} val {vl:.4f} {rec['tokens_per_s']:.0f} tok/s "
                  f"{rec['peak_vram_gb']:.2f}GB {el/60:.1f}min", flush=True)

    torch.cuda.synchronize()
    final = {
        "type": "final", "best_val": best_val,
        "wall_clock_s": time.perf_counter() - t0,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "tokens": tokens_seen,
    }
    logf.write(json.dumps(final) + "\n"); logf.close()
    print("FINAL", json.dumps(final))


if __name__ == "__main__":
    main()
