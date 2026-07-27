"""Honest accounting: params, FLOPs/token, and bytes moved.

Every claim in this repo is per-unit-of-one-of-these. Getting the accounting wrong is
the single easiest way to fake an architecture win, so it lives in one audited file.
"""

from __future__ import annotations

from .model import Config

# 1 multiply-accumulate = 2 FLOPs.
MAC = 2


def param_breakdown(cfg: Config) -> dict:
    d, V, L = cfg.dim, cfg.vocab_size, cfg.n_layer
    attn = 3 * d * d + d * d                       # qkv + proj
    if cfg.unit == "swiglu":
        h = cfg.d_ff
        ffn = 3 * d * h
        ffn_conv = h * cfg.conv_k
    else:
        M, N, td = cfg.n_branches, cfg.n_somas, cfg.effective_tree_depth
        ffn = d * M + d * N
        # per-branch alpha/theta + per-junction gamma/delta/mu/alpha'/theta'/gain
        tree = 2 * M + 6 * (M - N)                 # sum over levels of 6*(M/2^(l+1))
        ffn += tree
        ffn_conv = M * cfg.conv_k
    per_layer = attn + ffn + ffn_conv
    embed = V * d                                  # tied with lm_head
    return {
        "embedding": embed,
        "per_layer": per_layer,
        "attn_total": attn * L,
        "ffn_total": (ffn + ffn_conv) * L,
        "non_embedding": per_layer * L,
        "total": embed + per_layer * L,
    }


def flops_per_token(cfg: Config, ctx: int | None = None) -> dict:
    """Forward FLOPs for one token. Training ~= 3x (fwd + bwd)."""
    d, V, L = cfg.dim, cfg.vocab_size, cfg.n_layer
    T = ctx or cfg.ctx

    qkv = MAC * d * 3 * d
    proj = MAC * d * d
    # causal attention, averaged over positions: QK^T and AV each MAC*d*(T/2)
    scores = MAC * 2 * d * (T / 2)
    attn = qkv + proj + scores

    if cfg.unit == "swiglu":
        h = cfg.d_ff
        ffn_gemm = MAC * 3 * d * h
        ffn_elem = 2 * h + h * MAC * cfg.conv_k    # silu + mul, plus conv
        tree_gemm_out = MAC * d * h
    else:
        M, N = cfg.n_branches, cfg.n_somas
        ffn_gemm = MAC * d * M + MAC * d * N
        # branch NMDA ~4 ops/branch; each junction ~8 ops; conv
        n_junc = M - N
        ffn_elem = 4 * M + 8 * n_junc + M * MAC * cfg.conv_k
        tree_gemm_out = MAC * d * N

    ffn = ffn_gemm + ffn_elem
    head = MAC * d * V

    return {
        "attn": attn * L,
        "ffn": ffn * L,
        "ffn_elementwise": ffn_elem * L,
        "head": head,
        "total": (attn + ffn) * L + head,
        # the tensor crossing HBM into the output projection -- the communication metric
        "down_proj_input_floats": (cfg.d_ff if cfg.unit == "swiglu" else cfg.n_somas) * L,
        "out_gemm_flops": tree_gemm_out * L,
    }


def summarize(cfg: Config) -> dict:
    p = param_breakdown(cfg)
    f = flops_per_token(cfg)
    return {
        "unit": cfg.unit,
        "n_layer": cfg.n_layer,
        "dim": cfg.dim,
        "params_total": p["total"],
        "params_non_embedding": p["non_embedding"],
        "flops_per_token_fwd": f["total"],
        "ffn_flops_frac": f["ffn"] / f["total"],
        "elementwise_frac_of_ffn": f["ffn_elementwise"] / max(f["ffn"], 1),
        "down_proj_input_floats": f["down_proj_input_floats"],
        "unit_nonlinear_depth": 2 if cfg.unit == "swiglu" else cfg.effective_tree_depth + 2,
        "network_x_unit_depth": cfg.n_layer * (2 if cfg.unit == "swiglu" else cfg.effective_tree_depth + 2),
    }


def match_flops(target: Config, cand: Config, tol: float = 0.02) -> bool:
    a = flops_per_token(target)["total"]
    b = flops_per_token(cand)["total"]
    return abs(a - b) / a <= tol


def solve_branch_mult(cfg: Config, target_flops: float, lo: float = 0.5, hi: float = 64.0) -> float:
    """Find branch_mult giving ARBOR the same FLOPs/token as a target budget."""
    import copy
    for _ in range(60):
        mid = (lo + hi) / 2
        c = copy.deepcopy(cfg)
        c.branch_mult = mid
        # snap M to a multiple of 2**L * 64 so the GEMMs stay tensor-core friendly
        step = (2 ** c.effective_tree_depth) * 64
        M = max(step, round(mid * c.dim / step) * step)
        c.branch_mult = M / c.dim
        if flops_per_token(c)["total"] < target_flops:
            lo = mid
        else:
            hi = mid
    step = (2 ** cfg.effective_tree_depth) * 64
    M = max(step, round(((lo + hi) / 2) * cfg.dim / step) * step)
    return M / cfg.dim
