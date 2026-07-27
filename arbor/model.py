"""ARBOR models: a strong GPT baseline and the dendritic (deep-unit) variant.

Both share everything except the per-token feedforward unit, so any difference in the
results is attributable to the unit and nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------

@dataclass
class Config:
    # shared
    vocab_size: int = 50304          # padded gpt2 vocab, multiple of 128
    n_layer: int = 8
    n_head: int = 8
    dim: int = 512
    ctx: int = 1024
    rope_base: float = 10000.0

    # which feedforward unit
    unit: str = "swiglu"             # swiglu | arbor | flat | notree

    # swiglu baseline
    ffn_mult: float = 4.0            # d_ff = ffn_mult * dim (hidden of the gated MLP)

    # arbor / dendritic unit
    branch_mult: float = 8.0         # M = branch_mult * dim  (synaptic integration sites)
    tree_depth: int = 3              # L: number of junction levels; somas N = M / 2**L
    multiplicative: bool = True      # the NMDA coincidence term mu*(u*v)
    junction_act: bool = True        # NMDA nonlinearity at each junction
    learned_merge: bool = True       # learned gamma/delta (False => Wu et al. unweighted sum)

    # dendritic time constants: per-branch causal depthwise conv.
    # Granted to BOTH arms when enabled, so it can never be the source of a win.
    conv_k: int = 0

    # recompute the elementwise tree in backward instead of storing every level
    recompute_tree: bool = True

    def __post_init__(self):
        assert self.dim % self.n_head == 0
        if self.unit in ("arbor", "flat", "notree"):
            M = self.n_branches
            assert M % (2 ** self.tree_depth) == 0, "M must be divisible by 2**tree_depth"

    @property
    def d_ff(self) -> int:
        return int(self.ffn_mult * self.dim)

    @property
    def n_branches(self) -> int:
        return int(self.branch_mult * self.dim)

    @property
    def n_somas(self) -> int:
        return self.n_branches // (2 ** self.effective_tree_depth)

    @property
    def effective_tree_depth(self) -> int:
        return 0 if self.unit == "notree" else self.tree_depth

    def to_dict(self):
        d = asdict(self)
        d.update(d_ff=self.d_ff, n_branches=self.n_branches, n_somas=self.n_somas)
        return d


# --------------------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def rope_cache(seq: int, head_dim: int, base: float, device, dtype=torch.float32):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D); cos/sin: (T_max, D//2)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    cos = cos[None, None, : x.shape[2], :].to(x.dtype)
    sin = sin[None, None, : x.shape[2], :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalDepthwiseConv(nn.Module):
    """Per-channel causal FIR filter: the unit's own short-term memory.

    Biological role: synaptic / NMDA time constants (the ~153 ms of history a real
    dendrite integrates over). Architecturally this is the standard short conv.
    """

    def __init__(self, channels: int, k: int):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(channels, channels, k, groups=channels, bias=False)
        with torch.no_grad():  # init to identity => a no-op at step 0
            self.conv.weight.zero_()
            self.conv.weight[:, :, -1] = 1.0

    def forward(self, x):            # (B, T, C)
        x = x.transpose(1, 2)        # (B, C, T)
        x = F.pad(x, (self.k - 1, 0))
        return self.conv(x).transpose(1, 2)


# --------------------------------------------------------------------------------------
# feedforward units
# --------------------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """The baseline unit. Note it is already a ONE-level dendritic tree: a single
    multiplicative junction between two linear projections. ARBOR asks whether the
    rest of the tree is worth anything."""

    def __init__(self, cfg: Config):
        super().__init__()
        d, h = cfg.dim, cfg.d_ff
        self.w_gate = nn.Linear(d, h, bias=False)
        self.w_up = nn.Linear(d, h, bias=False)
        self.w_down = nn.Linear(h, d, bias=False)
        self.conv = CausalDepthwiseConv(h, cfg.conv_k) if cfg.conv_k else None

    def forward(self, x):
        g, u = self.w_gate(x), self.w_up(x)
        if self.conv is not None:
            g = self.conv(g)
        return self.w_down(F.silu(g) * u)


# Var[SiLU(z)] for z ~ N(0,1); used to keep the tree variance-stable at init.
_SILU_VAR = 0.3546
_SILU_GAIN = 1.0 / math.sqrt(_SILU_VAR)


class DendriticUnit(nn.Module):
    """A deep neuron.

        s = W_in x                                  synaptic integration   (the one big GEMM)
        b = s * sigmoid(alpha*s - theta)            NMDA branch nonlinearity
        repeat L times:                             the dendritic tree
            u, v = b.chunk(2)                       butterfly pairing (stride M/2, M/4, ...)
            z    = gamma*u + delta*v + mu*(u*v)     <- mu is the NMDA coincidence term
            b    = g * z * sigmoid(alpha'*z-theta') junction NMDA spike
        y = W_out b                                 soma

    The tree costs O(M) params and O(M) FLOPs against the GEMMs' O(M*d), so the L extra
    nonlinear levels are essentially free -- that is the whole trick a dendrite plays.

    chunk(2) pairing is a decimation-in-frequency butterfly: after L levels each soma
    aggregates 2**L branches spread at stride N across the full width, so receptive
    fields are distributed rather than local, matching how a real dendritic arbor
    samples its inputs.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.dim
        M = cfg.n_branches
        L = cfg.effective_tree_depth
        self.L = L
        self.multiplicative = cfg.multiplicative
        self.junction_act = cfg.junction_act
        self.learned_merge = cfg.learned_merge

        self.w_in = nn.Linear(d, M, bias=False)
        self.conv = CausalDepthwiseConv(M, cfg.conv_k) if cfg.conv_k else None

        # per-branch NMDA gain / threshold
        self.alpha = nn.Parameter(torch.ones(M))
        self.theta = nn.Parameter(torch.zeros(M))

        # per-level junction parameters
        gamma, delta, mu, a_j, t_j, gain = [], [], [], [], [], []
        w = M
        for _ in range(L):
            w //= 2
            # 1/sqrt(2) keeps the additive merge variance-preserving at init
            gamma.append(nn.Parameter(torch.full((w,), 1.0 / math.sqrt(2.0))))
            delta.append(nn.Parameter(torch.full((w,), 1.0 / math.sqrt(2.0))))
            mu.append(nn.Parameter(torch.zeros(w)))       # starts purely additive
            a_j.append(nn.Parameter(torch.ones(w)))
            t_j.append(nn.Parameter(torch.zeros(w)))
            gain.append(nn.Parameter(torch.full((w,), _SILU_GAIN if cfg.junction_act else 1.0)))
        self.gamma, self.delta, self.mu = nn.ParameterList(gamma), nn.ParameterList(delta), nn.ParameterList(mu)
        self.a_j, self.t_j, self.gain = nn.ParameterList(a_j), nn.ParameterList(t_j), nn.ParameterList(gain)

        self.soma_norm = RMSNorm(cfg.n_somas)
        self.w_out = nn.Linear(cfg.n_somas, d, bias=False)
        self.recompute_tree = cfg.recompute_tree

    def _tree(self, h):
        h = h * torch.sigmoid(self.alpha * h - self.theta)      # branch NMDA
        for l in range(self.L):
            u, v = h.chunk(2, dim=-1)
            if self.learned_merge:
                z = self.gamma[l] * u + self.delta[l] * v
            else:
                z = u + v                                        # Wu et al. unweighted soma sum
            if self.multiplicative:
                z = z + self.mu[l] * (u * v)                     # NMDA coincidence detection
            if self.junction_act:
                z = self.gain[l] * z * torch.sigmoid(self.a_j[l] * z - self.t_j[l])
            h = z
        return h

    def forward(self, x):
        h = self.w_in(x)
        if self.conv is not None:
            h = self.conv(h)
        # The tree is O(M) elementwise FLOPs but would otherwise stash L activation
        # tensors for backward. Recomputing it costs ~1% of the unit's FLOPs and drops
        # peak training memory to a single M-wide tensor -- so the memory comparison
        # against SwiGLU stays fair rather than being an artifact of autograd.
        if self.recompute_tree and self.training and torch.is_grad_enabled():
            h = torch.utils.checkpoint.checkpoint(self._tree, h, use_reentrant=False)
        else:
            h = self._tree(h)
        return self.w_out(self.soma_norm(h))


def make_ffn(cfg: Config) -> nn.Module:
    if cfg.unit == "swiglu":
        return SwiGLU(cfg)
    if cfg.unit in ("arbor", "flat", "notree"):
        return DendriticUnit(cfg)
    raise ValueError(f"unknown unit {cfg.unit!r}")


# --------------------------------------------------------------------------------------
# transformer
# --------------------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.hd = cfg.dim // cfg.n_head
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)                    # QK-norm
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.dim)
        self.ffn = make_ffn(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.ffn(self.n2(x))
        return x


class ARBOR(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight          # tied

        self.apply(self._init)
        # scale residual-path output projections by 1/sqrt(2*n_layer)
        for blk in self.blocks:
            s = (2 * cfg.n_layer) ** -0.5
            with torch.no_grad():
                blk.attn.proj.weight.mul_(s)
                out = blk.ffn.w_down if isinstance(blk.ffn, SwiGLU) else blk.ffn.w_out
                out.weight.mul_(s)
        self._rope = None

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if self._rope is None or self._rope[0].shape[0] < T or self._rope[0].device != idx.device:
            self._rope = rope_cache(max(T, self.cfg.ctx), self.cfg.dim // self.cfg.n_head,
                                    self.cfg.rope_base, idx.device)
        cos, sin = self._rope
        x = self.wte(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm_f(x)
        if targets is None:
            return self.lm_head(x)
        # Chunked head+loss: never materialize the full (B,T,V) fp32 logits tensor, which
        # at B=16,T=512,V=50304 would alone be 1.6 GB. Keeps peak VRAM in the unit, where
        # the experiment actually is.
        xf = x.reshape(-1, x.size(-1))
        tf = targets.reshape(-1)
        n = xf.size(0)
        chunk = max(1, 8192 // 1 if n > 8192 else n)
        total = xf.new_zeros((), dtype=torch.float32)
        for i in range(0, n, chunk):
            lg = self.lm_head(xf[i:i + chunk])
            total = total + F.cross_entropy(
                lg.float(), tf[i:i + chunk], reduction="sum")
        loss = total / n
        return None, loss
