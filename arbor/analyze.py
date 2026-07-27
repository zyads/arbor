"""Read runs/*.jsonl and produce the verdict.

The headline number is the depth-compression ratio rho: how many baseline layers are
needed to reach the val loss an ARBOR of L layers reaches. rho is obtained by linear
interpolation on the baseline's (layers -> best val loss) curve, so it is only reported
where the ARBOR loss actually falls inside the baseline's measured range -- extrapolated
ratios are reported as bounds, never as numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"


def load(runs_dir: Path = RUNS) -> dict:
    out = {}
    for p in sorted(runs_dir.glob("*.jsonl")):
        header, steps, final = None, [], None
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["type"] == "header":
                header = r
            elif r["type"] == "step":
                steps.append(r)
            elif r["type"] == "final":
                final = r
        if header and steps:
            out[p.stem] = {"header": header, "steps": steps, "final": final,
                           "best_val": min(s["val_loss"] for s in steps),
                           "complete": final is not None}
    return out


def _interp_layers(curve: list[tuple[int, float]], target: float):
    """curve: sorted [(layers, best_val)]. Return baseline layers needed to hit target."""
    curve = sorted(curve)
    losses = [v for _, v in curve]
    if target > losses[0]:
        return ("<", curve[0][0])          # even the shallowest baseline is better
    if target < losses[-1]:
        return (">", curve[-1][0])         # deepest measured baseline still worse
    for (l0, v0), (l1, v1) in zip(curve, curve[1:]):
        if v0 >= target >= v1:
            if v0 == v1:
                return ("=", l0)
            f = (v0 - target) / (v0 - v1)
            return ("=", l0 + f * (l1 - l0))
    return ("?", float("nan"))


def report(runs: dict) -> str:
    L = []
    add = L.append

    add("=" * 78)
    add("ARBOR phase-1 results")
    add("=" * 78)

    incomplete = [k for k, v in runs.items() if not v["complete"]]
    if incomplete:
        add(f"NOTE: {len(incomplete)} run(s) still in progress, reported at last eval: "
            + ", ".join(sorted(incomplete)))
    add("")

    add(f"{'run':28s} {'params':>9s} {'GF/tok':>7s} {'val':>7s} {'tok/s':>7s} {'GB':>5s} {'udep':>5s}")
    add("-" * 78)
    for name, r in sorted(runs.items()):
        h, s = r["header"], r["steps"][-1]
        sm = h["summary"]
        add(f"{name:28s} {sm['params_total']/1e6:8.1f}M "
            f"{sm['flops_per_token_fwd']/1e9:7.3f} {r['best_val']:7.4f} "
            f"{s['tokens_per_s']:7.0f} {s['peak_vram_gb']:5.2f} {sm['unit_nonlinear_depth']:5d}")
    add("")

    # ---- H1 -------------------------------------------------------------------
    base = sorted((runs[k]["header"]["config"]["n_layer"], runs[k]["best_val"])
                  for k in runs if k.startswith("h1_swiglu_"))
    arb = sorted((runs[k]["config"]["n_layer"] if "config" in runs[k]
                  else runs[k]["header"]["config"]["n_layer"], runs[k]["best_val"], k)
                 for k in runs if k.startswith("h1_arbor_"))

    add("H1 -- DEPTH COMPRESSION")
    add("-" * 78)
    if len(base) < 2 or not arb:
        add("  not enough runs yet (need >=2 baseline depths and >=1 arbor depth)")
    else:
        add(f"  baseline val-loss vs depth: " + ", ".join(f"L{l}={v:.4f}" for l, v in base))
        add("")
        add(f"  {'arbor':>10s} {'val':>8s} {'baseline layers to match':>26s} {'rho':>8s}")
        for l, v, name in arb:
            op, need = _interp_layers(base, v)
            if op == "=":
                add(f"  {'L'+str(l):>10s} {v:8.4f} {need:26.2f} {need/l:8.2f}")
            elif op == ">":
                add(f"  {'L'+str(l):>10s} {v:8.4f} {'>'+str(need)+' (off scale)':>26s} {'>'+f'{need/l:.2f}':>8s}")
            else:
                add(f"  {'L'+str(l):>10s} {v:8.4f} {'<'+str(need):>26s} {'<'+f'{need/l:.2f}':>8s}")
        add("")
        rhos = [need / l for l, v, _ in arb for op, need in [_interp_layers(base, v)] if op == "="]
        if rhos:
            m = sum(rhos) / len(rhos)
            add(f"  mean rho = {m:.2f}")
            verdict = ("FALSIFIED: depth in the unit does not substitute for depth in the network"
                       if m < 1.15 else
                       "WEAK: real but small compression" if m < 2 else
                       "STRONG: dendritic depth substitutes for network depth")
            add(f"  VERDICT: {verdict}")
    add("")

    # ---- H2 -------------------------------------------------------------------
    add("H2 -- WHAT IS THE TREE ACTUALLY CONTRIBUTING (fixed depth, matched FLOPs)")
    add("-" * 78)
    ctl = {k: runs[k]["best_val"] for k in runs if k.startswith("h2_")}
    ref = {k: runs[k]["best_val"] for k in runs if k.startswith("h1_swiglu_L8")}
    for k, v in sorted(ref.items()):
        add(f"  {k:28s} {v:8.4f}   (baseline)")
    for k, v in sorted(ctl.items(), key=lambda kv: kv[1]):
        d = f"{v - list(ref.values())[0]:+.4f}" if ref else ""
        add(f"  {k:28s} {v:8.4f}   {d} vs baseline")
    add("")

    # ---- communication --------------------------------------------------------
    add("COMMUNICATION / BANDWIDTH (analytic, per token)")
    add("-" * 78)
    add(f"  {'run':28s} {'floats into out-proj':>22s} {'elementwise % of ffn':>22s}")
    for name, r in sorted(runs.items()):
        sm = r["header"]["summary"]
        add(f"  {name:28s} {sm['down_proj_input_floats']:22d} "
            f"{sm['elementwise_frac_of_ffn']*100:21.1f}%")
    return "\n".join(L)


if __name__ == "__main__":
    r = load()
    if not r:
        print("no runs found in", RUNS)
    else:
        print(report(r))
