"""FineWeb-Edu -> uint16 token shards.

One pipeline, byte-identical tokens and order for every architecture variant. The val
split is carved off first and never touched by training.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
REPO = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample/10BT"
SHARD_TOKENS = 50_000_000
EOT = 50256  # gpt2 <|endoftext|>


def _encoder():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return lambda s: enc.encode_ordinary(s)


def build(n_files: int, out: Path = DATA, val_tokens: int = 5_000_000):
    from huggingface_hub import list_repo_files, hf_hub_download
    import pyarrow.parquet as pq
    from tqdm import tqdm

    out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                   if f.startswith(f"{SUBSET}/") and f.endswith(".parquet"))
    assert files, "no parquet files found"
    files = files[:n_files]
    print(f"{len(files)} parquet files from {REPO}/{SUBSET}")

    encode = _encoder()
    buf = np.empty(SHARD_TOKENS, dtype=np.uint16)
    n = 0
    shard = 0
    split = "val"                              # first shard is the held-out val set
    limit = val_tokens

    def flush(final=False):
        nonlocal n, shard, split, limit
        path = out / (f"val.bin" if split == "val" else f"train_{shard:04d}.bin")
        buf[:n].tofile(path)
        print(f"  wrote {path.name}: {n:,} tokens")
        if split == "val":
            split, limit = "train", SHARD_TOKENS
        else:
            shard += 1
        n = 0

    for fname in files:
        p = hf_hub_download(REPO, fname, repo_type="dataset")
        tbl = pq.read_table(p, columns=["text"])
        for chunk in tqdm(tbl.column("text").to_pylist(), desc=Path(fname).name, leave=False):
            toks = encode(chunk)
            toks.append(EOT)
            for t in toks:
                buf[n] = t
                n += 1
                if n >= limit:
                    flush()
        del tbl
    if n:
        flush(final=True)
    print("done")


class TokenLoader:
    """Deterministic sequential loader over memmapped shards."""

    def __init__(self, pattern: str, batch: int, ctx: int, data_dir: Path = DATA):
        self.paths = sorted(data_dir.glob(pattern))
        assert self.paths, f"no shards matching {pattern} in {data_dir} -- run `python -m arbor.data`"
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.paths]
        self.batch, self.ctx = batch, ctx
        self.reset()

    @property
    def total_tokens(self):
        return sum(a.shape[0] for a in self.arrays)

    def reset(self):
        self.si, self.pos, self.epochs = 0, 0, 0

    def next(self, device="cuda"):
        import torch
        need = self.batch * self.ctx + 1
        a = self.arrays[self.si]
        if self.pos + need > a.shape[0]:
            self.si = (self.si + 1) % len(self.arrays)
            if self.si == 0:
                self.epochs += 1
            self.pos = 0
            a = self.arrays[self.si]
        chunk = np.asarray(a[self.pos:self.pos + need], dtype=np.int64)
        self.pos += self.batch * self.ctx
        t = torch.from_numpy(chunk)
        x = t[:-1].view(self.batch, self.ctx).to(device, non_blocking=True)
        y = t[1:].view(self.batch, self.ctx).to(device, non_blocking=True)
        return x, y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=3, help="number of FineWeb-Edu parquet files")
    ap.add_argument("--val-tokens", type=int, default=5_000_000)
    a = ap.parse_args()
    build(a.files, val_tokens=a.val_tokens)
