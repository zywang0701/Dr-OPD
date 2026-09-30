#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Slice one difficulty band out of DeepMath-103K, column for column identical to deepmath-level6-train.parquet.
Usage: python make_deepmath_level.py <difficulty L> <out.parquet> [DeepMath-103K dir, default ./DeepMath-103K]
    keeps L <= difficulty < L+1 (e.g. L=3 keeps the 3.0 and 3.5 bands); prompt = question + "\\nPlease reason step by step, and put your final
    answer within \\boxed{}." (a single user message), ground_truth = final_answer, data_source = DeepMath-103K, exactly as in the level-6 file.
    The directory needs data/train-*.parquet; if absent, the data/ folder of zwhe99/DeepMath-103K is downloaded (about 2.1 GB).
    Sanity check: re-deriving the difficulty >= 6 rows the same way must reproduce at least 95% of the shipped level-6 file, otherwise nothing is written.
    Writes a .tmp file first, then renames it."""
import collections
import glob
import os
import socket
import sys

import pyarrow as pa
import pyarrow.parquet as pq

L = float(sys.argv[1])
out = sys.argv[2]
src = sys.argv[3] if len(sys.argv) > 3 else "/tmp/DeepMath-103K"
REF = os.environ.get("DM_REF", "/tmp/Dropd/datasets/deepmath-level6-train.parquet")
SUF = "\nPlease reason step by step, and put your final answer within \\boxed{}."

files = sorted(glob.glob(os.path.join(src, "data", "train-*.parquet")))
if not files:
    _o = socket.getaddrinfo
    socket.getaddrinfo = lambda h, p, f=0, t=0, pr=0, fl=0: _o(h, p, socket.AF_INET, t, pr, fl)   # force IPv4
    for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(k, None)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    from huggingface_hub import snapshot_download
    print("no %s/data/train-*.parquet found locally, downloading zwhe99/DeepMath-103K (about 2.1 GB)" % src, flush=True)
    snapshot_download("zwhe99/DeepMath-103K", repo_type="dataset", local_dir=src, allow_patterns=["data/train-*.parquet"], max_workers=4)
    files = sorted(glob.glob(os.path.join(src, "data", "train-*.parquet")))
assert len(files) == 10, "DeepMath-103K should have 10 shards, found %d" % len(files)
t = pa.concat_tables([pq.read_table(f, columns=["question", "final_answer", "difficulty"]) for f in files])
rows = list(zip(t.column(0).to_pylist(), t.column(1).to_pylist(), t.column(2).to_pylist()))
dist = collections.Counter(d for _, _, d in rows)
print("DeepMath-103K: %d problems, difficulty histogram: %s" % (len(rows), ", ".join("%g:%d" % (k, v) for k, v in sorted(dist.items()))), flush=True)

R = pq.read_table(REF)
refmap = {p[0]["content"]: rm["ground_truth"] for p, rm in zip(R.column("prompt").to_pylist(), R.column("reward_model").to_pylist())}
hit = sum(1 for q, a, d in rows if d >= 6 and refmap.get(q + SUF) == a)
print("sanity check: re-derived difficulty>=6 rows reproduce %d of the %d rows in the shipped level-6 file (%.1f%%)" % (hit, R.num_rows, 100.0 * hit / R.num_rows), flush=True)
if hit < 0.95 * R.num_rows:
    sys.exit("does not reproduce the shipped level-6 file, nothing written (question or answer formatting differs)")

sel = [(q, a) for q, a, d in rows if L <= d < L + 1]
recs = [{"data_source": "DeepMath-103K", "prompt": [{"content": q + SUF, "role": "user"}], "ability": "math",
         "reward_model": {"ground_truth": a, "style": "rule"}, "extra_info": {"index": k, "split": "train"}}
        for k, (q, a) in enumerate(sel)]
T = pa.Table.from_pylist(recs, schema=R.schema.remove_metadata())
pq.write_table(T, out + ".tmp", row_group_size=1024)
assert pq.ParquetFile(out + ".tmp").metadata.num_rows == len(recs)
os.replace(out + ".tmp", out)
print("wrote %s: difficulty %g <= d < %g, %d problems" % (os.path.basename(out), L, L + 1, len(recs)), flush=True)
