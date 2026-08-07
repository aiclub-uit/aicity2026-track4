#!/usr/bin/env python3
"""Encode a SHARD of the PAB2026 gallery with Argus-Colqwen3.5-9b and save the multi-vector
embeddings to a shared disk cache (like the ops cache). Each worker handles a gallery index
range [--start_idx, --end_idx) aligned to --chunk boundaries; chunks are written as
cache/chunk_{globalstart:06d}.pt so disjoint workers never collide.

encode_images is query-INDEPENDENT, so the cache is reusable for ANY query set.
Retrieve later with argus_retrieve_from_cache.py.
"""
import os
import argparse, json, time
from pathlib import Path

DEFAULT_MODEL = "DataScience-UIBK/Argus-Colqwen3.5-9b-v0"

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def list_gallery(d):
    return sorted(Path(d).glob("*.jpg"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--model_name", default=DEFAULT_MODEL)
    ap.add_argument("--hf_cache", default=os.environ.get("HF_HOME", ""))
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=0, help="0 = to end")
    ap.add_argument("--chunk", type=int, default=64, help="images per cache chunk (global boundary)")
    ap.add_argument("--image_batch_size", type=int, default=4)
    ap.add_argument("--max_visual_tokens", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn_implementation", default="sdpa")
    a = ap.parse_args()

    gallery = list_gallery(a.gallery_dir)
    N = len(gallery)
    end = a.end_idx if a.end_idx > 0 else N
    end = min(end, N)
    # align start to chunk boundary (so global chunk_{start} names are consistent across workers)
    assert a.start_idx % a.chunk == 0, "start_idx must be a multiple of --chunk"
    cache = Path(a.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    log(f"gallery={N} shard=[{a.start_idx},{end}) chunk={a.chunk} cache={cache}")

    # write/verify global gallery manifest once (worker 0 or whoever gets there first; idempotent)
    manif = cache / "gallery_names.json"
    if not manif.exists():
        try:
            manif.write_text(json.dumps([p.stem for p in gallery]))
            log("wrote gallery_names.json")
        except Exception as e:
            log(f"manifest write skipped ({e})")

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[a.dtype]
    log(f"loading {a.model_name} ...")
    model = AutoModel.from_pretrained(a.model_name, trust_remote_code=True, torch_dtype=dtype,
                                      attn_implementation=a.attn_implementation, device_map="cuda",
                                      cache_dir=a.hf_cache).eval()
    processor = AutoProcessor.from_pretrained(a.model_name, trust_remote_code=True,
                                              cache_dir=a.hf_cache, max_num_visual_tokens=a.max_visual_tokens)
    dev = next(model.parameters()).device
    log(f"model loaded (device={dev})")

    def batched(xs, bs):
        for i in range(0, len(xs), bs): yield xs[i:i+bs]

    t0 = time.time(); done = 0; skipped = 0
    starts = list(range(a.start_idx, end, a.chunk))
    for si, start in enumerate(starts):
        cstop = min(start + a.chunk, end)
        cpath = cache / f"chunk_{start:06d}.pt"
        if cpath.exists():
            skipped += 1; done += 1; continue
        sub = gallery[start:cstop]
        embs = []
        with torch.no_grad():
            for sp in batched(sub, a.image_batch_size):
                imgs = [Image.open(p).convert("RGB") for p in sp]
                e = model.encode_images(processor, imgs)
                embs.extend([x.detach().to("cpu", dtype=torch.float16) if torch.is_tensor(x) else x for x in e])
        tmp = cache / f".chunk_{start:06d}.pt.tmp"
        torch.save({"doc_start": start, "names": [p.stem for p in sub], "embeddings": embs}, tmp)
        tmp.rename(cpath)  # atomic
        done += 1
        if si < 3 or si % 20 == 0:
            rate = (time.time()-t0)/max(done-skipped,1)
            left = (len(starts)-si-1)*rate/60.0
            log(f"chunk {si+1}/{len(starts)} start={start} tok={tuple(embs[0].shape) if embs else 'NA'} "
                f"{rate:.1f}s/chunk eta={left:.1f}min (skipped_existing={skipped})")
    log(f"SHARD_DONE range=[{a.start_idx},{end}) wrote={done-skipped} skipped={skipped} in {(time.time()-t0)/60:.1f}min")

if __name__ == "__main__":
    main()
