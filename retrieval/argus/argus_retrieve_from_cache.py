#!/usr/bin/env python3
"""Argus retrieval reusing a prebuilt gallery embedding cache (argus_encode_shard.py output).
Loads the model ONLY to encode queries (MoE); scoring is pure MaxSim over cached doc embeddings.
Chunk-by-chunk scoring keeps memory bounded. Output = rankings.jsonl + similarities.jsonl (same
format as retrieve_argus_full_gallery.py)."""
import os
import argparse, json, time
from pathlib import Path

DEFAULT_MODEL = "DataScience-UIBK/Argus-Colqwen3.5-9b-v0"

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def load_queries(path, limit=0):
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(text); rows = raw if isinstance(raw, (list, dict)) else None
    except json.JSONDecodeError:
        rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    items = []
    if isinstance(rows, dict):
        for qid, v in rows.items():
            items.append((qid, v.get("caption", v.get("text")) if isinstance(v, dict) else v))
    else:
        for row in rows:
            items.append((row.get("query_index", row.get("id")), row.get("caption", row.get("text"))))
    return items[:limit] if limit > 0 else items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_text", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--fallback_cache_dir", default="",
                    help="optional source cache used if a staged chunk disappears or cannot be read")
    ap.add_argument("--cache_chunk_size", type=int, default=64)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_name", default=DEFAULT_MODEL)
    ap.add_argument("--hf_cache", default=os.environ.get("HF_HOME", ""))
    ap.add_argument("--topk", type=int, default=400)
    ap.add_argument("--query_batch_size", type=int, default=8)
    ap.add_argument("--score_query_batch", type=int, default=256)
    ap.add_argument("--max_visual_tokens", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn_implementation", default="sdpa")
    ap.add_argument("--query_limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache_dir)
    fallback = Path(a.fallback_cache_dir) if a.fallback_cache_dir else None
    manifest_path = cache / "gallery_names.json"
    if not manifest_path.exists() and fallback:
        manifest_path = fallback / "gallery_names.json"
    assert manifest_path.exists(), f"missing gallery_names.json in {cache} and fallback cache"
    gallery_manifest = json.loads(manifest_path.read_text())
    expected_starts = range(0, len(gallery_manifest), a.cache_chunk_size)
    chunk_files = [cache / f"chunk_{start:06d}.pt" for start in expected_starts]
    missing = [p.name for p in chunk_files if not p.exists()]
    if missing and not fallback:
        raise FileNotFoundError(f"staged cache is missing {len(missing)} expected chunks: {missing[:8]}")
    if missing:
        fallback_missing = [name for name in missing if not (fallback / name).exists()]
        if fallback_missing:
            raise FileNotFoundError(
                f"{len(fallback_missing)} chunks absent from both staged and fallback caches: {fallback_missing[:8]}"
            )
        log(f"WARNING: {len(missing)} staged chunks missing; they will be read from {fallback}")
    queries = load_queries(a.query_text, a.query_limit)
    log(f"queries={len(queries)} cache_chunks={len(chunk_files)}")

    import torch
    from transformers import AutoModel, AutoProcessor
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[a.dtype]
    log(f"loading {a.model_name} (queries only) ...")
    model = AutoModel.from_pretrained(a.model_name, trust_remote_code=True, torch_dtype=dtype,
                                      attn_implementation=a.attn_implementation, device_map="cuda",
                                      cache_dir=a.hf_cache).eval()
    processor = AutoProcessor.from_pretrained(a.model_name, trust_remote_code=True,
                                              cache_dir=a.hf_cache, max_num_visual_tokens=a.max_visual_tokens)
    dev = next(model.parameters()).device

    def batched(xs, bs):
        for i in range(0, len(xs), bs): yield xs[i:i+bs]

    t0 = time.time(); q_emb = []
    with torch.no_grad():
        for batch in batched([t for _, t in queries], a.query_batch_size):
            q_emb.extend([e.to(dev) if torch.is_tensor(e) else e for e in model.encode_queries(processor, batch)])
    nq = len(q_emb)
    log(f"encoded {nq} queries in {time.time()-t0:.1f}s")
    qb = a.score_query_batch if a.score_query_batch > 0 else nq
    q_groups = [list(range(s, min(s+qb, nq))) for s in range(0, nq, qb)]

    doc_names, top_scores, top_idx = [], None, None
    gstart = 0; pstart = time.time()
    for ci, cf in enumerate(chunk_files):
        load_path = cf
        if not load_path.exists() and fallback:
            load_path = fallback / cf.name
        try:
            payload = torch.load(load_path, map_location="cpu", weights_only=False)
        except OSError:
            fallback_path = fallback / cf.name if fallback else None
            if not fallback_path or load_path == fallback_path or not fallback_path.exists():
                raise
            log(f"WARNING: failed to read staged {cf.name}; retrying from fallback cache")
            payload = torch.load(fallback_path, map_location="cpu", weights_only=False)
        names = payload["names"]
        expected_start = ci * a.cache_chunk_size
        expected_names = gallery_manifest[expected_start:expected_start + a.cache_chunk_size]
        if payload.get("doc_start") != expected_start or names != expected_names:
            raise ValueError(
                f"cache integrity error in {load_path}: doc_start={payload.get('doc_start')} "
                f"expected={expected_start}, names_match={names == expected_names}"
            )
        d_emb = [x.to(dev, dtype=next(model.parameters()).dtype) for x in payload["embeddings"]]
        cols = len(d_emb)
        base = len(doc_names)
        doc_names.extend(names)
        chunk_scores = torch.empty((nq, cols), dtype=torch.float32)
        with torch.no_grad():
            for grp in q_groups:
                s = processor.score([q_emb[i] for i in grp], d_emb)
                chunk_scores[grp[0]:grp[-1]+1] = s.detach().float().cpu()
        ids = torch.arange(base, base+cols, dtype=torch.long).unsqueeze(0).expand(nq, -1)
        if top_scores is None:
            k = min(a.topk, cols); top_scores, pos = torch.topk(chunk_scores, k=k, dim=1); top_idx = ids.gather(1, pos)
        else:
            ms = torch.cat([top_scores, chunk_scores], 1); mi = torch.cat([top_idx, ids], 1)
            k = min(a.topk, ms.shape[1]); top_scores, pos = torch.topk(ms, k=k, dim=1); top_idx = mi.gather(1, pos)
        del d_emb, chunk_scores
        if ci < 3 or ci % 25 == 0:
            rate = (time.time()-pstart)/(ci+1)
            log(f"chunk {ci+1}/{len(chunk_files)} rate={rate:.2f}s eta={rate*(len(chunk_files)-ci-1)/60:.1f}min")
    log(f"scoring done in {time.time()-pstart:.0f}s over {len(doc_names)} docs")
    if doc_names != gallery_manifest:
        raise ValueError(f"cache coverage mismatch: loaded {len(doc_names)} docs, expected {len(gallery_manifest)}")

    sims, ranks = [], []
    for qi, (qid, qtext) in enumerate(queries):
        matches, gal = [], []
        for s, di in zip(top_scores[qi].tolist(), top_idx[qi].tolist()):
            name = doc_names[di]; matches.append({"id": name, "image": name, "score": float(s)}); gal.append(name)
        sims.append({"query_id": qid, "query": qtext, "topk": matches})
        ranks.append({"query_index": qid, "topk_gallery": gal})
    with (out/"similarities.jsonl").open("w") as f:
        for r in sims: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    with (out/"rankings.jsonl").open("w") as f:
        for r in ranks: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    (out/"metadata.json").write_text(json.dumps({"model_name": a.model_name, "queries": len(queries),
        "gallery_images": len(doc_names), "topk": a.topk, "from_cache": str(cache)}, indent=2))
    log(f"DONE -> {out}/rankings.jsonl ({len(ranks)} queries)")

if __name__ == "__main__":
    main()
