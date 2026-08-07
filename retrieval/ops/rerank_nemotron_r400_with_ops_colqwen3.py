#!/usr/bin/env python3
"""Second-stage retrieval: rescore Nemotron top-K candidates with Ops-Colqwen3."""

import os
import argparse
import importlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path


DEFAULT_MODEL = "OpenSearch-AI/Ops-Colqwen3-4B"


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_runtime_dependencies():
    global torch, Image, tqdm, snapshot_download

    log("importing torch")
    import torch as torch_module

    log("importing PIL.Image")
    from PIL import Image as image_module

    log("importing tqdm")
    from tqdm import tqdm as tqdm_module

    log("importing huggingface_hub.snapshot_download")
    from huggingface_hub import snapshot_download as snapshot_download_fn

    torch = torch_module
    Image = image_module
    tqdm = tqdm_module
    snapshot_download = snapshot_download_fn
    log("runtime dependencies imported")


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "id" not in row:
                row["id"] = f"{Path(path).stem}_{line_no}"
            yield row


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def open_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def dtype_from_name(name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def resolve_ops_script_root(model_name, cache_dir, local_repo):
    if local_repo:
        return Path(local_repo)
    return Path(
        snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            allow_patterns=["scripts/*.py", "*.py"],
        )
    )


def import_ops_embedder(repo_root):
    script_path = Path(repo_root) / "scripts" / "ops_colqwen3_embedder.py"
    if not script_path.exists():
        matches = list(Path(repo_root).rglob("ops_colqwen3_embedder.py"))
        if not matches:
            raise FileNotFoundError(f"Could not find ops_colqwen3_embedder.py under {repo_root}")
        script_path = matches[0]
    sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("ops_colqwen3_embedder", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OpsColQwen3Embedder


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_queries(path):
    queries = {}
    for row in iter_jsonl(path):
        qid = row.get("query_index", row.get("id"))
        text = row.get("caption", row.get("text"))
        if qid is None or text is None:
            raise ValueError(f"Unsupported query row: {row}")
        queries[qid] = text
    return queries


def load_nemotron_heads(rankings_path, topk, query_limit=0):
    heads = {}
    order = []
    for row in iter_jsonl(rankings_path):
        qid = row["query_index"]
        order.append(qid)
        heads[qid] = [Path(name).name for name in row.get("topk_gallery", [])[:topk]]
        if query_limit > 0 and len(order) >= query_limit:
            break
    return order, heads


def unique_candidates(query_order, heads):
    seen = set()
    docs = []
    for qid in query_order:
        for name in heads[qid]:
            if name not in seen:
                seen.add(name)
                docs.append(name)
    return docs


def encode_queries(embedder, query_order, queries, batch_size):
    texts = [queries[qid] for qid in query_order]
    outputs = []
    for _, batch in tqdm(list(batched(texts, batch_size)), desc="Encode queries"):
        outputs.extend(embedder.encode_queries(batch))
    return [emb.detach().cpu() if torch.is_tensor(emb) else emb for emb in outputs]


def encode_images(embedder, image_paths):
    images = [open_rgb(path) for path in image_paths]
    embeddings = embedder.encode_images(images)
    return [emb.detach().cpu() if torch.is_tensor(emb) else emb for emb in embeddings]


def cache_metadata(args, doc_names):
    return {
        "cache_version": 1,
        "model_name": args.model_name,
        "dims": args.dims,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "image_batch_size": args.image_batch_size,
        "score_chunk_size": args.score_chunk_size,
        "candidate_source_rankings": str(Path(args.nemotron_rankings).resolve()),
        "candidate_topk": args.candidate_topk,
        "documents": doc_names,
    }


def prepare_cache(cache_dir, expected_metadata, refresh):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    if refresh:
        for path in cache_dir.glob("chunk_*.pt"):
            path.unlink()
        metadata_path.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")
        return
    if metadata_path.exists():
        current = json.loads(metadata_path.read_text(encoding="utf-8"))
        if current != expected_metadata:
            raise RuntimeError(
                f"Document cache metadata mismatch: {metadata_path}. "
                "Use --refresh_doc_embedding_cache or a different cache dir."
            )
        return
    if list(cache_dir.glob("chunk_*.pt")):
        raise RuntimeError(f"Cache has chunks but no metadata: {cache_dir}")
    metadata_path.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")


def chunk_path(cache_dir, start):
    return Path(cache_dir) / f"chunk_{start:06d}.pt"


def load_or_encode_doc_chunk(embedder, doc_names, doc_start, args, cache_dir, gallery_dir):
    path = chunk_path(cache_dir, doc_start)
    batch_names = doc_names[doc_start : doc_start + args.score_chunk_size]
    if path.exists():
        payload = torch_load(path)
        if payload.get("doc_start") != doc_start or payload.get("names") != batch_names:
            raise RuntimeError(f"Document cache chunk mismatch: {path}")
        return payload["embeddings"]

    embeddings = []
    for _, small_names in batched(batch_names, args.image_batch_size):
        paths = [gallery_dir / name for name in small_names]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing gallery images: {missing[:5]}")
        embeddings.extend(encode_images(embedder, paths))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch.save({"doc_start": doc_start, "names": batch_names, "embeddings": embeddings}, path)
    return embeddings


def build_candidate_positions(query_order, heads, doc_index):
    positions = []
    for qid in query_order:
        positions.append({doc_index[name] for name in heads[qid] if name in doc_index})
    return positions


def update_topk(top_scores, top_indices, scores, doc_start, candidate_positions, topk):
    scores = scores.detach().float().cpu()
    doc_ids = torch.arange(doc_start, doc_start + scores.shape[1], dtype=torch.long)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for q_idx, allowed in enumerate(candidate_positions):
        for local_idx, doc_id in enumerate(doc_ids.tolist()):
            if doc_id in allowed:
                mask[q_idx, local_idx] = True
    scores = scores.masked_fill(~mask, float("-inf"))
    doc_ids = doc_ids.unsqueeze(0).expand(scores.shape[0], -1)

    if top_scores is None:
        k = min(topk, scores.shape[1])
        values, positions = torch.topk(scores, k=k, dim=1)
        return values, doc_ids.gather(1, positions)

    merged_scores = torch.cat([top_scores, scores], dim=1)
    merged_indices = torch.cat([top_indices, doc_ids], dim=1)
    k = min(topk, merged_scores.shape[1])
    values, positions = torch.topk(merged_scores, k=k, dim=1)
    return values, merged_indices.gather(1, positions)


def build_outputs(query_order, queries, doc_names, top_scores, top_indices):
    similarities = []
    rankings = []
    for q_idx, qid in enumerate(query_order):
        matches = []
        gallery = []
        for score, doc_idx in zip(top_scores[q_idx].tolist(), top_indices[q_idx].tolist()):
            if score == float("-inf"):
                continue
            name = doc_names[doc_idx]
            matches.append({"id": name, "image": name, "score": float(score)})
            gallery.append(name)
        similarities.append({"query_id": qid, "query": queries.get(qid), "topk": matches})
        rankings.append({"query_index": qid, "topk_gallery": gallery})
    return similarities, rankings


def main():
    parser = argparse.ArgumentParser(description="Ops-Colqwen3 second-stage retrieval over Nemotron top-K.")
    parser.add_argument("--nemotron_rankings", required=True)
    parser.add_argument("--query_text", required=True)
    parser.add_argument("--gallery_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--cache_dir", default=os.environ.get("HF_HOME", ""))
    parser.add_argument("--local_repo", default=None)
    parser.add_argument("--candidate_topk", type=int, default=400)
    parser.add_argument("--topk", type=int, default=400)
    parser.add_argument("--query_limit", type=int, default=0)
    parser.add_argument("--dims", type=int, default=320)
    parser.add_argument("--query_batch_size", type=int, default=8)
    parser.add_argument("--image_batch_size", type=int, default=1)
    parser.add_argument("--score_chunk_size", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", choices=["flash_attention_2", "sdpa", "eager"], default="sdpa")
    parser.add_argument("--doc_embedding_cache_dir", default=None)
    parser.add_argument("--refresh_doc_embedding_cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"loading queries from {args.query_text}")
    queries = load_queries(args.query_text)
    log(f"loaded queries={len(queries)}")
    log(f"loading Nemotron top{args.candidate_topk} from {args.nemotron_rankings}")
    query_order, heads = load_nemotron_heads(args.nemotron_rankings, args.candidate_topk, args.query_limit)
    log(f"loaded ranking rows={len(query_order)}")
    doc_names = unique_candidates(query_order, heads)
    log(f"unique candidate images={len(doc_names)}")

    load_runtime_dependencies()
    repo_root = resolve_ops_script_root(args.model_name, args.cache_dir, args.local_repo)
    OpsColQwen3Embedder = import_ops_embedder(repo_root)
    log(f"loaded OpsColQwen3Embedder from {repo_root}")

    embedder = OpsColQwen3Embedder(
        model_name=args.model_name,
        dims=args.dims,
        dtype=dtype_from_name(args.dtype),
        attn_implementation=args.attn_implementation,
    )
    log(f"model loaded: {args.model_name} dims={args.dims}")

    gallery_dir = Path(args.gallery_dir)
    doc_index = {name: idx for idx, name in enumerate(doc_names)}
    candidate_positions = build_candidate_positions(query_order, heads, doc_index)
    doc_cache_dir = Path(args.doc_embedding_cache_dir or (output_dir / "doc_embedding_cache"))
    prepare_cache(doc_cache_dir, cache_metadata(args, doc_names), args.refresh_doc_embedding_cache)

    with torch.no_grad():
        query_embeddings = encode_queries(embedder, query_order, queries, args.query_batch_size)

    # ---- Phase 1: encode every unique candidate image once -> cache (the one-time cost).
    #      No scoring here, so per-chunk time is just the image encode (reuses existing chunks).
    for doc_start in tqdm(range(0, len(doc_names), args.score_chunk_size), desc="Encode candidate chunks"):
        load_or_encode_doc_chunk(embedder, doc_names, doc_start, args, doc_cache_dir, gallery_dir)

    # ---- Phase 2: load cached doc embeddings indexed by candidate position.
    doc_embs = [None] * len(doc_names)
    for doc_start in tqdm(range(0, len(doc_names), args.score_chunk_size), desc="Load cached embeddings"):
        payload = torch_load(chunk_path(doc_cache_dir, doc_start))
        for offset, emb in enumerate(payload["embeddings"]):
            doc_embs[doc_start + offset] = emb

    # ---- Phase 3: rerank each query over ONLY its own Nemotron top-K candidates (no cross-product,
    #      no per-chunk empty_cache). 1978 x ~400 MaxSim instead of 1978 x 33,312.
    similarities = []
    rankings = []
    for q_idx, qid in enumerate(tqdm(query_order, desc="Rerank per query")):
        cand_names = [name for name in heads[qid] if name in doc_index]
        if not cand_names:
            similarities.append({"query_id": qid, "query": queries.get(qid), "topk": []})
            rankings.append({"query_index": qid, "topk_gallery": []})
            continue
        cand_embs = [doc_embs[doc_index[name]] for name in cand_names]
        with torch.no_grad():
            scores = embedder.compute_scores([query_embeddings[q_idx]], cand_embs)
        scores = scores.detach().float().cpu().reshape(-1)
        k = min(args.topk, scores.numel())
        values, positions = torch.topk(scores, k=k)
        matches = []
        gallery = []
        for score, pos in zip(values.tolist(), positions.tolist()):
            name = cand_names[pos]
            matches.append({"id": name, "image": name, "score": float(score)})
            gallery.append(name)
        similarities.append({"query_id": qid, "query": queries.get(qid), "topk": matches})
        rankings.append({"query_index": qid, "topk_gallery": gallery})
    top_scores = None
    top_indices = None
    write_jsonl(output_dir / "similarities.jsonl", similarities)
    write_jsonl(output_dir / "rankings.jsonl", rankings)
    torch.save(
        {
            "query_ids": query_order,
            "candidate_images": doc_names,
            "query_embeddings": query_embeddings,
            "top_scores": top_scores,
            "top_indices": top_indices,
        },
        output_dir / "retrieval_state.pt",
    )
    metadata = {
        "model_name": args.model_name,
        "nemotron_rankings": str(Path(args.nemotron_rankings).resolve()),
        "query_text": str(Path(args.query_text).resolve()),
        "gallery_dir": str(gallery_dir.resolve()),
        "queries": len(query_order),
        "unique_candidate_images": len(doc_names),
        "candidate_topk": args.candidate_topk,
        "topk": args.topk,
        "dims": args.dims,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "query_batch_size": args.query_batch_size,
        "image_batch_size": args.image_batch_size,
        "score_chunk_size": args.score_chunk_size,
        "rankings_path": str(output_dir / "rankings.jsonl"),
        "similarities_path": str(output_dir / "similarities.jsonl"),
        "doc_embedding_cache_dir": str(doc_cache_dir.resolve()),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
