#!/usr/bin/env python3
"""Run Nemotron ColEmbed late-interaction retrieval on JSONL inputs.

Input rows use the same simple schema as the Qwen2B benchmark:
  {"id": "q1", "text": "..."}
  {"id": "d1", "image": "/path/to/image.jpg"}

Nemotron ColEmbed outputs multi-vector embeddings, so this script streams
document chunks and keeps only the top-k scores for each query.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_IMG_POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("IMG_LOAD_WORKERS", "16")))

torch = None
tqdm = None
AutoModel = None
load_image = None


DEFAULT_MODEL = "nvidia/nemotron-colembed-vl-8b-v2"
DEFAULT_QUERY_INSTRUCTION = (
    "person image retrieval:"
)


def load_runtime_modules():
    global torch, tqdm, AutoModel, load_image
    if torch is not None:
        return
    print("Importing torch/transformers runtime modules...", flush=True)
    import torch as torch_module
    from tqdm import tqdm as tqdm_module
    from transformers import AutoModel as auto_model_cls
    from transformers.image_utils import load_image as load_image_fn

    torch = torch_module
    tqdm = tqdm_module
    AutoModel = auto_model_cls
    load_image = load_image_fn
    print("Imported torch/transformers runtime modules.", flush=True)


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
        yield start, items[start:start + batch_size]


def dtype_from_name(name):
    return {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_rows(path, kind):
    rows = list(iter_jsonl(path))
    if not rows:
        raise SystemExit(f"No {kind} rows loaded from {path}")
    return rows


def load_model(args):
    print(f"Loading model: {args.model_name}", flush=True)
    kwargs = {
        "trust_remote_code": True,
        "device_map": args.device_map,
    }
    dtype = dtype_from_name(args.dtype)
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    else:
        kwargs["torch_dtype"] = "auto"
    if args.attn_implementation != "auto":
        kwargs["attn_implementation"] = args.attn_implementation
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    model = AutoModel.from_pretrained(args.model_name, **kwargs).eval()
    print("Loaded model.", flush=True)
    return model


def encode_queries(model, query_rows, batch_size):
    texts = [row["text"] for row in query_rows]
    with torch.no_grad():
        return model.forward_queries(texts, batch_size=batch_size)


def encode_images(model, doc_rows, batch_size):
    paths = [row.get("image") for row in doc_rows]
    for image, row in zip(paths, doc_rows):
        if not image:
            raise ValueError(f"Document row {row.get('id')} has no image path")
    # Parallel decode preserves order (pool.map is ordered) -> embeddings byte-identical.
    images = list(_IMG_POOL.map(load_image, paths))
    with torch.no_grad():
        return model.forward_images(images, batch_size=batch_size)


def apply_query_instruction(model, query_instruction):
    if not query_instruction:
        return None

    if hasattr(model, "_get_processor"):
        processor = model._get_processor()
    else:
        processor = getattr(model, "processor", None)

    if processor is None or not hasattr(processor, "query_prefix"):
        raise RuntimeError("Could not set query instruction: model processor has no query_prefix")

    processor.query_prefix = query_instruction
    print(f"Applied query_instruction to processor.query_prefix: {processor.query_prefix}", flush=True)
    return processor.query_prefix


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def document_signature(doc_rows):
    return [
        {
            "id": row["id"],
            "image": row.get("image"),
            "text": row.get("text"),
        }
        for row in doc_rows
    ]


def doc_cache_metadata(args, doc_rows):
    return {
        "cache_version": 1,
        "model_name": args.model_name,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "device_map": args.device_map,
        "doc_encode_batch_size": args.doc_encode_batch_size,
        "doc_score_chunk_size": args.doc_score_chunk_size,
        "documents": document_signature(doc_rows),
    }


def cache_chunk_path(cache_dir, doc_start):
    return Path(cache_dir) / f"chunk_{doc_start:06d}.pt"


def expected_chunk_payload(doc_start, doc_batch):
    return {
        "doc_start": doc_start,
        "ids": [row["id"] for row in doc_batch],
        "images": [row.get("image") for row in doc_batch],
        "texts": [row.get("text") for row in doc_batch],
    }


def validate_chunk_payload(path, payload, expected):
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Document embedding cache mismatch in {path}: {key}")
    if "embeddings" not in payload:
        raise RuntimeError(f"Document embedding cache missing embeddings: {path}")


def prepare_doc_cache(cache_dir, expected_metadata, refresh=False):
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
                f"Document embedding cache metadata mismatch: {metadata_path}. "
                "Use --refresh_doc_embedding_cache or a different --doc_embedding_cache_dir."
            )
        return

    existing_chunks = list(cache_dir.glob("chunk_*.pt"))
    if existing_chunks:
        raise RuntimeError(
            f"Document embedding cache has chunks but no metadata: {cache_dir}. "
            "Use --refresh_doc_embedding_cache or a different --doc_embedding_cache_dir."
        )
    metadata_path.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")


def load_or_encode_doc_embeddings(model, doc_batch, doc_start, args, cache_dir):
    expected = expected_chunk_payload(doc_start, doc_batch)
    path = cache_chunk_path(cache_dir, doc_start)
    if path.exists():
        payload = torch_load(path)
        validate_chunk_payload(path, payload, expected)
        return payload["embeddings"]

    embeddings = encode_images(model, doc_batch, args.doc_encode_batch_size)
    embeddings_cpu = embeddings.detach().cpu()
    torch.save({**expected, "embeddings": embeddings_cpu}, path)
    del embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings_cpu


def update_topk(top_scores, top_indices, scores, doc_start, topk):
    scores = scores.detach().float().cpu()
    doc_ids = torch.arange(doc_start, doc_start + scores.shape[1], dtype=torch.long)
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


def similarity_rows(query_rows, doc_rows, top_scores, top_indices):
    rows = []
    for q_idx, query in enumerate(query_rows):
        matches = []
        for score, doc_idx in zip(top_scores[q_idx].tolist(), top_indices[q_idx].tolist()):
            doc = doc_rows[doc_idx]
            matches.append(
                {
                    "id": doc["id"],
                    "score": float(score),
                    "image": doc.get("image"),
                    "text": doc.get("text"),
                }
            )
        rows.append({"query_id": query["id"], "query": query.get("text"), "topk": matches})
    return rows


def pab_rankings(rows):
    output = []
    for row in rows:
        gallery = []
        for match in row["topk"]:
            image = match.get("image")
            gallery.append(Path(image).name if image else match["id"])
        output.append({"query_index": row["query_id"], "topk_gallery": gallery})
    return output


def main():
    parser = argparse.ArgumentParser(description="Benchmark Nemotron ColEmbed on PAB-style retrieval.")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--output_dir", default="outputs/pab2025_nemotron_colembed")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--query_batch_size", type=int, default=8)
    parser.add_argument("--query_instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--doc_encode_batch_size", type=int, default=1)
    parser.add_argument("--doc_score_chunk_size", type=int, default=8)
    parser.add_argument(
        "--doc_embedding_cache_dir",
        default=None,
        help="Directory for cached document/image embeddings. Defaults to output_dir/doc_embedding_cache.",
    )
    parser.add_argument(
        "--refresh_doc_embedding_cache",
        action="store_true",
        help="Delete existing cached document chunks in the selected cache dir before encoding.",
    )
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device_map", default="cuda")
    parser.add_argument(
        "--attn_implementation",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        default="flash_attention_2",
    )
    args = parser.parse_args()

    query_rows = load_rows(args.queries, "query")
    doc_rows = load_rows(args.documents, "document")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_embedding_cache_dir = (
        Path(args.doc_embedding_cache_dir)
        if args.doc_embedding_cache_dir
        else output_dir / "doc_embedding_cache"
    )
    prepare_doc_cache(
        doc_embedding_cache_dir,
        doc_cache_metadata(args, doc_rows),
        refresh=args.refresh_doc_embedding_cache,
    )
    print(f"Document embedding cache: {doc_embedding_cache_dir}", flush=True)

    load_runtime_modules()
    model = load_model(args)
    apply_query_instruction(model, args.query_instruction)
    query_embeddings = encode_queries(model, query_rows, args.query_batch_size)

    top_scores = None
    top_indices = None
    chunks = list(batched(doc_rows, args.doc_score_chunk_size))
    for doc_start, doc_batch in tqdm(chunks, desc="Score document chunks"):
        doc_embeddings = load_or_encode_doc_embeddings(
            model, doc_batch, doc_start, args, doc_embedding_cache_dir
        )
        with torch.no_grad():
            scores = model.get_scores(query_embeddings, doc_embeddings)
        top_scores, top_indices = update_topk(top_scores, top_indices, scores, doc_start, args.topk)
        del doc_embeddings, scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sim_rows = similarity_rows(query_rows, doc_rows, top_scores, top_indices)
    write_jsonl(output_dir / "similarities.jsonl", sim_rows)
    write_jsonl(output_dir / "rankings.jsonl", pab_rankings(sim_rows))

    metadata = {
        "model_name": args.model_name,
        "queries": len(query_rows),
        "documents": len(doc_rows),
        "topk": args.topk,
        "query_batch_size": args.query_batch_size,
        "query_instruction": args.query_instruction,
        "doc_encode_batch_size": args.doc_encode_batch_size,
        "doc_score_chunk_size": args.doc_score_chunk_size,
        "doc_embedding_cache_dir": str(doc_embedding_cache_dir),
        "refresh_doc_embedding_cache": args.refresh_doc_embedding_cache,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "rankings_path": str(output_dir / "rankings.jsonl"),
        "similarities_path": str(output_dir / "similarities.jsonl"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
