# PAB2026: running the pipeline end to end

How to apply the pipeline to PAB2026 (1978 queries, 36,773 gallery images): the
bundled model outputs, a checksum-verified CPU derivation of the final ranking
from them, and the GPU commands that regenerate the model stages.

All paths below are relative to the repository root; run the commands from there.

The pipeline (full definition in [PIPELINE.md](PIPELINE.md)):

```
three retrievers over the full gallery (Argus-Colqwen3.5-9b, Nemotron ColEmbed 8B v2, Ops-Colqwen3-4B)
        |
RRF (k=60) fusion  ->  top-200 candidate pool
        |
Qwen3-VL-Reranker-8B on the pool top-50
        |
blend (lambda=0.5, top-20)  ->  SCA (tau=0.5)  ->  slack-arc component assignment (topn=2, delta=1.0)
        |
export top-10  ->  answer.zip
```

## 1. Re-derive the ranking without a GPU (minutes)

Every stage between the model runs is deterministic. Given the bundled model
outputs, this re-derives all intermediates and the final answer and md5-checks
each against `expected/`:

```bash
DATASET_DIR=/path/to/PAB2026/name-masked_test-set bash verify.sh
```

`DATASET_DIR` must contain `query_index.txt` (the gallery is not needed for
verification). Expected output ends with `ALL CHECKS PASSED`.

## 2. Re-run the model stages (GPU)

Model stages are floating-point GPU runs; different hardware or library builds
can reorder near-tied candidates, so exact score equality is not guaranteed;
the bundled artifacts are the exact outputs of the runs described below,
produced with these environments.

### Environments

| stage | key pins |
|---|---|
| Argus + Nemotron retrieval, Qwen reranker | Python 3.11, `torch 2.11.0+cu128`, `transformers 5.8.0`, `sentence-transformers 5.4.1`, `qwen-vl-utils`, sdpa attention |
| Ops-Colqwen3 ranker | Python 3.11, `torch 2.10.0+cu129`, `transformers 4.57.6` |

GPUs used: NVIDIA A100-80GB (Nemotron, reranker) and L40 (Argus
encode/retrieve, Ops). Set `HF_HOME` to a writable model cache.

### 2.1 Retriever inputs

All stages use the official query captions as-is. Input files for the
retrievers:

```bash
python - <<'EOF'
import json, os
D = os.environ["DATASET_DIR"]
text = json.load(open(os.path.join(D, "query_text.json")))
order = [l.strip() for l in open(os.path.join(D, "query_index.txt")) if l.strip()]
with open("work/retrieval_queries.jsonl", "w") as f:
    for q in order:
        f.write(json.dumps({"id": q, "text": text[q]}) + "\n")
gal = os.path.join(D, "gallery")
with open("work/retrieval_documents.jsonl", "w") as f:
    for name in sorted(os.listdir(gal)):
        f.write(json.dumps({"id": name, "image": os.path.join(gal, name)}) + "\n")
EOF
```

### 2.2 Argus-Colqwen3.5-9b (encode once, then retrieve)

```bash
python retrieval/argus/argus_encode_shard.py \
  --gallery_dir $DATASET_DIR/gallery --cache_dir work/argus_cache \
  --start_idx 0 --end_idx 0 --chunk 64 --image_batch_size 4 \
  --max_visual_tokens 2048 --dtype bfloat16 --attn_implementation sdpa

python retrieval/argus/argus_retrieve_from_cache.py \
  --query_text work/retrieval_queries.jsonl --cache_dir work/argus_cache \
  --output_dir work/retr_argus --cache_chunk_size 64 \
  --topk 400 --query_batch_size 8 --score_query_batch 256 \
  --max_visual_tokens 2048 --dtype bfloat16 --attn_implementation sdpa --query_limit 0
```

### 2.3 Nemotron ColEmbed 8B v2

```bash
python retrieval/nemotron/infer_nemotron_colembed.py \
  --queries work/retrieval_queries.jsonl --documents work/retrieval_documents.jsonl \
  --output_dir work/retr_nemotron \
  --query_instruction "person image retrieval:" \
  --query_batch_size 2 --doc_encode_batch_size 1 --doc_score_chunk_size 2 \
  --topk 36773 --dtype float16 --device_map cuda --attn_implementation sdpa \
  --doc_embedding_cache_dir work/nemotron_cache
```

### 2.4 Ops-Colqwen3-4B (bounded to the Nemotron top-400)

```bash
python retrieval/ops/rerank_nemotron_r400_with_ops_colqwen3.py \
  --nemotron_rankings work/retr_nemotron/rankings.jsonl \
  --query_text work/retrieval_queries.jsonl --gallery_dir $DATASET_DIR/gallery \
  --output_dir work/retr_ops --model_name OpenSearch-AI/Ops-Colqwen3-4B \
  --candidate_topk 400 --topk 400 --query_limit 0 --dims 2560 \
  --query_batch_size 8 --image_batch_size 1 --score_chunk_size 16 \
  --dtype float16 --attn_implementation sdpa \
  --doc_embedding_cache_dir work/ops_cache --overwrite
```

Truncate each retriever's `rankings.jsonl` to its top-400 per query, then fuse
(this and everything below is CPU; `verify.sh` runs the same commands against
the bundled artifacts):

```bash
python src/pipeline.py build-hybrid-rrf \
  --rankings work/retr_argus/rankings.jsonl work/retr_nemotron/rankings.jsonl work/retr_ops/rankings.jsonl \
  --query-order $DATASET_DIR/query_index.txt --rrf-k 60 --source-topk 400 --topk 200 \
  --output work/01_pool/hybrid_pool.jsonl
```

### 2.5 Qwen3-VL-Reranker-8B

```bash
python src/pipeline.py rerank-qwen \
  --rankings work/01_pool/hybrid_pool.jsonl \
  --query-text $DATASET_DIR/query_text.json \
  --gallery-dir $DATASET_DIR/gallery --output-dir work/02_rerank \
  --rerank-topk 50 --batch-size 8 --torch-dtype bfloat16 --overwrite
```

Troubleshooting: if loading the reranker fails with
`LogitScore.__init__() missing ... true_token_id`, write the cached snapshot's
`1_LogitScore/config.json` as `{"true_token_id": 9693, "false_token_id": 2152}`
(the ids of the "yes"/"no" answer tokens) and retry.

### 2.6 Postprocess (blend -> SCA -> component assignment)

```bash
python src/pipeline.py postprocess-fixed \
  --rankings work/01_pool/hybrid_pool.jsonl --scores work/02_rerank/qwen_scores.json \
  --query-order $DATASET_DIR/query_index.txt --output-dir work/03_postprocess
```

### 2.7 Export

```bash
python src/pipeline.py export-answer \
  --rankings work/03_postprocess/final_0.5_0.5_1.0.jsonl \
  --query-order $DATASET_DIR/query_index.txt --output-dir work/04_answer
```

## Checksums

See `expected/checksums.md5` for the md5 of every bundled artifact and
reference output; `md5sum -c expected/checksums.md5` from the repository root
checks them all.
