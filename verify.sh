#!/bin/bash
# Deterministic (CPU-only) re-derivation of the PAB2026 ranking from bundled artifacts.
#
# Rebuilds every non-GPU stage from the bundled model outputs and checks each
# intermediate byte-for-byte against expected/.
#
# Usage:
#   DATASET_DIR=/path/to/PAB2026/name-masked_test-set bash verify.sh
#
# DATASET_DIR must contain query_index.txt.
set -euo pipefail
cd "$(dirname "$0")"
: "${DATASET_DIR:?set DATASET_DIR to the PAB2026 name-masked_test-set directory}"
ORDER=$DATASET_DIR/query_index.txt
PY=${PYTHON:-python3}
W=work/verify
mkdir -p "$W"

check() { # check <label> <produced> <reference>
  local a b
  a=$(md5sum "$2" | cut -d' ' -f1); b=$(md5sum "$3" | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then echo "OK   $1"; else echo "FAIL $1: $2 ($a) != $3 ($b)"; exit 1; fi
}

echo "== stage 1: RRF fusion pool from the three retriever rankings =="
"$PY" src/pipeline.py build-hybrid-rrf \
  --rankings artifacts/legs/argus_top400.jsonl artifacts/legs/nemotron_top400.jsonl artifacts/legs/ops_top400.jsonl \
  --query-order "$ORDER" --rrf-k 60 --source-topk 400 --topk 200 \
  --output "$W/01_pool/hybrid_pool.jsonl" >/dev/null
check "hybrid_pool" "$W/01_pool/hybrid_pool.jsonl" artifacts/pool/hybrid_pool.jsonl

echo "== stage 2: blend -> SCA -> component assignment (fixed parameters) =="
"$PY" src/pipeline.py postprocess-fixed \
  --rankings artifacts/pool/hybrid_pool.jsonl \
  --scores artifacts/rerank/qwen_scores.json \
  --query-order "$ORDER" --output-dir "$W/02_postprocess" >/dev/null
check "blend"      "$W/02_postprocess/blend_0.5.jsonl"          expected/blend.jsonl
check "sca"        "$W/02_postprocess/sca_0.5_0.5.jsonl"        expected/sca.jsonl
check "assignment" "$W/02_postprocess/final_0.5_0.5_1.0.jsonl"  expected/rankings_final.jsonl

echo "== stage 3: export top-10 =="
"$PY" src/pipeline.py export-answer \
  --rankings "$W/02_postprocess/final_0.5_0.5_1.0.jsonl" \
  --query-order "$ORDER" --output-dir "$W/03_answer" >/dev/null
check "answer.txt" "$W/03_answer/answer.txt" expected/answer.txt

echo
echo "ALL CHECKS PASSED: work/verify/03_answer/answer.txt reproduces the bundled ranking."
