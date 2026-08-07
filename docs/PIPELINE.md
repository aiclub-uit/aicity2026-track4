# Pipeline: exact definition

This document defines every stage of the pipeline and the exact top-k
boundaries, prompts, and thresholds. `SUBMISSION.md` gives the concrete
commands; `verify.sh` re-derives the final ranking from the bundled model
outputs and checks every intermediate byte-for-byte.

The stages below are described on PAB2026, which uses the larger 36,773-image
gallery. **PAB2025, the benchmark the paper reports, runs the identical stages**
on its own 1978-image gallery; see `../pab2025/reproduce_ladder.py`. Both
benchmarks feed every stage the official query captions exactly as distributed.

## Inputs

- PAB2026 name-masked test set: `query_index.txt` (1978 ids), `query_text.json`
  (1978 captions), `gallery/` (36773 images).

## Stage 1: three first-stage retrievers

Each retriever ranks the gallery for all 1978 queries using the official
captions, then is truncated to its **top 400** per query for fusion.

1. **Argus-Colqwen3.5-9b** (`DataScience-UIBK/Argus-Colqwen3.5-9b-v0`), late
   interaction over the full gallery: bfloat16, sdpa, `max_visual_tokens 2048`,
   score chunk 64 (`retrieval/argus/argus_encode_shard.py` +
   `retrieval/argus/argus_retrieve_from_cache.py`).
2. **Nemotron ColEmbed** (`nvidia/nemotron-colembed-vl-8b-v2`)
   (`retrieval/nemotron/infer_nemotron_colembed.py`): query instruction
   `"person image retrieval:"`, float16, sdpa, `query_batch_size 2`,
   `doc_encode_batch_size 1`, full-gallery scoring truncated to the top 400.
3. **Ops-Colqwen3-4B** (`OpenSearch-AI/Ops-Colqwen3-4B`), run as a *bounded*
   ranker over the Nemotron top-400 candidates per query
   (`retrieval/ops/rerank_nemotron_r400_with_ops_colqwen3.py`:
   `candidate_topk 400`, `topk 400`, `dims 2560`, float16, sdpa).

## Stage 2: reciprocal-rank fusion

RRF with one-based ranks over the three top-400 lists:

```text
rrf_score(image) = sum_retrievers 1 / (60 + rank_in_that_retriever)
```

Keep the **top 200** per query. This is the candidate pool.

## Stage 3: Qwen3-VL-Reranker-8B

`Qwen/Qwen3-VL-Reranker-8B` as a cross-encoder, bfloat16, batch size 8. Only
the **first 50** pool images per query are scored (`rerank_topk 50`), for all
1978 queries, with the instruction:

```text
Rank images for exact text-based person anomaly search. Give highest score only when visible people, clothing, action, object interaction, relations, and scene match. Penalize generic scene-only matches and wrong object/person/action.
```

## Stage 4: fixed postprocess on the pool order

Run on the **pool order** (not the reranked order) with the reranker scores.
The reranker is the strongest per-pair signal but is overconfident on some
hard negatives; the RRF pool order encodes the consensus of three independent
retrievers. This stage combines the two and then resolves the specific
failure mode of a one-positive benchmark: several queries claiming the same
gallery image.

1. **Blend** *fuses reranker score with retrieval consensus as a confidence
   gate.* Reorder only the top 20 of each query's pool by
   `qwen_score(image) − 0.5 · i` where `i` is the zero-based position inside
   the top-20; positions 21..200 stay unchanged. The 0.5-per-rank penalty (in
   reranker logit units) means a candidate sitting deeper in the fusion order
   is promoted only when its reranker advantage is large, so that small score
   edges cannot override the three-retriever consensus.
2. **SCA** (`src/sca_postprocess.py`) performs *pairwise collision arbitration*,
   following the Similarity Coverage Analysis strategy of He et al.,
   "Efficient Vision Language Model Fine-tuning for Text-based Person
   Anomaly Search" (arXiv:2502.03230), a top solution of the previous
   edition of this benchmark, which uses the same kind of tunable
   score-margin threshold.
   Each gallery image answers at most one query, so when several queries share
   the same rank-1 image, at most one of them can be right. SCA keeps the
   image for the query with the strongest top1−top2 reranker margin and moves
   a lower-margin collider to its own rank-2 only when the margin gap exceeds
   `tau 0.5` (iterative, `max_passes 10`), so only clearly-losing claims
   are moved.
3. **Global assignment** (`src/global_assignment_postprocess.py`)
   performs *component-wide collision resolution.* SCA arbitrates pairs; chains of
   colliding queries form larger connected components, which are solved as one
   min-cost-flow assignment per component: every query chooses among its
   top-`2` candidates **plus a private slack arc priced at
   `max_score_drop 1.0`** that keeps its current (possibly colliding) top-1 at
   that fixed cost. Utilities are purely reranker scores (`rank_penalty 0`,
   `keep_bonus 0`, `margin_weight 0`). The slack arcs make every component
   feasible and fold the per-query sacrifice bound into the objective; only
   components containing a duplicated rank-1 image are assigned
   (`--process rank1_collision`).

## Stage 5: export

Top-10 per query, one line per query in `query_index.txt` order, zipped as
`answer.zip`.

## Determinism notes

- Stages 2, 4 and 5 are pure CPU transforms: given the bundled retriever
  rankings and reranker scores, they reproduce the final ranking byte-for-byte
  (`verify.sh`).
- The GPU stages (retrievers, reranker) are floating-point model runs;
  re-running them on different hardware or library builds can reorder near-tied
  candidates. The bundled artifacts are the exact outputs produced with the
  pinned environments in `SUBMISSION.md`.
- Configuration provenance: the model stack and pipeline design, namely the three
  retrievers, their RRF ensemble, the reranker and its instruction, and the
  blend/SCA/assignment stages, were selected on the public, labeled PAB2025
  development set, where each component is measured standalone
  (`../pab2025/reproduce_ladder.py`), and applied to PAB2026 without further
  tuning. Using data from previous editions of the challenge is explicitly
  permitted by the official FAQ, item 10
  (https://www.aicitychallenge.org/2026-faqs/).
- Threshold provenance: the decision layer has three constants, namely blend penalty
  `lambda 0.5` per rank, SCA `tau 0.5`, assignment `max_score_drop 1.0`, all
  expressed in Qwen-reranker logit units and all fixed on the public, labeled
  PAB2025 development set. No PAB2026 input, output, or leaderboard feedback
  was used to select them. They are round values interpretable in those units:
  `tau 0.5` moves a colliding query off a shared rank-1 only when its top1-top2
  reranker margin loses by more than half a logit. RRF `k = 60` is the standard
  literature default.
- Measured sensitivity on PAB2025 (mAP@10 of the complete pipeline):
  - `max_score_drop` (the slack price) is flat across the interval containing
    the chosen value: 0.5 and 1.0 both give **98.64**; 2.0 gives 98.61. The
    chosen 1.0 sits on that plateau.
  - `lambda` is a genuine optimum rather than a plateau: 0.35 gives 98.46,
    **0.5 gives 98.64**, 0.75 gives 98.45. The whole sweep `lambda` in [0, 1]
    spans 98.26-98.64, so the pipeline tolerates the choice within ~0.38 mAP,
    but 0.5 is a peak and is selected as the PAB2025 end-to-end optimum.
  - `sca tau` composed with assignment peaks at the chosen 0.5 (0.3 gives
    98.56, **0.5 and 0.75 give 98.64**, 1.0 gives 98.56), although SCA alone
    would prefer tau 0.3, which is another instance of calibrating a stage for the
    pipeline rather than in isolation.
- The stages these constants control only re-order an already-fixed candidate
  set; they never introduce a new candidate. Recall is carried entirely by the
  retrieval and rerank stages, whose only fusion constant is RRF `k = 60`.

The threshold-gated collision post-processing itself is inherited practice for
this benchmark, not invented here: see the SCA citation in Stage 4
(arXiv:2502.03230, previous edition's top solution, same kind of tunable
score-margin parameter).
