# Protocol-Aware Global Assignment for Text-Based Person Anomaly Search

Code and artifacts for the paper *Protocol-Aware Global Assignment for Text-Based
Person Anomaly Search*, together with the same pipeline applied to the AI City
Challenge 2026 Track 4 (PAB2026) test set.

Text-based person anomaly search is evaluated under a one-positive, injective
protocol: every query has exactly one correct gallery image, and no image is the
answer to two queries. Independent per-query top-1 selection ignores that
constraint, so once candidate recall saturates the dominant remaining errors are
**collisions**: several queries claiming the same image while their true matches
sit unclaimed one rank below. We resolve them by solving a maximum-weight
one-to-one assignment inside each collision component, in which every query also
holds a private slack arc priced at Δ that lets it keep its current top-1: every
component is solvable and the score-sacrifice bound sits inside the objective.

On PAB2025 the collision-resolution stages (SCA + component assignment) improve the
blended reranker by **+2.0 mAP@10** (96.61 → 98.64) and **+4.0 Recall@1** (93.48 → 97.47).

## Layout

```
src/          pipeline CLI + the post-processing stages (blend, SCA, assignment)
retrieval/    the three retriever scripts (GPU)
pab2025/      paper Table 2: the full stage ladder, reproducible on CPU
docs/         PIPELINE.md   exact stage-by-stage method definition
              SUBMISSION.md running the pipeline on PAB2026 (incl. GPU commands)
artifacts/    PAB2026 model outputs (retriever rankings, reranker scores)
expected/     PAB2026 reference intermediates + answer.txt / answer.zip
verify.sh     PAB2026 end-to-end re-derivation with byte-for-byte checks (CPU)
```

Every stage between the model runs is deterministic, so both benchmarks re-derive
without a GPU from the bundled model outputs. Regenerating those model outputs
needs GPUs and the datasets; the commands are in
[docs/SUBMISSION.md](docs/SUBMISSION.md).

## Reproducing the paper (PAB2025, CPU, ~1 minute)

Needs only the public PAB2025 dataset, for its ground truth. No GPU, no model
download.

```bash
python pab2025/reproduce_ladder.py --pab-root /path/to/PAB2025
```

This reproduces Table 2 of the paper exactly:

| stage | mAP@10 | R@1 |
|---|---|---|
| Argus-ColQwen3.5-9B alone | 93.94 | 88.83 |
| Nemotron-ColEmbed-8B-v2 alone | 93.58 | 88.12 |
| Ops-ColQwen3-4B alone | 93.93 | 89.13 |
| RRF fusion pool (k=60, top-200) | 95.28 | 91.20 |
| + Qwen3-VL-Reranker-8B (raw sort) | 96.80 | 93.73 |
| + rank-prior blend (λ=0.5) | 96.61 | 93.48 |
| + SCA (τ=0.5) | 98.03 | 96.26 |
| **+ component assignment** | **98.64** | **97.47** |

`--pab-root` must contain `test.zip`, `name-masked_test-set/{gallery.zip,query.json}`
and `annotation/test/attr.json`. The bundled retriever rankings are truncated to the
top-256 per query, which is lossless here: the deepest ground-truth positive in any
of the three lists sits at rank 146.

## Running the pipeline on PAB2026 (CPU)

```bash
DATASET_DIR=/path/to/PAB2026/name-masked_test-set bash verify.sh
```

This applies the same pipeline to PAB2026 (1978 queries, 36,773 gallery images),
re-deriving every intermediate and the final answer from the bundled model outputs
and md5-checking each against `expected/`. `DATASET_DIR` needs `query_index.txt`;
the gallery is not required. Both benchmarks use the official query captions as-is.

## Method parameters

The whole configuration was selected on the public, labeled PAB2025 development
set and applied to PAB2026 unchanged: the choice of the three retrievers, their
RRF ensemble, the cross-encoder reranker and its instruction, and the
blend/SCA/assignment postprocessing (each reported standalone on PAB2025 in the
table above). Three constants define the decision layer; no PAB2026 input,
output, or leaderboard feedback was used to select any of these choices. Using
data from previous editions of the challenge is explicitly permitted by the
[official FAQ](https://www.aicitychallenge.org/2026-faqs/) (item 10).

| parameter | value | stage |
|---|---|---|
| λ, the rank-prior blend | 0.5 logits per rank | `blend`, in `src/pipeline.py` |
| τ, the SCA margin gap | 0.5 | `src/sca_postprocess.py` |
| Δ, the slack-arc price (max score sacrifice) | 1.0 | `src/global_assignment_postprocess.py` |

## Requirements

The CPU reproduction needs Python 3.10+ and nothing else: the fusion, blend, SCA
and min-cost assignment stages are pure standard library. The GPU stages (the three
retrievers and the cross-encoder) have their own pinned environments,
listed in [docs/SUBMISSION.md](docs/SUBMISSION.md).

## Citation

```bibtex
@inproceedings{phamduy2026protocol,
  title     = {Protocol-Aware Global Assignment for Text-Based Person Anomaly Search},
  author    = {Pham Duy, Hien and Nguyen Pham Gia, Huy and Tran, Bao and Nguyen, Thua and Do, Tien and Ngo, Thanh Duc and Le, Duy-Dinh and Satoh, Shin'ichi},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE). The PAB2025 and PAB2026 datasets are distributed by
their own authors under their own terms and are not included in this repository.
