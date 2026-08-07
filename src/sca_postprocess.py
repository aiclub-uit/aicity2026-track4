#!/usr/bin/env python3
"""Score-margin collision arbitration for PAB rankings.

SCA is a conservative post-process for one-positive retrieval protocols:
if several queries choose the same rank-1 gallery image, keep that image for
the query with the strongest top1-top2 reranker margin and swap lower-margin
colliders to their rank-2 only when the margin gap is larger than tau.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_name(value):
    return Path(value).name


def load_rankings(path):
    rows = read_jsonl(path)
    by_query = {}
    order = []
    for row in rows:
        query_id = row["query_index"]
        gallery = [image_name(x) for x in row["topk_gallery"]]
        by_query[query_id] = {**row, "topk_gallery": gallery}
        order.append(query_id)
    return by_query, order


def load_score_margins(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        rows = []
    elif text[0] == "[":
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    margins = {}
    scores = {}
    for row in rows:
        query_id = row["query_index"]
        reranked = row.get("reranked", [])
        score_map = {image_name(item["image"]): float(item["score"]) for item in reranked}
        scores[query_id] = score_map
        if len(reranked) >= 2:
            margins[query_id] = float(reranked[0]["score"]) - float(reranked[1]["score"])
        elif len(reranked) == 1:
            margins[query_id] = float("inf")
        else:
            margins[query_id] = float("-inf")
    return margins, scores


def current_margin(query_id, gallery, score_maps, fallback_margins):
    if len(gallery) < 2:
        return fallback_margins.get(query_id, float("-inf"))
    score_map = score_maps.get(query_id, {})
    if gallery[0] not in score_map or gallery[1] not in score_map:
        return fallback_margins.get(query_id, float("-inf"))
    return score_map[gallery[0]] - score_map[gallery[1]]


def apply_sca(rankings, margins, score_maps, tau, max_passes):
    all_swaps = []
    for pass_idx in range(max_passes):
        groups = defaultdict(list)
        for query_id, row in rankings.items():
            gallery = row["topk_gallery"]
            if gallery:
                groups[gallery[0]].append(query_id)

        pass_swaps = []
        for top1, query_ids in sorted(groups.items()):
            if len(query_ids) < 2:
                continue

            ranked_queries = sorted(
                query_ids,
                key=lambda q: (
                    current_margin(q, rankings[q]["topk_gallery"], score_maps, margins),
                    q,
                ),
                reverse=True,
            )
            keeper = ranked_queries[0]
            keeper_margin = current_margin(keeper, rankings[keeper]["topk_gallery"], score_maps, margins)

            for loser in ranked_queries[1:]:
                gallery = rankings[loser]["topk_gallery"]
                if len(gallery) < 2:
                    continue
                loser_margin = current_margin(loser, gallery, score_maps, margins)
                if keeper_margin - loser_margin <= tau:
                    continue
                before = gallery[:2]
                gallery[0], gallery[1] = gallery[1], gallery[0]
                pass_swaps.append(
                    {
                        "pass": pass_idx + 1,
                        "query_index": loser,
                        "collision_image": top1,
                        "new_rank1": gallery[0],
                        "old_top2": before[1],
                        "keeper_query": keeper,
                        "keeper_margin": keeper_margin,
                        "loser_margin": loser_margin,
                        "margin_gap": keeper_margin - loser_margin,
                    }
                )

        all_swaps.extend(pass_swaps)
        if not pass_swaps:
            break
    return all_swaps


def apply_sca_iterative(rankings, margins, tau, max_passes):
    """Monotone position-advance fixed-point SCA (frozen top-1 margin variant).

    Each pass groups queries by their CURRENT candidate ``cand[q][pos[q]]``; in any
    collision group the max-frozen-margin query keeps the image and every other query
    whose margin gap exceeds ``tau`` advances its pointer +1 (never backward). Repeat
    until a pass demotes nobody (fixed point) or ``max_passes`` is reached.

    Unlike the single-pass swap (which reverses rank1<->rank2 and so oscillates when
    re-detected), the pointer only moves forward through the frozen reranked list, so
    chained collisions (loser's rank-2 is itself contested) resolve and the process
    provably converges. At tau>=1 the worst case is a no-op (pure upside).
    """
    NEG = float("-inf")
    cand = {q: list(rankings[q]["topk_gallery"]) for q in rankings}
    pos = {q: 0 for q in cand}
    swaps = []
    passes = 0
    for pass_idx in range(max_passes):
        groups = defaultdict(list)
        for q, gallery in cand.items():
            if gallery:
                groups[gallery[pos[q]]].append(q)

        demotions = []
        for image, query_ids in sorted(groups.items()):
            if len(query_ids) < 2:
                continue
            keeper = max(query_ids, key=lambda q: (margins.get(q, NEG), q))
            keeper_margin = margins.get(keeper, NEG)
            for loser in query_ids:
                if loser == keeper:
                    continue
                loser_margin = margins.get(loser, NEG)
                if keeper_margin - loser_margin <= tau:
                    continue
                if pos[loser] + 1 >= len(cand[loser]):
                    continue
                demotions.append((loser, image, keeper, keeper_margin, loser_margin))

        if not demotions:
            break
        passes = pass_idx + 1
        for loser, image, keeper, keeper_margin, loser_margin in demotions:
            old_rank1 = cand[loser][pos[loser]]
            pos[loser] += 1
            swaps.append(
                {
                    "pass": passes,
                    "query_index": loser,
                    "collision_image": image,
                    "old_rank1": old_rank1,
                    "new_rank1": cand[loser][pos[loser]],
                    "keeper_query": keeper,
                    "keeper_margin": keeper_margin,
                    "loser_margin": loser_margin,
                    "margin_gap": keeper_margin - loser_margin,
                }
            )

    # Materialize: bring each advanced query's chosen candidate to the front,
    # preserving the original reranked order of the remaining images.
    for q, p in pos.items():
        if p > 0:
            gallery = cand[q]
            chosen = gallery[p]
            rankings[q]["topk_gallery"] = [chosen] + gallery[:p] + gallery[p + 1:]
    return swaps, passes


def main():
    parser = argparse.ArgumentParser(description="Apply SCA collision postprocess to PAB rankings.")
    parser.add_argument("--rankings", required=True, help="Input rankings.jsonl.")
    parser.add_argument("--scores", required=True, help="Qwen reranker qwen_scores.json.")
    parser.add_argument("--output", required=True, help="Output rankings.jsonl.")
    parser.add_argument("--report", default=None, help="Output JSON report. Defaults beside output.")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--max_passes", type=int, default=1)
    parser.add_argument(
        "--iterative",
        action="store_true",
        help="Monotone pointer-advance fixed-point SCA (frozen top-1 margin). Needs --max_passes > 1 "
        "to iterate; converges in ~3 passes. Drop-in upgrade over single-pass: >= fixed count, 0 broken at tau>=1.",
    )
    args = parser.parse_args()

    rankings, order = load_rankings(args.rankings)
    margins, score_maps = load_score_margins(args.scores)
    missing_scores = [q for q in order if q not in margins]
    if missing_scores:
        raise ValueError(f"Missing score rows for {len(missing_scores)} queries, first={missing_scores[0]}")

    before_top1 = {q: rankings[q]["topk_gallery"][0] for q in order if rankings[q]["topk_gallery"]}
    if args.iterative:
        swaps, passes = apply_sca_iterative(rankings, margins, args.tau, args.max_passes)
    else:
        swaps = apply_sca(rankings, margins, score_maps, args.tau, args.max_passes)
        passes = max((s["pass"] for s in swaps), default=0)
    after_top1 = {q: rankings[q]["topk_gallery"][0] for q in order if rankings[q]["topk_gallery"]}

    output_rows = [rankings[q] for q in order]
    write_jsonl(args.output, output_rows)

    collision_groups = defaultdict(list)
    for q, img in before_top1.items():
        collision_groups[img].append(q)
    report = {
        "rankings": args.rankings,
        "scores": args.scores,
        "output": args.output,
        "tau": args.tau,
        "mode": "iterative" if args.iterative else "single",
        "max_passes": args.max_passes,
        "passes_run": passes,
        "queries": len(order),
        "collision_images_before": sum(1 for qs in collision_groups.values() if len(qs) > 1),
        "colliding_queries_before": sum(len(qs) for qs in collision_groups.values() if len(qs) > 1),
        "swaps": len(swaps),
        "changed_rank1": sum(1 for q in order if before_top1.get(q) != after_top1.get(q)),
        "swap_examples": swaps[:20],
    }
    report_path = Path(args.report) if args.report else Path(args.output).with_name("sca_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
