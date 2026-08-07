#!/usr/bin/env python3
"""Global collision assignment post-process for PAB rankings.

This fixes a specific one-positive retrieval failure mode: multiple queries
claiming the same rank-1 gallery image. For each connected collision component,
it assigns each query a unique candidate from its top-N list, maximizing a
margin-aware utility from Qwen reranker scores.

No model calls are made.
"""

import argparse
import heapq
import json
from collections import defaultdict, deque
from pathlib import Path


INF = 10**18


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_score_rows(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_name(value):
    return Path(value).name


def load_rankings(path):
    rows = []
    by_query = {}
    for row in iter_jsonl(path):
        query_id = row["query_index"]
        gallery = [image_name(x) for x in row["topk_gallery"]]
        item = {**row, "topk_gallery": gallery}
        rows.append(item)
        by_query[query_id] = item
    return rows, by_query


def load_score_maps(path):
    score_maps = {}
    for row in read_score_rows(path):
        query_id = row["query_index"]
        score_maps[query_id] = {
            image_name(item["image"]): float(item["score"])
            for item in row.get("reranked", [])
        }
    return score_maps


def count_rank1_collisions(rows):
    groups = defaultdict(list)
    for row in rows:
        gallery = row["topk_gallery"]
        if gallery:
            groups[gallery[0]].append(row["query_index"])
    return {
        "collision_images": sum(1 for qs in groups.values() if len(qs) > 1),
        "colliding_queries": sum(len(qs) for qs in groups.values() if len(qs) > 1),
    }


def build_components(rows, topn):
    query_to_images = {}
    image_to_queries = defaultdict(list)
    for row in rows:
        query_id = row["query_index"]
        candidates = row["topk_gallery"][:topn]
        query_to_images[query_id] = candidates
        for img in candidates:
            image_to_queries[img].append(query_id)

    seen_q = set()
    components = []
    for row in rows:
        start_q = row["query_index"]
        if start_q in seen_q:
            continue
        qset = set()
        iset = set()
        queue = deque([("q", start_q)])
        seen_q.add(start_q)
        while queue:
            kind, node = queue.popleft()
            if kind == "q":
                qset.add(node)
                for img in query_to_images.get(node, []):
                    if img not in iset:
                        iset.add(img)
                        queue.append(("i", img))
            else:
                for q in image_to_queries.get(node, []):
                    if q not in seen_q:
                        seen_q.add(q)
                        queue.append(("q", q))
        components.append((sorted(qset), sorted(iset)))
    return components


def component_has_rank1_collision(query_ids, by_query):
    groups = defaultdict(list)
    for q in query_ids:
        gallery = by_query[q]["topk_gallery"]
        if gallery:
            groups[gallery[0]].append(q)
    return any(len(qs) > 1 for qs in groups.values())


def score_value(score_map, image, rank):
    if image in score_map:
        return score_map[image]
    return -1000.0 - float(rank)


def query_margin(row, score_map):
    gallery = row["topk_gallery"]
    if len(gallery) < 2:
        return 0.0
    s1 = score_value(score_map, gallery[0], 0)
    s2 = score_value(score_map, gallery[1], 1)
    return s1 - s2


def make_utilities(query_ids, image_ids, by_query, score_maps, args):
    image_index = {img: idx for idx, img in enumerate(image_ids)}
    utilities = {}
    for q in query_ids:
        row = by_query[q]
        gallery = row["topk_gallery"]
        score_map = score_maps.get(q, {})
        if not gallery:
            continue
        top1 = gallery[0]
        top1_score = score_value(score_map, top1, 0)
        margin = query_margin(row, score_map)
        for rank, img in enumerate(gallery[: args.topn]):
            if img not in image_index:
                continue
            score = score_value(score_map, img, rank)
            utility = score - top1_score
            if rank == 0:
                utility += args.keep_bonus + args.margin_weight * margin
            else:
                utility -= args.rank_penalty * rank
            utilities[(q, img)] = utility
    return utilities


class MinCostMaxFlow:
    def __init__(self, n):
        self.graph = [[] for _ in range(n)]

    def add_edge(self, u, v, cap, cost):
        fwd = [v, cap, cost, None]
        rev = [u, 0, -cost, fwd]
        fwd[3] = rev
        self.graph[u].append(fwd)
        self.graph[v].append(rev)

    def solve(self, source, sink, need_flow):
        n = len(self.graph)
        potential = [0] * n
        flow = 0
        cost = 0
        while flow < need_flow:
            dist = [INF] * n
            prev_node = [-1] * n
            prev_edge = [None] * n
            dist[source] = 0
            heap = [(0, source)]
            while heap:
                cur_dist, u = heapq.heappop(heap)
                if cur_dist != dist[u]:
                    continue
                for edge in self.graph[u]:
                    v, cap, edge_cost, _rev = edge
                    if cap <= 0:
                        continue
                    nd = cur_dist + edge_cost + potential[u] - potential[v]
                    if nd < dist[v]:
                        dist[v] = nd
                        prev_node[v] = u
                        prev_edge[v] = edge
                        heapq.heappush(heap, (nd, v))
            if dist[sink] == INF:
                break
            for i in range(n):
                if dist[i] < INF:
                    potential[i] += dist[i]
            add = need_flow - flow
            v = sink
            while v != source:
                edge = prev_edge[v]
                add = min(add, edge[1])
                v = prev_node[v]
            v = sink
            while v != source:
                edge = prev_edge[v]
                edge[1] -= add
                edge[3][1] += add
                cost += edge[2] * add
                v = prev_node[v]
            flow += add
        return flow, cost


def solve_assignment(query_ids, image_ids, utilities, scale=100000):
    q_count = len(query_ids)
    i_count = len(image_ids)
    source = 0
    q_base = 1
    i_base = q_base + q_count
    sink = i_base + i_count
    mcmf = MinCostMaxFlow(sink + 1)

    for qi, _q in enumerate(query_ids):
        mcmf.add_edge(source, q_base + qi, 1, 0)
    for ii, _img in enumerate(image_ids):
        mcmf.add_edge(i_base + ii, sink, 1, 0)

    for qi, q in enumerate(query_ids):
        for ii, img in enumerate(image_ids):
            util = utilities.get((q, img))
            if util is None:
                continue
            # Min-cost flow minimizes, so negate utility.
            mcmf.add_edge(q_base + qi, i_base + ii, 1, -int(round(util * scale)))

    flow, neg_cost = mcmf.solve(source, sink, q_count)
    if flow < q_count:
        return None, None

    assignments = {}
    for qi, q in enumerate(query_ids):
        qnode = q_base + qi
        for edge in mcmf.graph[qnode]:
            v, cap, _cost, rev = edge
            if i_base <= v < i_base + i_count and cap == 0 and rev[1] == 1:
                assignments[q] = image_ids[v - i_base]
                break
    return assignments, -neg_cost / scale


def move_assigned_to_front(row, assigned_image):
    gallery = row["topk_gallery"]
    if not gallery or gallery[0] == assigned_image or assigned_image not in gallery:
        return False
    gallery.remove(assigned_image)
    gallery.insert(0, assigned_image)
    return True


def main():
    parser = argparse.ArgumentParser(description="Global assignment postprocess for rank-1 collisions.")
    parser.add_argument("--rankings", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--rank_penalty", type=float, default=0.05)
    parser.add_argument("--keep_bonus", type=float, default=0.10)
    parser.add_argument("--margin_weight", type=float, default=1.0)
    parser.add_argument(
        "--max_score_drop",
        type=float,
        default=None,
        help="Slack price Delta: every query holds a private slack arc at this cost "
        "through which it keeps its current (possibly colliding) top-1, so every "
        "component is feasible and the score-sacrifice bound sits inside the objective.",
    )
    parser.add_argument(
        "--process",
        choices=["rank1_collision", "all_components"],
        default="rank1_collision",
        help="Limit assignment to components that already have duplicate rank-1 images.",
    )
    parser.add_argument("--max_component_queries", type=int, default=200)
    args = parser.parse_args()

    rows, by_query = load_rankings(args.rankings)
    score_maps = load_score_maps(args.scores)
    before = count_rank1_collisions(rows)
    components = build_components(rows, args.topn)

    processed = 0
    skipped_large = 0
    changed = []
    slack_kept = 0
    assignment_examples = []
    for query_ids, image_ids in components:
        if len(query_ids) < 2:
            continue
        if args.process == "rank1_collision" and not component_has_rank1_collision(query_ids, by_query):
            continue
        if len(query_ids) > args.max_component_queries:
            skipped_large += 1
            continue
        utilities = make_utilities(query_ids, image_ids, by_query, score_maps, args)
        slack_ids = [f"__SLACK__{q}" for q in query_ids]
        for q, slack in zip(query_ids, slack_ids):
            utilities[(q, slack)] = -(args.max_score_drop or 0.0)
        assignments, utility_sum = solve_assignment(query_ids, list(image_ids) + slack_ids, utilities)
        if assignments is None:
            continue
        processed += 1
        comp_changes = []
        for q, assigned in assignments.items():
            row = by_query[q]
            old_top1 = row["topk_gallery"][0] if row["topk_gallery"] else None
            if assigned.startswith("__SLACK__"):
                slack_kept += 1
                continue
            if move_assigned_to_front(row, assigned):
                item = {
                    "query_index": q,
                    "old_rank1": old_top1,
                    "new_rank1": assigned,
                    "component_queries": len(query_ids),
                    "component_images": len(image_ids),
                }
                changed.append(item)
                comp_changes.append(item)
        if comp_changes and len(assignment_examples) < 20:
            assignment_examples.append(
                {
                    "queries": query_ids[:20],
                    "component_queries": len(query_ids),
                    "component_images": len(image_ids),
                    "utility_sum": utility_sum,
                    "changes": comp_changes[:20],
                }
            )

    after = count_rank1_collisions(rows)
    write_jsonl(args.output, rows)
    report = {
        "rankings": str(Path(args.rankings).resolve()),
        "scores": str(Path(args.scores).resolve()),
        "output": str(Path(args.output).resolve()),
        "topn": args.topn,
        "rank_penalty": args.rank_penalty,
        "keep_bonus": args.keep_bonus,
        "margin_weight": args.margin_weight,
        "max_score_drop": args.max_score_drop,
        "process": args.process,
        "components": len(components),
        "components_processed": processed,
        "components_skipped_large": skipped_large,
        "slack_kept": slack_kept,
        "rank1_collisions_before": before,
        "rank1_collisions_after": after,
        "changed_rank1": len(changed),
        "change_examples": changed[:50],
        "assignment_examples": assignment_examples,
    }
    report_path = Path(args.report) if args.report else Path(args.output).with_name("assignment_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
