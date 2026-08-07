#!/usr/bin/env python3
"""Reproduce the PAB2025 stage ladder (paper Table 2) from the bundled artifacts.

Every stage below the model runs is deterministic, so this reproduces the reported
numbers exactly on CPU in a few seconds:

    argus / nemotron / ops alone
      -> RRF fusion pool (k=60, top-200)
      -> reranker raw sort (top-50 of the pool by cross-encoder score)
      -> blend (lambda=0.5)
      -> SCA (tau=0.5)
      -> component assignment (topn=2, max score drop=1.0)

Usage:
    python reproduce_ladder.py --pab-root /path/to/PAB2025

--pab-root must be the PAB2025 dataset root (it supplies the public ground truth):
it needs test.zip, name-masked_test-set/{gallery.zip,query.json} and
annotation/test/attr.json.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PIPELINE = HERE.parent / "src" / "pipeline.py"
ORDER = DATA / "query_index.txt"
SCORES = DATA / "qwen_scores.json"

sys.path.insert(0, str(HERE))
import evaluate_pab2025 as EV  # noqa: E402


def jpg(value):
    return os.path.splitext(os.path.basename(value))[0] + ".jpg"


def write_rankings(path, rankings, order):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for q in order:
            f.write(json.dumps({"query_index": q, "topk_gallery": rankings[q]}) + "\n")


def load_rankings(path, k=None):
    out = {}
    for line in open(path):
        if line.strip():
            row = json.loads(line)
            gallery = [jpg(x) for x in row["topk_gallery"]]
            out[row["query_index"]] = gallery[:k] if k else gallery
    return out


def run_pipeline(subcmd, args):
    subprocess.run([sys.executable, str(PIPELINE), subcmd] + args, check=True, capture_output=True)


def build_ground_truth(pab_root):
    root = Path(pab_root)
    test_to_gallery, _, _, _ = EV.build_test_to_gallery_map(
        root / "test.zip", root / "name-masked_test-set/gallery.zip")
    gt, _ = EV.build_ground_truth(
        root / "annotation/test/attr.json", root / "name-masked_test-set/query.json", test_to_gallery)
    return gt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pab-root", required=True, help="PAB2025 dataset root (supplies the public ground truth)")
    ap.add_argument("--output-dir", default=str(HERE / "work"))
    args = ap.parse_args()

    order = [l.strip() for l in open(ORDER) if l.strip()]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gt = build_ground_truth(args.pab_root)
    print(f"ground truth: {len(gt)} queries\n")

    def score(path):
        metrics, _ = EV.evaluate(EV.load_rankings_jsonl(path), gt)
        return metrics["mAP"], metrics["R1"], metrics["R5"], metrics["R10"]

    rows = []

    # --- individual retrievers ---
    for name, leg in [("Argus alone", "argus"), ("Nemotron alone", "nemotron"), ("Ops alone", "ops")]:
        path = out / f"leg_{leg}.jsonl"
        write_rankings(path, load_rankings(DATA / "legs" / f"{leg}_top256.jsonl"), order)
        rows.append((name, *score(path)))

    # --- RRF fusion pool ---
    pool_path = out / "hybrid_pool.jsonl"
    run_pipeline("build-hybrid-rrf", [
        "--rankings", *[str(DATA / "legs" / f"{leg}_top256.jsonl") for leg in ("argus", "nemotron", "ops")],
        "--query-order", str(ORDER), "--rrf-k", "60", "--source-topk", "256", "--topk", "200",
        "--output", str(pool_path)])
    rows.append(("RRF fusion pool", *score(pool_path)))

    # --- reranker raw sort: top-50 of the pool re-sorted by cross-encoder score ---
    pool = load_rankings(pool_path)
    score_map = {r["query_index"]: {jpg(x["image"]): x["score"] for x in r["reranked"] if x.get("score") is not None}
                 for r in json.load(open(SCORES))}
    raw = {}
    for q, gallery in pool.items():
        sm = score_map.get(q, {})
        raw[q] = sorted(gallery[:50], key=lambda x: sm.get(x, -999), reverse=True) + gallery[50:]
    raw_path = out / "rerank_sort.jsonl"
    write_rankings(raw_path, raw, order)
    rows.append(("+ reranker (raw sort)", *score(raw_path)))

    # --- blend -> SCA -> component assignment ---
    run_pipeline("postprocess-fixed", [
        "--rankings", str(pool_path), "--scores", str(SCORES),
        "--query-order", str(ORDER), "--output-dir", str(out / "pp")])
    rows.append(("+ blend (lambda=0.5)", *score(out / "pp/blend_0.5.jsonl")))
    rows.append(("+ SCA (tau=0.5)", *score(out / "pp/sca_0.5_0.5.jsonl")))
    rows.append(("+ component assignment", *score(out / "pp/final_0.5_0.5_1.0.jsonl")))

    print(f"{'stage':30s} {'mAP@10':>8s} {'R@1':>8s} {'R@5':>8s} {'R@10':>8s} {'d mAP':>8s}")
    prev = None
    for name, mp, r1, r5, r10 in rows:
        delta = f"{mp - prev:+8.2f}" if prev is not None and name.startswith("+") else " " * 8
        print(f"{name:30s} {mp:8.2f} {r1:8.2f} {r5:8.2f} {r10:8.2f} {delta}")
        if name.startswith("+") or name == "RRF fusion pool":
            prev = mp

    table = [{"stage": n, "mAP": mp, "R1": r1, "R5": r5, "R10": r10} for n, mp, r1, r5, r10 in rows]
    json.dump(table, open(out / "ladder_table.json", "w"), indent=2)
    print(f"\nwrote {out / 'ladder_table.json'}")


if __name__ == "__main__":
    main()
