#!/usr/bin/env python3
"""PAB Track 4 retrieval pipeline (deterministic stages).

This CLI implements every non-GPU stage of the submission pipeline:

  build-hybrid-rrf   -> reciprocal-rank fusion of the three retriever rankings
  rerank-qwen        -> Qwen3-VL-Reranker-8B scoring (GPU; also runnable standalone)
  postprocess-fixed  -> blend -> SCA -> global assignment (fixed parameters)
  export-answer      -> answer.txt / answer.zip in query_index.txt order
  validate           -> structural checks on rankings/score files

Intermediate artifact formats:

  rankings.jsonl:
      {"query_index": "...", "topk_gallery": ["IMG.jpg", ...]}

  qwen_scores.json:
      [{"query_index": "...", "reranked": [{"image": "IMG.jpg", "score": 1.23}, ...]}]

The expensive model runs (the three retrievers and the reranker) can be executed
on any machine; every stage in between is deterministic given their output files,
so the ranking can be re-derived and checksum-verified without a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


EXACT_VISIBLE_PROMPT = (
    "Rank images for exact text-based person anomaly search. Give highest score "
    "only when visible people, clothing, action, object interaction, relations, "
    "and scene match. Penalize generic scene-only matches and wrong "
    "object/person/action."
)

def stem(value: str) -> str:
    name = os.path.basename(str(value))
    return name[:-4] if name.endswith(".jpg") else name


def jpg(value: str) -> str:
    return stem(value) + ".jpg"


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data, *, indent: int | None = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_rankings(path: str | Path, *, limit: int | None = None) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gallery = item.get("topk_gallery", [])
            if limit is not None:
                gallery = gallery[:limit]
            rows[item["query_index"]] = [jpg(x) for x in gallery]
    return rows


def write_rankings(path: str | Path, rankings: dict[str, list[str]], order: list[str] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    order = order or sorted(rankings)
    with open(path, "w", encoding="utf-8") as f:
        for q in order:
            f.write(json.dumps({"query_index": q, "topk_gallery": rankings[q]}, ensure_ascii=False) + "\n")


def load_scores(path: str | Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in read_json(path):
        out[row["query_index"]] = [
            {"image": jpg(item["image"]), "score": float(item["score"])}
            for item in row.get("reranked", [])
        ]
    return out


def write_scores(path: str | Path, scores: dict[str, list[dict]], order: list[str] | None = None) -> None:
    order = order or sorted(scores)
    write_json(path, [{"query_index": q, "reranked": scores[q]} for q in order], indent=2)


def load_query_text(path: str | Path) -> dict[str, str]:
    """Load common query text formats.

    Accepted forms:
    - JSONL rows with query_index plus caption/query/text.
    - JSON dict mapping query id -> string or object.
    - JSON list of objects with query_index plus caption/query/text.
    """
    p = Path(path)
    if p.suffix == ".jsonl":
        out: dict[str, str] = {}
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                q = row.get("query_index") or row.get("id")
                text = row.get("caption") or row.get("query") or row.get("text")
                if q and text is not None:
                    out[str(q)] = str(text)
        return out
    try:
        data = read_json(p)
    except json.JSONDecodeError:
        out: dict[str, str] = {}
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                q = row.get("query_index") or row.get("id")
                text = row.get("caption") or row.get("query") or row.get("text")
                if q and text is not None:
                    out[str(q)] = str(text)
        if out:
            return out
        raise
    if isinstance(data, dict):
        out = {}
        for q, val in data.items():
            if isinstance(val, str):
                out[str(q)] = val
            elif isinstance(val, dict):
                text = val.get("caption") or val.get("query") or val.get("text")
                if text is not None:
                    out[str(q)] = str(text)
        return out
    if isinstance(data, list):
        out = {}
        for row in data:
            q = row.get("query_index") or row.get("id")
            text = row.get("caption") or row.get("query") or row.get("text")
            if q and text is not None:
                out[str(q)] = str(text)
        return out
    raise ValueError(f"unsupported query text format: {path}")


def load_order(path: str | Path | None, fallback: list[str]) -> list[str]:
    if not path:
        return fallback
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def rrf_for_query(qid: str, retrievers: list[dict[str, list[str]]], *, k: float, topk: int) -> list[str]:
    scores: dict[str, float] = {}
    for rows in retrievers:
        for rank, image in enumerate(rows.get(qid, []), start=1):
            scores[image] = scores.get(image, 0.0) + 1.0 / (k + rank)
    return [img for img, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:topk]]


def cmd_build_hybrid_rrf(args: argparse.Namespace) -> None:
    retrievers = [load_rankings(p, limit=args.source_topk) for p in args.rankings]
    all_qids = sorted(set().union(*(set(x) for x in retrievers)))
    order = load_order(args.query_order, all_qids)
    out: dict[str, list[str]] = {}
    for q in order:
        out[q] = rrf_for_query(q, retrievers, k=args.rrf_k, topk=args.topk)
        if not out[q]:
            raise RuntimeError(f"RRF produced empty candidate list for {q}")

    output = Path(args.output)
    write_rankings(output, out, order)
    print(f"wrote {output}: {len(out)} queries, topk={args.topk}, rrf_k={args.rrf_k}")


def cmd_rerank_qwen(args: argparse.Namespace) -> None:
    try:
        import torch
        from sentence_transformers import CrossEncoder
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit(
            "Qwen reranking requires sentence_transformers, torch, tqdm, "
            "and the Qwen3-VL model-card dependencies."
        ) from exc

    rows = load_rankings(args.rankings)
    qtext = load_query_text(args.query_text)
    order = load_order(args.query_order, list(rows))
    order = [q for q in order if q in rows]

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ce_kwargs = {}
    if args.torch_dtype:
        ce_kwargs["model_kwargs"] = {"torch_dtype": getattr(torch, args.torch_dtype)}
    if args.device:
        ce_kwargs["device"] = args.device
    model = CrossEncoder(args.model_name, **ce_kwargs)
    prompt = EXACT_VISIBLE_PROMPT
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    out_rankings: dict[str, list[str]] = {}
    out_scores: dict[str, list[dict]] = {}
    gallery_dir = Path(args.gallery_dir)
    for q in tqdm(order, desc="Qwen rerank"):
        candidates = rows[q]
        head = candidates[: args.rerank_topk]
        docs = [{"image": str((gallery_dir / img).resolve())} for img in head]
        pairs = [(qtext[q], doc) for doc in docs]
        scores = model.predict(pairs, batch_size=args.batch_size, prompt=prompt)
        if torch.is_tensor(scores):
            scores = scores.detach().cpu().tolist()
        scored = sorted(zip(head, scores), key=lambda kv: kv[1], reverse=True)
        ranked = [img for img, _ in scored]
        out_rankings[q] = ranked[: args.output_topk]
        out_scores[q] = [{"image": img, "score": float(score)} for img, score in scored]

    write_rankings(output_dir / "rankings.jsonl", out_rankings, order)
    write_scores(output_dir / "qwen_scores.json", out_scores, order)
    write_json(
        output_dir / "metadata.json",
        {
            "model_name": args.model_name,
            "rerank_topk": args.rerank_topk,
            "output_topk": args.output_topk,
            "batch_size": args.batch_size,
            "torch_dtype": args.torch_dtype,
            "queries": len(order),
        },
    )
    print(f"wrote {output_dir}/rankings.jsonl and qwen_scores.json")


def score_map_from_rows(scores: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    return {q: {jpg(x["image"]): float(x["score"]) for x in rows} for q, rows in scores.items()}


def blend_rankings(rankings: dict[str, list[str]], scores: dict[str, list[dict]], lam: float) -> dict[str, list[str]]:
    smap = score_map_from_rows(scores)
    out = {}
    for q, gallery in rankings.items():
        top = gallery[:20]
        rest = gallery[20:]
        qscore = smap.get(q, {})
        blended = sorted(
            ((qscore.get(img, -999.0) - lam * idx, img) for idx, img in enumerate(top)),
            key=lambda kv: kv[0],
            reverse=True,
        )
        out[q] = [img for _score, img in blended] + rest
    return out


def run_checked(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        print("FAILED:", " ".join(cmd), file=sys.stderr)
        print(proc.stdout[-2000:], file=sys.stderr)
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(proc.returncode)


def cmd_postprocess_fixed(args: argparse.Namespace) -> None:
    scripts_dir = Path(args.scripts_dir) if args.scripts_dir else Path(__file__).resolve().parent
    sca = scripts_dir / "sca_postprocess.py"
    assign = scripts_dir / "global_assignment_postprocess.py"
    if not sca.exists() or not assign.exists():
        raise SystemExit(f"missing postprocess scripts under {scripts_dir}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rankings = load_rankings(args.rankings)
    scores = load_scores(args.scores)
    order = load_order(args.query_order, list(rankings))

    blend_path = outdir / f"blend_{args.lam}.jsonl"
    sca_path = outdir / f"sca_{args.lam}_{args.sca_tau}.jsonl"
    final_path = outdir / f"final_{args.lam}_{args.sca_tau}_{args.max_score_drop}.jsonl"
    write_rankings(blend_path, blend_rankings(rankings, scores, args.lam), order)
    run_checked([
        sys.executable, str(sca),
        "--rankings", str(blend_path),
        "--scores", str(args.scores),
        "--output", str(sca_path),
        "--tau", str(args.sca_tau),
        "--iterative",
        "--max_passes", "10",
    ])
    run_checked([
        sys.executable, str(assign),
        "--rankings", str(sca_path),
        "--scores", str(args.scores),
        "--output", str(final_path),
        "--topn", str(args.assignment_topn),
        "--rank_penalty", "0",
        "--keep_bonus", "0",
        "--margin_weight", "0",
        "--max_score_drop", str(args.max_score_drop),
        "--process", "rank1_collision",
    ])
    write_json(outdir / "config.json", {k: v for k, v in vars(args).items() if k != "func"})
    print(f"wrote fixed postprocess final: {final_path}")


def cmd_export_answer(args: argparse.Namespace) -> None:
    rankings = load_rankings(args.rankings)
    order = load_order(args.query_order, list(rankings))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    answer = outdir / "answer.txt"
    with answer.open("w", encoding="utf-8") as f:
        for q in order:
            gallery = rankings[q]
            if len(gallery) < args.topk:
                raise RuntimeError(f"{q} has only {len(gallery)} images")
            f.write(" ".join(stem(x) for x in gallery[: args.topk]) + "\n")
    answer_zip = outdir / "answer.zip"
    answer_zip.unlink(missing_ok=True)
    with ZipFile(answer_zip, "w", ZIP_DEFLATED) as zf:
        zf.write(answer, arcname="answer.txt")
    print(f"wrote {answer} and {answer_zip}")


def cmd_validate(args: argparse.Namespace) -> None:
    checks = []
    if args.rankings:
        r = load_rankings(args.rankings)
        depths = [len(x) for x in r.values()]
        checks.append({"path": args.rankings, "type": "rankings", "queries": len(r), "min_depth": min(depths), "max_depth": max(depths)})
    if args.scores:
        s = load_scores(args.scores)
        depths = [len(x) for x in s.values()]
        checks.append({"path": args.scores, "type": "scores", "queries": len(s), "min_depth": min(depths), "max_depth": max(depths)})
    print(json.dumps(checks, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("build-hybrid-rrf")
    r.add_argument("--rankings", nargs=3, required=True, metavar=("ARGUS", "NEMOTRON", "OPS"))
    r.add_argument("--query-order")
    r.add_argument("--rrf-k", type=float, default=60.0)
    r.add_argument("--source-topk", type=int, default=400,
                   help="Per-retriever input depth cap before RRF (the shipped pool used 400).")
    r.add_argument("--topk", type=int, default=200)
    r.add_argument("--output", required=True)
    r.set_defaults(func=cmd_build_hybrid_rrf)

    rq = sub.add_parser("rerank-qwen")
    rq.add_argument("--rankings", required=True)
    rq.add_argument("--query-text", required=True)
    rq.add_argument("--gallery-dir", required=True)
    rq.add_argument("--output-dir", required=True)
    rq.add_argument("--model-name", default="Qwen/Qwen3-VL-Reranker-8B")
    rq.add_argument("--prompt-file")
    rq.add_argument("--rerank-topk", type=int, default=50)
    rq.add_argument("--output-topk", type=int, default=200)
    rq.add_argument("--batch-size", type=int, default=8)
    rq.add_argument("--query-order")
    rq.add_argument("--torch-dtype", default="bfloat16")
    rq.add_argument("--device", default="cuda:0")
    rq.add_argument("--overwrite", action="store_true")
    rq.set_defaults(func=cmd_rerank_qwen)

    pp = sub.add_parser("postprocess-fixed")
    pp.add_argument("--scripts-dir", help="Directory holding sca_postprocess.py and global_assignment_postprocess.py (default: alongside this file).")
    pp.add_argument("--rankings", required=True)
    pp.add_argument("--scores", required=True)
    pp.add_argument("--query-order")
    pp.add_argument("--output-dir", required=True)
    pp.add_argument("--lam", type=float, default=0.5)
    pp.add_argument("--sca-tau", type=float, default=0.5)
    pp.add_argument("--assignment-topn", type=int, default=2)
    pp.add_argument("--max-score-drop", type=float, default=1.0)
    pp.set_defaults(func=cmd_postprocess_fixed)

    ex = sub.add_parser("export-answer")
    ex.add_argument("--rankings", required=True)
    ex.add_argument("--query-order", required=True)
    ex.add_argument("--output-dir", required=True)
    ex.add_argument("--topk", type=int, default=10)
    ex.set_defaults(func=cmd_export_answer)

    va = sub.add_parser("validate")
    va.add_argument("--rankings")
    va.add_argument("--scores")
    va.set_defaults(func=cmd_validate)

    return p


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
