import argparse
import json
from pathlib import Path
from zipfile import ZipFile


def iter_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def stem_or_name(value):
    return Path(value).stem


def gallery_name(value, keep_ext=True):
    name = Path(value).name
    return name if keep_ext else Path(name).stem


def build_test_to_gallery_map(test_zip_path, gallery_zip_path):
    """Resolve the PAB2025 gallery's filename masking against the public release.

    The name-masked gallery distributed for the challenge contains byte-identical
    copies of the images in the public annotated PAB2025 test set, only under
    randomized filenames. Matching on (CRC32, file size) from the two zip
    directories joins each masked file to its public original, so the public
    annotation files can score rankings expressed in masked names. The join adds
    no label information of its own; the caller checks that no two files share a
    (CRC32, size) pair, which makes it unambiguous.
    """
    with ZipFile(test_zip_path) as z:
        test_infos = [info for info in z.infolist() if info.filename.lower().endswith(".jpg")]
        test_by_crc_size = {(info.CRC, info.file_size): info.filename for info in test_infos}

    with ZipFile(gallery_zip_path) as z:
        gallery_infos = [info for info in z.infolist() if info.filename.lower().endswith(".jpg")]
        mapping = {}
        collisions = []
        for info in gallery_infos:
            key = (info.CRC, info.file_size)
            test_name = test_by_crc_size.get(key)
            if test_name is None:
                continue
            if test_name in mapping:
                collisions.append(test_name)
            mapping[test_name] = gallery_name(info.filename, keep_ext=True)

    return mapping, len(test_infos), len(gallery_infos), collisions


def build_ground_truth(attr_path, query_path, test_to_gallery):
    attrs = list(iter_jsonl(attr_path))
    queries = list(iter_jsonl(query_path))
    if len(attrs) != len(queries):
        raise ValueError(f"attr/query length mismatch: {len(attrs)} != {len(queries)}")

    gt = {}
    mismatched_captions = []
    missing_images = []
    for idx, (attr, query) in enumerate(zip(attrs, queries)):
        attr_caption = attr.get("caption")
        if isinstance(attr_caption, list):
            attr_caption = attr_caption[0] if attr_caption else ""
        query_caption = query.get("caption", "")
        if attr_caption != query_caption:
            mismatched_captions.append(idx)

        test_image = attr["image"]
        gallery_image = test_to_gallery.get(test_image)
        if gallery_image is None:
            missing_images.append(test_image)
            continue
        gt[query["query_index"]] = gallery_image

    return gt, {
        "attrs": len(attrs),
        "queries": len(queries),
        "mismatched_captions": len(mismatched_captions),
        "missing_images": len(missing_images),
        "missing_image_examples": missing_images[:5],
    }


def load_rankings_jsonl(path):
    rows = {}
    for item in iter_jsonl(path):
        rows[item["query_index"]] = [gallery_name(name, keep_ext=True) for name in item["topk_gallery"]]
    return rows


def load_answer_txt(path, query_path):
    query_ids = [item["query_index"] for item in iter_jsonl(query_path)]
    lines = [line.strip() for line in open(path, "r") if line.strip()]
    if len(lines) != len(query_ids):
        raise ValueError(f"answer/query length mismatch: {len(lines)} != {len(query_ids)}")
    rows = {}
    for query_id, line in zip(query_ids, lines):
        rows[query_id] = [f"{stem_or_name(token)}.jpg" for token in line.split()]
    return rows


def evaluate(rankings, ground_truth, max_rank=None):
    ranks = []
    missing_queries = []
    not_found = []
    for query_id, positive in ground_truth.items():
        ranked = rankings.get(query_id)
        if ranked is None:
            missing_queries.append(query_id)
            continue
        if max_rank is not None:
            ranked = ranked[:max_rank]
        try:
            rank = ranked.index(positive) + 1
            ranks.append(rank)
        except ValueError:
            not_found.append(query_id)

    total = len(ground_truth)
    evaluated = len(ranks) + len(not_found)
    ap_values = [(1.0 / rank) for rank in ranks] + [0.0] * len(not_found)
    metrics = {
        "queries": total,
        "evaluated_queries": evaluated,
        "missing_queries": len(missing_queries),
        "not_found": len(not_found),
        "R1": sum(rank <= 1 for rank in ranks) / total * 100.0,
        "R5": sum(rank <= 5 for rank in ranks) / total * 100.0,
        "R10": sum(rank <= 10 for rank in ranks) / total * 100.0,
        "R20": sum(rank <= 20 for rank in ranks) / total * 100.0,
        "R50": sum(rank <= 50 for rank in ranks) / total * 100.0,
        "R100": sum(rank <= 100 for rank in ranks) / total * 100.0,
        "mAP": sum(ap_values) / total * 100.0,
        "mINP": sum(ap_values) / total * 100.0,
    }
    return metrics, {
        "missing_query_examples": missing_queries[:10],
        "not_found_examples": not_found[:10],
        "rank_examples": ranks[:10],
    }


def write_ground_truth(path, ground_truth):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for query_id, image_name in ground_truth.items():
            f.write(json.dumps({"query_index": query_id, "positive": image_name}) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pab_root", required=True, help="PAB2025 dataset root")
    parser.add_argument("--rankings", default=None, help="Ranking JSONL with query_index/topk_gallery.")
    parser.add_argument("--answer", default=None, help="Official answer.txt style ranking.")
    parser.add_argument("--max_rank", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--write_gt", default=None)
    args = parser.parse_args()

    if bool(args.rankings) == bool(args.answer):
        raise SystemExit("Provide exactly one of --rankings or --answer.")

    pab_root = Path(args.pab_root)
    attr_path = pab_root / "annotation/test/attr.json"
    query_path = pab_root / "name-masked_test-set/query.json"
    test_zip_path = pab_root / "test.zip"
    gallery_zip_path = pab_root / "name-masked_test-set/gallery.zip"

    test_to_gallery, test_count, gallery_count, collisions = build_test_to_gallery_map(
        test_zip_path, gallery_zip_path
    )
    ground_truth, gt_info = build_ground_truth(attr_path, query_path, test_to_gallery)

    if args.write_gt:
        write_ground_truth(args.write_gt, ground_truth)

    if args.rankings:
        rankings = load_rankings_jsonl(args.rankings)
        input_path = args.rankings
        input_type = "rankings_jsonl"
    else:
        rankings = load_answer_txt(args.answer, query_path)
        input_path = args.answer
        input_type = "answer_txt"

    metrics, details = evaluate(rankings, ground_truth, max_rank=args.max_rank)
    result = {
        "input": str(Path(input_path).resolve()),
        "input_type": input_type,
        "pab_root": str(pab_root.resolve()),
        "test_images": test_count,
        "gallery_images": gallery_count,
        "test_to_gallery_matches": len(test_to_gallery),
        "crc_size_collisions": len(collisions),
        "ground_truth": gt_info,
        "max_rank": args.max_rank,
        "metrics": {key: round(value, 6) if isinstance(value, float) else value for key, value in metrics.items()},
        "details": details,
    }

    print(json.dumps(result, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
