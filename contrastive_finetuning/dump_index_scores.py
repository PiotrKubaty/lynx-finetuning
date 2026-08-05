from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from contrastive_finetuning.models import build_rdd, build_masked_lg
from contrastive_finetuning.train_common import (
    _lg_scores, _lynx_id, _video_id, batch_features, features_from_batch, seed_all,
)

"""
Scores every (query_frame, candidate) pair in --index with the exact
features_from_batch/batch_features/_lg_scores path eval_pseudo_accuracy uses
(fresh RDD extraction from the raw images, chunked to --chunk_size — default
64, matching helios_scripts/eval_video_accuracy_train_path.sh, which was
shown to reproduce training's own logged val/video_accuracy).

Dumps per-pair scores in the same schema
rdd-benchmark/scripts/lynx_dump_index_scores.py writes (that script re-scores
the identical index file with rdd-benchmark's own LightGlueMasked + cached
features + batching), so the two can be joined by (query_frame, candidate)
path with contrastive_finetuning/compare_index_scores.py — both scripts read
paths relative to the same --data_root / --dataset_root, so no translation
is needed.
"""


def load_image(root: Path | None, rel: str, transform) -> torch.Tensor:
    path = root / rel if root is not None else Path(rel)
    return transform(default_loader(path))


@torch.no_grad()
def score_entry(rdd, lg, entry, root, resize, transform, device, chunk_size):
    cand_paths = list(entry["positives"]) + list(entry["negatives"])
    n_pos = len(entry["positives"])

    query_img = load_image(root, entry["query_frame"], transform).unsqueeze(0)
    cand_imgs = torch.stack([load_image(root, p, transform) for p in cand_paths])

    feats_q, H_q, W_q = features_from_batch(query_img, rdd, resize, device, chunk_size=chunk_size)
    feats_c, H_c, W_c = features_from_batch(cand_imgs, rdd, resize, device, chunk_size=chunk_size)

    n_cand = len(cand_paths)
    data_q = batch_features(feats_q * n_cand, H_q, W_q)
    data_c = batch_features(feats_c, H_c, W_c)
    pred = lg({"image0": data_q, "image1": data_c})
    scores = _lg_scores(pred, data_q, data_c)

    return [
        {"path": p, "lynx_id": _lynx_id(p), "is_positive": i < n_pos,
         "score": float(scores[i]), "n_keypoints": int(feats_c[i]["keypoints"].shape[0])}
        for i, p in enumerate(cand_paths)
    ], int(feats_q[0]["keypoints"].shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index",       type=Path, required=True)
    parser.add_argument("--data_root",   type=Path, default=None)
    parser.add_argument("--rdd_weights", type=str,  default="rdd/weights/RDD-v2.pth")
    parser.add_argument("--lg_weights",  type=str,  default="rdd/weights/RDD_lg-v2.pth")
    parser.add_argument("--resize",      type=int,  default=512)
    parser.add_argument("--top_k",       type=int,  default=512)
    parser.add_argument(
        "--chunk_size", type=int, default=64,
        help="RDD forward-call chunk size for the candidate pool — see "
             "helios_scripts/eval_video_accuracy_train_path.sh (64 matches the "
             "historical --eval_max_gpu_batch the checkpoint was validated with "
             "during training; eval_video_accuracy.py's own default is 4).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out",  type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.index) as f:
        entries = json.load(f)
    print(f"{len(entries)} index entries")

    rdd = build_rdd(args.rdd_weights, device, args.top_k)
    lg = build_masked_lg(device, weights=args.lg_weights)
    for p in rdd.parameters():
        p.requires_grad_(False)
    for p in lg.parameters():
        p.requires_grad_(False)
    lg.eval()
    transform = transforms.ToTensor()

    per_query = []
    for i, entry in enumerate(entries):
        candidates, n_kp_q = score_entry(
            rdd, lg, entry, args.data_root, args.resize, transform, device, args.chunk_size)
        per_query.append({
            "query_frame": entry["query_frame"],
            "video": _video_id(entry["query_frame"]),
            "true_lynx": _lynx_id(entry["query_frame"]),
            "n_keypoints_query": n_kp_q,
            "candidates": candidates,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(entries)}")

    videos: dict = {}
    for pq in per_query:
        best = max(pq["candidates"], key=lambda c: c["score"])
        rec = videos.setdefault(pq["video"], {"true_lynx": pq["true_lynx"], "best_score": -1e9})
        if best["score"] > rec["best_score"]:
            rec.update(best_score=best["score"], best_lynx=best["lynx_id"],
                       best_query_frame=pq["query_frame"], best_candidate=best["path"])

    n_correct = sum(1 for v in videos.values() if v["best_lynx"] == v["true_lynx"])
    print(f"\nvideo_accuracy = {n_correct}/{len(videos)} = {n_correct / len(videos):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "per_query": per_query,
            "videos": videos,
            "video_accuracy": n_correct / len(videos),
            "lg_weights": str(args.lg_weights),
            "chunk_size": args.chunk_size,
        }, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
