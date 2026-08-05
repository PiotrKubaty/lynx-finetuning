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
Isolates why ONE video's pseudo-accuracy prediction changes with the RDD
candidate-pool chunk size — `chunk_size` inside eval_pseudo_accuracy,
controlled by --batch_size in both train_by_lg_matches.py (--eval_only) and
eval_video_accuracy.py. See helios_scripts/eval_video_accuracy_train_path.sh
for why that chunk size matters: it used to be a dedicated --eval_max_gpu_batch
flag (default 64), removed and folded into --batch_size (default 4 in
eval_video_accuracy.py) after the checkpoints under test were trained.

For every query frame belonging to --video, this runs RDD+LightGlue against
its full positive+negative candidate pool once per value in --chunk_sizes,
using the exact same features_from_batch/batch_features/_lg_scores building
blocks eval_pseudo_accuracy itself calls — just single-process, restricted to
one video, with chunk_size swept explicitly instead of fixed by --batch_size.

For each (query_frame, chunk_size) this prints the best positive/negative
score AND the keypoint count RDD detected for the query and for the winning
candidates. Comparing those keypoint counts across chunk sizes tells apart:
  - a detection-threshold flip (keypoint count changes -> RDD saw a
    genuinely different set of points for the same image), from
  - plain floating-point drift (same keypoint counts, match confidence still
    moves by a few thousandths — batched GPU matmuls/attention are not
    exactly invariant to how many images share a forward call).
A final per-video section mirrors eval_pseudo_accuracy's own aggregation
(highest-scoring query frame across the whole video picks the video's
predicted lynx_id) so a flip in *which query frame wins*, not just which
candidate wins for a fixed query frame, also shows up.
"""


def load_entries(index_path: Path, video: str) -> list[dict]:
    with open(index_path) as f:
        all_entries = json.load(f)
    entries = [e for e in all_entries if _video_id(e["query_frame"]) == video]
    if not entries:
        raise ValueError(f"No entries for video={video!r} in {index_path}")
    return entries


def load_image(root: Path | None, rel: str, transform) -> torch.Tensor:
    path = root / rel if root is not None else Path(rel)
    return transform(default_loader(path))


@torch.no_grad()
def run_one_chunk_size(
    device: torch.device,
    rdd: torch.nn.Module,
    lg: torch.nn.Module,
    entries: list[dict],
    root: Path | None,
    resize: int,
    transform,
    chunk_size: int,
) -> list[dict]:
    results = []
    for entry in entries:
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
        scores = _lg_scores(pred, data_q, data_c)  # (n_cand,)

        pos_scores, neg_scores = scores[:n_pos], scores[n_pos:]
        best_pos_i = int(pos_scores.argmax())
        best_neg_i = int(neg_scores.argmax())
        best_i = int(scores.argmax())

        results.append({
            "query_frame":    entry["query_frame"],
            "n_kp_query":     int(feats_q[0]["keypoints"].shape[0]),
            "best_pos_score": float(pos_scores[best_pos_i]),
            "best_pos_path":  cand_paths[best_pos_i],
            "n_kp_best_pos":  int(feats_c[best_pos_i]["keypoints"].shape[0]),
            "best_neg_score": float(neg_scores[best_neg_i]),
            "best_neg_path":  cand_paths[n_pos + best_neg_i],
            "n_kp_best_neg":  int(feats_c[n_pos + best_neg_i]["keypoints"].shape[0]),
            "best_score":     float(scores[best_i]),
            "best_path":      cand_paths[best_i],
        })
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index",       type=Path, required=True)
    parser.add_argument("--data_root",   type=Path, default=None)
    parser.add_argument("--rdd_weights", type=str,  default="rdd/weights/RDD-v2.pth")
    parser.add_argument("--lg_weights",  type=str,  default="rdd/weights/RDD_lg-v2.pth")
    parser.add_argument("--video",       type=str,  required=True,
                         help="Video id as it appears in MISMATCH logs, e.g. "
                              "test/lynx_19/Kiczora_Turniska_177a/0059")
    parser.add_argument("--resize",      type=int,  default=512)
    parser.add_argument("--top_k",       type=int,  default=512)
    parser.add_argument("--chunk_sizes", type=str,  default="1,2,4,8,16,32,64",
                         help="Comma-separated RDD forward-call chunk sizes to sweep")
    parser.add_argument("--seed",        type=int,  default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    entries = load_entries(args.index, args.video)
    true_lynx = _lynx_id(entries[0]["query_frame"])
    print(f"video={args.video} true_lynx={true_lynx}: {len(entries)} query frame(s) in index")

    rdd = build_rdd(args.rdd_weights, device, args.top_k)
    lg = build_masked_lg(device, weights=args.lg_weights)
    for p in rdd.parameters():
        p.requires_grad_(False)
    for p in lg.parameters():
        p.requires_grad_(False)
    lg.eval()

    transform = transforms.ToTensor()
    chunk_sizes = [int(c) for c in args.chunk_sizes.split(",")]

    all_results = {
        cs: run_one_chunk_size(device, rdd, lg, entries, args.data_root, args.resize, transform, cs)
        for cs in chunk_sizes
    }

    for cs in chunk_sizes:
        print(f"\n=== chunk_size={cs} ===")
        for r in all_results[cs]:
            margin = r["best_pos_score"] - r["best_neg_score"]
            print(
                f"  query={Path(r['query_frame']).name} n_kp_q={r['n_kp_query']} | "
                f"best_pos={r['best_pos_score']:.4f} (n_kp={r['n_kp_best_pos']}, {Path(r['best_pos_path']).name}) | "
                f"best_neg={r['best_neg_score']:.4f} (n_kp={r['n_kp_best_neg']}, {Path(r['best_neg_path']).name}) | "
                f"margin={margin:+.4f}"
            )

    print("\n=== video-level winner per chunk_size (mirrors eval_pseudo_accuracy's aggregation) ===")
    for cs in chunk_sizes:
        best = max(all_results[cs], key=lambda r: r["best_score"])
        predicted_lynx = _lynx_id(best["best_path"])
        verdict = "OK" if predicted_lynx == true_lynx else "WRONG"
        print(
            f"  chunk_size={cs:>3}: query_frame={Path(best['query_frame']).name} "
            f"best_score={best['best_score']:.4f} -> predicted_lynx={predicted_lynx} ({verdict})"
        )

    lo_cs, hi_cs = chunk_sizes[0], chunk_sizes[-1]
    print(f"\n=== per-query diff: chunk_size={lo_cs} vs chunk_size={hi_cs} ===")
    for r_lo, r_hi in zip(all_results[lo_cs], all_results[hi_cs]):
        assert r_lo["query_frame"] == r_hi["query_frame"]
        kp_changed = (
            r_lo["n_kp_query"] != r_hi["n_kp_query"]
            or r_lo["n_kp_best_pos"] != r_hi["n_kp_best_pos"]
            or r_lo["n_kp_best_neg"] != r_hi["n_kp_best_neg"]
        )
        cand_changed = (
            r_lo["best_pos_path"] != r_hi["best_pos_path"]
            or r_lo["best_neg_path"] != r_hi["best_neg_path"]
        )
        print(
            f"  query={Path(r_lo['query_frame']).name}: "
            f"d(best_pos)={r_hi['best_pos_score'] - r_lo['best_pos_score']:+.4f}  "
            f"d(best_neg)={r_hi['best_neg_score'] - r_lo['best_neg_score']:+.4f}  "
            f"keypoint_counts_changed={kp_changed}  winning_candidate_changed={cand_changed}"
        )


if __name__ == "__main__":
    main()
