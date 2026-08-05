from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from contrastive_finetuning.models import build_rdd, build_masked_lg
from contrastive_finetuning.train_common import (
    _lg_scores, _lynx_id, _video_id, batch_features, features_from_batch, seed_all,
)

"""
Deep dive on ONE video where contrastive_finetuning/dump_index_scores.py
("ours") and rdd-benchmark/scripts/lynx_dump_index_scores.py ("theirs")
disagreed. Answers two questions for every (query_frame, candidate) pair of
that video, all from within this one environment (so nothing needs cross-repo
imports):

  1. Are the FEATURES the same? Every frame is extracted two ways — fresh
     from the image (contrastive_finetuning.extract_train, what "ours" uses)
     and loaded from rdd-benchmark's cache (--cache_dir, what "theirs" uses)
     — and the two keypoint sets/descriptors are compared directly (count,
     nearest-neighbour position/descriptor distance).

  2. Are the SCORES the same GIVEN the same features? Every pair is scored
     four ways: {fresh, cached} features x {pretrained, finetuned} weights —
     all through OUR LightGlueForTraining (the underlying LightGlueMasked
     formulas were already diffed byte-for-byte identical against
     rdd-benchmark's copy in an earlier pass, so running rdd-benchmark's own
     class here would not add information the way varying the FEATURES does).
     If cached-features scores land close to what dump_index_scores_theirs.json
     reported for the same pairs, the LightGlue math agrees and the whole gap
     is upstream, in feature extraction. If they don't, the gap survives even
     with identical inputs and is in the scoring code itself.

     Pairs are scored one at a time (no batching) so a mismatch can never be
     blamed on batch composition — that variable was already isolated and
     characterized separately (helios_scripts/debug_video_mismatch.sh).

--index is only read to list which query/candidate frames belong to --video
and which candidates are positive — since both dump scripts iterate the same
JSON file, this pool is *guaranteed* identical between "ours" and "theirs" by
construction; it's printed here for visibility, not because it could differ.
"""


def load_cached_feat(cache_dir: Path, rel: str):
    rel_p = Path(rel)
    path = cache_dir / rel_p.parent / f"{rel_p.stem}.npz"
    data = np.load(path)
    return {
        "keypoints": torch.from_numpy(data["keypoints"]).float(),
        "descriptors": torch.from_numpy(data["descriptors"]).float(),
        "image_size": data["image_size"],  # (H, W)
    }


def load_image(root, rel, transform):
    path = root / rel if root is not None else Path(rel)
    return transform(default_loader(path))


@torch.no_grad()
def fresh_feat(rdd, image_root, rel, resize, transform, device):
    img = load_image(image_root, rel, transform).unsqueeze(0)
    feats, h, w = features_from_batch(img, rdd, resize, device, chunk_size=None)
    return {"keypoints": feats[0]["keypoints"], "descriptors": feats[0]["descriptors"],
            "image_size": (h, w)}


def compare_features(fresh: dict, cached: dict) -> dict:
    """Nearest-neighbour comparison: for every fresh keypoint, the closest
    cached keypoint by pixel distance, and the descriptor cosine similarity
    at that match — cheap, order-independent way to tell "same points,
    same descriptors" from "different detector output" without assuming the
    two extractions return points in the same order."""
    fk, ck = fresh["keypoints"], cached["keypoints"].to(fresh["keypoints"].device)
    fd, cd = fresh["descriptors"], cached["descriptors"].to(fresh["descriptors"].device)
    out = {
        "n_kp_fresh": int(fk.shape[0]), "n_kp_cached": int(ck.shape[0]),
        "image_size_fresh": [int(v) for v in fresh["image_size"]],
        "image_size_cached": [int(v) for v in cached["image_size"]],
    }
    if fk.shape[0] == 0 or ck.shape[0] == 0:
        out.update(mean_nn_pixel_dist=None, mean_nn_cos_sim=None)
        return out
    d = torch.cdist(fk, ck)  # (n_fresh, n_cached)
    nn_dist, nn_idx = d.min(dim=1)
    cos_sim = torch.nn.functional.cosine_similarity(fd, cd[nn_idx], dim=1)
    out.update(
        mean_nn_pixel_dist=float(nn_dist.mean()),
        max_nn_pixel_dist=float(nn_dist.max()),
        mean_nn_cos_sim=float(cos_sim.mean()),
        min_nn_cos_sim=float(cos_sim.min()),
    )
    return out


@torch.no_grad()
def score_pair(lg, q_feat: dict, c_feat: dict, device) -> float:
    q = {"keypoints": q_feat["keypoints"].to(device), "descriptors": q_feat["descriptors"].to(device)}
    c = {"keypoints": c_feat["keypoints"].to(device), "descriptors": c_feat["descriptors"].to(device)}
    h_q, w_q = q_feat["image_size"]
    h_c, w_c = c_feat["image_size"]
    data_q = batch_features([q], h_q, w_q)
    data_c = batch_features([c], h_c, w_c)
    pred = lg({"image0": data_q, "image1": data_c})
    return float(_lg_scores(pred, data_q, data_c)[0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--video", type=str, required=True,
                   help="e.g. test/lynx_19/Kudłoń_PodBacą_12f/0010 — one of the "
                        "videos compare_index_scores.py flagged as DISAGREE")
    p.add_argument("--data_root", type=Path, required=True, help="For fresh extraction.")
    p.add_argument("--cache_dir", type=Path, required=True,
                   help="rdd-benchmark's cached-feature dir (theirs' input).")
    p.add_argument("--rdd_weights", type=str, default="rdd/weights/RDD-v2.pth")
    p.add_argument("--pretrained_lg_weights", type=str, default="rdd/weights/RDD_lg-v2.pth")
    p.add_argument("--finetuned_lg_weights", type=str, required=True)
    p.add_argument("--resize", type=int, default=512)
    p.add_argument("--top_k", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.index) as f:
        all_entries = json.load(f)
    entries = [e for e in all_entries if _video_id(e["query_frame"]) == args.video]
    if not entries:
        raise ValueError(f"No entries for video={args.video!r} in {args.index}")
    true_lynx = _lynx_id(entries[0]["query_frame"])
    print(f"video={args.video} true_lynx={true_lynx}: {len(entries)} query frame(s)")
    for e in entries:
        print(f"  query={Path(e['query_frame']).name}  "
              f"positives={[Path(p).name for p in e['positives']]}  "
              f"negatives={[Path(n).name for n in e['negatives']]}")

    rdd = build_rdd(args.rdd_weights, device, args.top_k)
    lg_pre = build_masked_lg(device, weights=args.pretrained_lg_weights)
    lg_ft = build_masked_lg(device, weights=args.finetuned_lg_weights)
    for model in (rdd, lg_pre, lg_ft):
        for prm in model.parameters():
            prm.requires_grad_(False)
    lg_pre.eval()
    lg_ft.eval()
    transform = transforms.ToTensor()

    # Cache extracted features per unique frame — a video's candidates repeat
    # across its query frames' entries, so this avoids re-running RDD/re-
    # loading the cache for the same path.
    fresh_cache: dict = {}
    cached_cache: dict = {}

    def get_fresh(rel):
        if rel not in fresh_cache:
            fresh_cache[rel] = fresh_feat(rdd, args.data_root, rel, args.resize, transform, device)
        return fresh_cache[rel]

    def get_cached(rel):
        if rel not in cached_cache:
            cached_cache[rel] = load_cached_feat(args.cache_dir, rel)
        return cached_cache[rel]

    results = []
    for entry in entries:
        q_rel = entry["query_frame"]
        q_fresh, q_cached = get_fresh(q_rel), get_cached(q_rel)
        cand_rels = list(entry["positives"]) + list(entry["negatives"])
        n_pos = len(entry["positives"])

        for i, c_rel in enumerate(cand_rels):
            c_fresh, c_cached = get_fresh(c_rel), get_cached(c_rel)
            feat_cmp_q = compare_features(q_fresh, q_cached)
            feat_cmp_c = compare_features(c_fresh, c_cached)

            row = {
                "query_frame": q_rel, "candidate": c_rel,
                "lynx_id": _lynx_id(c_rel), "is_positive": i < n_pos,
                "features_query": feat_cmp_q, "features_candidate": feat_cmp_c,
                "score_pretrained_fresh":  score_pair(lg_pre, q_fresh,  c_fresh,  device),
                "score_pretrained_cached": score_pair(lg_pre, q_cached, c_cached, device),
                "score_finetuned_fresh":   score_pair(lg_ft,  q_fresh,  c_fresh,  device),
                "score_finetuned_cached":  score_pair(lg_ft,  q_cached, c_cached, device),
            }
            results.append(row)
            print(
                f"  {Path(q_rel).name} vs {Path(c_rel).name} ({row['lynx_id']}"
                f"{'*' if row['is_positive'] else ''}): "
                f"n_kp fresh/cached q={feat_cmp_q['n_kp_fresh']}/{feat_cmp_q['n_kp_cached']} "
                f"c={feat_cmp_c['n_kp_fresh']}/{feat_cmp_c['n_kp_cached']}  "
                f"nn_cos_sim q={feat_cmp_q['mean_nn_cos_sim']} c={feat_cmp_c['mean_nn_cos_sim']}  |  "
                f"pretrained fresh={row['score_pretrained_fresh']:.4f} cached={row['score_pretrained_cached']:.4f}  "
                f"finetuned fresh={row['score_finetuned_fresh']:.4f} cached={row['score_finetuned_cached']:.4f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"video": args.video, "true_lynx": true_lynx, "pairs": results}, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
