from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from contrastive_finetuning.models import build_lightglue, build_rdd
from contrastive_finetuning.pair_quality import (
    CACHE_VERSION,
    assign_quality_bands,
    compute_composite_score,
    normalize_path_str,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build offline pair-quality cache for training mining.")
    p.add_argument("--train_data", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--rdd_weights", type=str, default="rdd/weights/RDD-v2.pth")
    p.add_argument("--lg_weights", type=str, default="rdd/weights/RDD_lg-v2.pth")
    p.add_argument("--resize", type=int, default=256)
    p.add_argument("--top_k", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_anchors", type=int, default=0, help="Debug: limit anchors scored (0 = all).")
    p.add_argument("--max_positive_candidates_per_anchor", type=int, default=32)
    p.add_argument("--max_negative_candidates_per_anchor", type=int, default=32)
    p.add_argument(
        "--sequence_aware_sampling",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y", "on"},
        default=True,
        help="Use sequence-aware structural pools when sampling candidates.",
    )
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_image_tensor(path: Path, resize: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if resize > 0:
        w, h = img.size
        scale = resize / max(w, h)
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return transforms.ToTensor()(img)


def resize_tensor_long_side(tensor: torch.Tensor, size: int) -> torch.Tensor:
    _, h, w = tensor.shape
    scale = size / max(h, w)
    new_h = max(32, int(h * scale) // 32 * 32)
    new_w = max(32, int(w * scale) // 32 * 32)
    return torch.nn.functional.interpolate(
        tensor.unsqueeze(0), (new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)


@torch.no_grad()
def extract_features(rdd: torch.nn.Module, image: torch.Tensor, device: torch.device) -> dict:
    image = image.unsqueeze(0).to(device)
    out = rdd.extract(image)[0]
    return {
        "keypoints": out["keypoints"].detach().cpu(),
        "descriptors": out["descriptors"].detach().cpu(),
        "image_size_wh": torch.tensor([image.shape[-1], image.shape[-2]], dtype=torch.int32),
    }


@torch.no_grad()
def move_features_to_device(feat: dict, device: torch.device) -> dict:
    return {
        "keypoints": feat["keypoints"].to(device, non_blocking=True),
        "descriptors": feat["descriptors"].to(device, non_blocking=True),
        "image_size_wh": feat["image_size_wh"].to(device, non_blocking=True),
    }


@torch.no_grad()
def compute_mutual_nn_rate_torch(desc0: torch.Tensor, desc1: torch.Tensor) -> float:
    if desc0.shape[0] == 0 or desc1.shape[0] == 0:
        return 0.0
    sim = desc0 @ desc1.transpose(0, 1)
    row_best = sim.argmax(dim=1)
    col_best = sim.argmax(dim=0)
    row_ids = torch.arange(sim.shape[0], device=sim.device)
    mutual = row_ids == col_best[row_best]
    return float(mutual.float().mean().item())


@torch.no_grad()
def score_pair(
    lg: torch.nn.Module,
    feat_a: dict,
    feat_b: dict,
) -> dict[str, float]:
    k0 = feat_a["keypoints"].unsqueeze(0)
    k1 = feat_b["keypoints"].unsqueeze(0)
    d0 = feat_a["descriptors"].unsqueeze(0)
    d1 = feat_b["descriptors"].unsqueeze(0)
    size0 = feat_a["image_size_wh"].unsqueeze(0)
    size1 = feat_b["image_size_wh"].unsqueeze(0)

    pred = lg({"image0": {"keypoints": k0, "descriptors": d0, "image_size": size0},
               "image1": {"keypoints": k1, "descriptors": d1, "image_size": size1}})

    scores = pred["scores"][0]
    matches = pred["matches"][0]
    match_count = int(matches.shape[0]) if matches.numel() > 0 else 0
    sum_conf = float(scores.sum().item()) if scores.numel() > 0 else 0.0
    mean_conf = float(scores.mean().item()) if scores.numel() > 0 else 0.0

    n_kpts_a = max(1, int(feat_a["keypoints"].shape[0]))
    n_kpts_b = max(1, int(feat_b["keypoints"].shape[0]))
    normalized_sum_conf = sum_conf / min(n_kpts_a, n_kpts_b)

    if match_count > 0:
        matched_anchor = int(torch.unique(matches[:, 0]).numel())
        anchor_participation = float(matched_anchor / n_kpts_a)
    else:
        anchor_participation = 0.0

    mutual_nn = compute_mutual_nn_rate_torch(feat_a["descriptors"], feat_b["descriptors"])
    composite = compute_composite_score(normalized_sum_conf, anchor_participation, mutual_nn)

    return {
        "match_count": match_count,
        "mean_confidence": mean_conf,
        "sum_confidence": sum_conf,
        "normalized_sum_conf": normalized_sum_conf,
        "anchor_participation": anchor_participation,
        "mutual_nn_rate": mutual_nn,
        "composite_score": composite,
    }


def build_metadata_index(root: Path) -> tuple[list[tuple[str, int]], dict[int, dict], dict]:
    dataset = ImageFolder(root=str(root), transform=None)
    index_to_path: list[str] = []
    sample_meta: dict[int, dict] = {}
    class_to_indices: dict[int, list[int]] = defaultdict(list)
    class_to_source_to_indices: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    class_to_source_sequence_to_indices: dict[int, dict[tuple[str, str], list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for idx, (sample_path, label) in enumerate(dataset.samples):
        path = Path(sample_path)
        rel_parts = path.relative_to(root).parts
        if len(rel_parts) < 4:
            raise ValueError(f"Expected identity/source/sequence/frame hierarchy, got {path}")
        identity, source_id, sequence_id = rel_parts[0], rel_parts[1], rel_parts[2]
        index_to_path.append(str(path.resolve()))
        sample_meta[idx] = {
            "label": label,
            "identity": identity,
            "source_id": source_id,
            "sequence_id": sequence_id,
            "path": str(path.resolve()),
        }
        class_to_indices[label].append(idx)
        class_to_source_to_indices[label][source_id].append(idx)
        class_to_source_sequence_to_indices[label][(source_id, sequence_id)].append(idx)

    lookups = {
        "class_to_indices": class_to_indices,
        "class_to_source_to_indices": class_to_source_to_indices,
        "class_to_source_sequence_to_indices": class_to_source_sequence_to_indices,
    }
    return index_to_path, sample_meta, lookups


def sample_positive_candidate_indices(
    anchor_idx: int,
    sample_meta: dict[int, dict],
    lookups: dict,
    rng: random.Random,
    max_candidates: int,
    sequence_aware: bool,
) -> list[int]:
    label = sample_meta[anchor_idx]["label"]
    anchor = sample_meta[anchor_idx]
    class_to_indices = lookups["class_to_indices"]
    class_to_source_to_indices = lookups["class_to_source_to_indices"]
    class_to_source_sequence_to_indices = lookups["class_to_source_sequence_to_indices"]

    pool: list[int] = []
    if sequence_aware:
        cross_source = [
            i
            for source_id, indices in class_to_source_to_indices[label].items()
            if source_id != anchor["source_id"]
            for i in indices
            if i != anchor_idx
        ]
        cross_sequence = [
            i
            for (source_id, sequence_id), indices in class_to_source_sequence_to_indices[label].items()
            if source_id == anchor["source_id"] and sequence_id != anchor["sequence_id"]
            for i in indices
            if i != anchor_idx
        ]
        same_sequence = [
            i
            for i in class_to_source_sequence_to_indices[label][(anchor["source_id"], anchor["sequence_id"])]
            if i != anchor_idx
        ]
        for tier in (cross_source, cross_sequence, same_sequence):
            rng.shuffle(tier)
            pool.extend(tier)
    else:
        pool = [i for i in class_to_indices[label] if i != anchor_idx]
        rng.shuffle(pool)

    if len(pool) > max_candidates:
        pool = pool[:max_candidates]
    return pool


def sample_negative_candidate_indices(
    anchor_idx: int,
    sample_meta: dict[int, dict],
    lookups: dict,
    rng: random.Random,
    max_candidates: int,
) -> list[int]:
    label = sample_meta[anchor_idx]["label"]
    class_to_indices = lookups["class_to_indices"]
    all_labels = [l for l in class_to_indices.keys() if l != label]
    if not all_labels:
        return []

    pool: list[int] = []
    rng.shuffle(all_labels)
    for neg_label in all_labels:
        candidates = class_to_indices[neg_label]
        rng.shuffle(candidates)
        pool.extend(candidates)
        if len(pool) >= max_candidates:
            break
    return pool[:max_candidates]


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    device_name = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(device_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index_to_path, sample_meta, lookups = build_metadata_index(args.train_data)
    path_to_index = {p: i for i, p in enumerate(index_to_path)}

    anchor_indices = list(range(len(index_to_path)))
    if args.max_anchors > 0:
        rng = random.Random(args.seed)
        anchor_indices = rng.sample(anchor_indices, min(args.max_anchors, len(anchor_indices)))

    print(f"Building pair-quality cache for {len(anchor_indices)} anchors...", flush=True)
    rdd = build_rdd(args.rdd_weights, device, args.top_k)
    lg = build_lightglue(device, weights=args.lg_weights)

    feature_cache: dict[int, dict] = {}

    def get_features(idx: int) -> dict:
        if idx not in feature_cache:
            tensor = load_image_tensor(Path(index_to_path[idx]), args.resize)
            tensor = resize_tensor_long_side(tensor, args.resize)
            feature_cache[idx] = extract_features(rdd, tensor, device)
        return feature_cache[idx]

    positive_candidates_by_anchor: dict[int, list[dict]] = {}
    negative_candidates_by_anchor: dict[int, list[dict]] = {}
    rng = random.Random(args.seed)

    total_anchors = len(anchor_indices)
    processed_pairs = 0
    start_time = time.time()
    next_eta_log_time = start_time + 60.0

    for processed_anchors, anchor_idx in enumerate(
        tqdm(anchor_indices, desc="Scoring anchors"), start=1
    ):
        pos_indices = sample_positive_candidate_indices(
            anchor_idx,
            sample_meta,
            lookups,
            rng,
            args.max_positive_candidates_per_anchor,
            args.sequence_aware_sampling,
        )
        neg_indices = sample_negative_candidate_indices(
            anchor_idx,
            sample_meta,
            lookups,
            rng,
            args.max_negative_candidates_per_anchor,
        )
        processed_pairs += len(pos_indices) + len(neg_indices)

        feat_a_dev = move_features_to_device(get_features(anchor_idx), device)

        pos_records: list[dict] = []
        for cand_idx in pos_indices:
            cand_meta = sample_meta[cand_idx]
            feat_b_dev = move_features_to_device(get_features(cand_idx), device)
            metrics = score_pair(lg, feat_a_dev, feat_b_dev)
            pos_records.append(
                {
                    "candidate_index": cand_idx,
                    "candidate_path": cand_meta["path"],
                    "same_identity": True,
                    "source_id": cand_meta["source_id"],
                    "sequence_id": cand_meta["sequence_id"],
                    **metrics,
                }
            )
        pos_records = assign_quality_bands(pos_records)
        positive_candidates_by_anchor[anchor_idx] = pos_records

        neg_records: list[dict] = []
        for cand_idx in neg_indices:
            cand_meta = sample_meta[cand_idx]
            feat_b_dev = move_features_to_device(get_features(cand_idx), device)
            metrics = score_pair(lg, feat_a_dev, feat_b_dev)
            neg_records.append(
                {
                    "candidate_index": cand_idx,
                    "candidate_path": cand_meta["path"],
                    "same_identity": False,
                    "source_id": cand_meta["source_id"],
                    "sequence_id": cand_meta["sequence_id"],
                    **metrics,
                }
            )
        neg_records = sorted(neg_records, key=lambda r: r["composite_score"], reverse=True)
        negative_candidates_by_anchor[anchor_idx] = neg_records

        now = time.time()
        if now >= next_eta_log_time or processed_anchors == total_anchors:
            elapsed = max(1e-6, now - start_time)
            anchors_per_sec = processed_anchors / elapsed
            pairs_per_sec = processed_pairs / elapsed
            remaining_anchors = max(0, total_anchors - processed_anchors)
            eta_seconds = remaining_anchors / max(1e-6, anchors_per_sec)
            eta_minutes = eta_seconds / 60.0
            elapsed_minutes = elapsed / 60.0
            print(
                (
                    "[progress] anchors "
                    f"{processed_anchors}/{total_anchors} "
                    f"({processed_anchors / max(1, total_anchors) * 100:.1f}%), "
                    f"pairs_scored={processed_pairs}, "
                    f"anchors/s={anchors_per_sec:.2f}, pairs/s={pairs_per_sec:.2f}, "
                    f"elapsed={elapsed_minutes:.1f}m, eta={eta_minutes:.1f}m"
                ),
                flush=True,
            )
            next_eta_log_time = now + 60.0

    metadata = {
        "cache_version": CACHE_VERSION,
        "data_root": normalize_path_str(args.train_data),
        "rdd_weights": normalize_path_str(args.rdd_weights),
        "lg_weights": normalize_path_str(args.lg_weights),
        "resize": args.resize,
        "top_k": args.top_k,
        "seed": args.seed,
        "sequence_aware_sampling": args.sequence_aware_sampling,
        "max_positive_candidates_per_anchor": args.max_positive_candidates_per_anchor,
        "max_negative_candidates_per_anchor": args.max_negative_candidates_per_anchor,
        "num_anchors": len(anchor_indices),
    }

    pairs = {
        "index_to_path": index_to_path,
        "path_to_index": path_to_index,
        "positive_candidates_by_anchor": positive_candidates_by_anchor,
        "negative_candidates_by_anchor": negative_candidates_by_anchor,
    }

    metadata_path = args.output_dir / "metadata.json"
    pairs_path = args.output_dir / "pairs.pt"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    torch.save(pairs, pairs_path)
    print(f"Saved cache to {args.output_dir}", flush=True)
    print(f"  metadata: {metadata_path}", flush=True)
    print(f"  pairs:    {pairs_path}", flush=True)


if __name__ == "__main__":
    main()
