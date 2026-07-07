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

from contrastive_finetuning.models import build_masked_lg, build_rdd
from contrastive_finetuning.pair_quality import (
    CACHE_VERSION,
    assign_quality_bands,
    compute_composite_score,
    normalize_path_str,
)
from contrastive_finetuning.process import align_tensors_to_max_length


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build offline pair-quality cache for training mining.")
    p.add_argument("--train_data", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--rdd_weights", type=str, default="rdd/weights/RDD-v2.pth")
    p.add_argument("--lg_weights", type=str, default="rdd/weights/RDD_lg-v2.pth")
    p.add_argument("--resize", type=int, default=512)
    p.add_argument("--top_k", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_anchors", type=int, default=0, help="Debug: limit anchors scored (0 = all).")
    p.add_argument("--max_positive_candidates_per_anchor", type=int, default=32)
    p.add_argument("--max_negative_candidates_per_anchor", type=int, default=32)
    p.add_argument(
        "--rdd_batch_size",
        type=int,
        default=8,
        help="Micro-batch size for RDD feature extraction over same-shape resized images.",
    )
    p.add_argument(
        "--lg_batch_size",
        type=int,
        default=16,
        help="Micro-batch size for batched LightGlue scoring per anchor.",
    )
    p.add_argument(
        "--disable_lg_batching",
        action="store_true",
        help="Score one candidate pair at a time (debug/parity).",
    )
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


def load_image_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
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
def extract_features_batch(rdd: torch.nn.Module, images: torch.Tensor, device: torch.device) -> list[dict]:
    images = images.to(device, non_blocking=True)
    outs = rdd.extract(images)
    image_size_wh = torch.tensor([images.shape[-1], images.shape[-2]], dtype=torch.int32)
    return [
        {
            "keypoints": out["keypoints"].detach().cpu(),
            "descriptors": out["descriptors"].detach().cpu(),
            "image_size_wh": image_size_wh.clone(),
        }
        for out in outs
    ]


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


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def metrics_from_pair(
    feat_a: dict,
    feat_b: dict,
    matches: torch.Tensor,
    scores: torch.Tensor,
) -> dict[str, float]:
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


def _anchor_masks(feat_a: dict, batch_size: int) -> torch.Tensor:
    m = int(feat_a["keypoints"].shape[0])
    device = feat_a["keypoints"].device
    return torch.ones(batch_size, 1, m, 1, dtype=torch.bool, device=device)


def _collate_image0_batch(feat_a: dict, batch_size: int) -> dict:
    return {
        "keypoints": feat_a["keypoints"].unsqueeze(0).expand(batch_size, -1, -1).contiguous(),
        "descriptors": feat_a["descriptors"].unsqueeze(0).expand(batch_size, -1, -1).contiguous(),
        "image_size": feat_a["image_size_wh"].unsqueeze(0).expand(batch_size, -1).contiguous(),
        "masks": _anchor_masks(feat_a, batch_size),
    }


def _collate_image1_batch(feats_b: list[dict]) -> dict:
    ks = [f["keypoints"] for f in feats_b]
    ds = [f["descriptors"] for f in feats_b]
    sizes = torch.stack([f["image_size_wh"] for f in feats_b])
    ks_pad, masks = align_tensors_to_max_length(ks)
    ds_pad, _ = align_tensors_to_max_length(ds)
    return {
        "keypoints": ks_pad,
        "descriptors": ds_pad,
        "image_size": sizes,
        "masks": masks.unsqueeze(1),
    }


@torch.no_grad()
def score_pair(
    lg: torch.nn.Module,
    feat_a: dict,
    feat_b: dict,
) -> dict[str, float]:
    data0 = _collate_image0_batch(feat_a, 1)
    data1 = _collate_image1_batch([feat_b])
    pred = lg({"image0": data0, "image1": data1})
    return metrics_from_pair(feat_a, feat_b, pred["matches"][0], pred["scores"][0])


@torch.no_grad()
def score_pairs_batch(
    lg: torch.nn.Module,
    feat_a: dict,
    feats_b: list[dict],
) -> list[dict[str, float]]:
    if not feats_b:
        return []
    batch_size = len(feats_b)
    data0 = _collate_image0_batch(feat_a, batch_size)
    data1 = _collate_image1_batch(feats_b)
    pred = lg({"image0": data0, "image1": data1})
    return [
        metrics_from_pair(feat_a, feat_b, pred["matches"][i], pred["scores"][i])
        for i, feat_b in enumerate(feats_b)
    ]


@torch.no_grad()
def score_candidates_with_fallback(
    lg: torch.nn.Module,
    feat_a: dict,
    feats_b: list[dict],
    batch_size: int,
    use_batching: bool,
) -> list[dict[str, float]]:
    if not feats_b:
        return []

    if not use_batching or batch_size <= 1:
        return [score_pair(lg, feat_a, feat_b) for feat_b in feats_b]

    results: list[dict[str, float]] = []
    cursor = 0
    while cursor < len(feats_b):
        chunk = feats_b[cursor : cursor + batch_size]
        attempt_size = len(chunk)
        while True:
            sub = chunk[:attempt_size]
            try:
                results.extend(score_pairs_batch(lg, feat_a, sub))
                cursor += attempt_size
                if attempt_size < len(chunk):
                    chunk = chunk[attempt_size:]
                    attempt_size = len(chunk)
                    continue
                break
            except RuntimeError as exc:
                if _is_cuda_oom(exc) and attempt_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    attempt_size = max(1, attempt_size // 2)
                    continue
                raise
    return results


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


def _build_candidate_records(
    candidate_indices: list[int],
    sample_meta: dict[int, dict],
    same_identity: bool,
    metrics_list: list[dict[str, float]],
) -> list[dict]:
    return [
        {
            "candidate_index": cand_idx,
            "candidate_path": sample_meta[cand_idx]["path"],
            "same_identity": same_identity,
            "source_id": sample_meta[cand_idx]["source_id"],
            "sequence_id": sample_meta[cand_idx]["sequence_id"],
            **metrics,
        }
        for cand_idx, metrics in zip(candidate_indices, metrics_list)
    ]


def build_anchor_candidate_pools(
    anchor_indices: list[int],
    sample_meta: dict[int, dict],
    lookups: dict,
    rng: random.Random,
    *,
    max_positive_candidates_per_anchor: int,
    max_negative_candidates_per_anchor: int,
    sequence_aware_sampling: bool,
) -> tuple[dict[int, list[int]], dict[int, list[int]], int, list[int]]:
    positive_indices_by_anchor: dict[int, list[int]] = {}
    negative_indices_by_anchor: dict[int, list[int]] = {}
    required_feature_indices: set[int] = set()
    total_pairs = 0

    for anchor_idx in anchor_indices:
        pos_indices = sample_positive_candidate_indices(
            anchor_idx,
            sample_meta,
            lookups,
            rng,
            max_positive_candidates_per_anchor,
            sequence_aware_sampling,
        )
        neg_indices = sample_negative_candidate_indices(
            anchor_idx,
            sample_meta,
            lookups,
            rng,
            max_negative_candidates_per_anchor,
        )
        positive_indices_by_anchor[anchor_idx] = pos_indices
        negative_indices_by_anchor[anchor_idx] = neg_indices
        required_feature_indices.add(anchor_idx)
        required_feature_indices.update(pos_indices)
        required_feature_indices.update(neg_indices)
        total_pairs += len(pos_indices) + len(neg_indices)

    return (
        positive_indices_by_anchor,
        negative_indices_by_anchor,
        total_pairs,
        sorted(required_feature_indices),
    )


def preextract_feature_cache(
    index_to_path: list[str],
    required_indices: list[int],
    rdd: torch.nn.Module,
    device: torch.device,
    resize: int,
    rdd_batch_size: int,
) -> dict[int, dict]:
    feature_cache: dict[int, dict] = {}
    total_images = len(required_indices)
    start_time = time.time()
    next_eta_log_time = start_time + 60.0
    pending_by_shape: dict[tuple[int, int], list[tuple[int, torch.Tensor]]] = defaultdict(list)
    processed_images = 0
    shape_bucket_count = 0

    def flush_entries(entries: list[tuple[int, torch.Tensor]]) -> None:
        nonlocal processed_images, next_eta_log_time
        if not entries:
            return

        cursor = 0
        while cursor < len(entries):
            chunk = entries[cursor : cursor + rdd_batch_size]
            attempt_size = len(chunk)
            while True:
                sub = chunk[:attempt_size]
                try:
                    batch = torch.stack([tensor for _, tensor in sub], dim=0)
                    feats = extract_features_batch(rdd, batch, device)
                    for (idx, _), feat in zip(sub, feats):
                        feature_cache[idx] = feat
                    processed_images += len(sub)
                    cursor += len(sub)
                    if attempt_size < len(chunk):
                        chunk = chunk[attempt_size:]
                        attempt_size = len(chunk)
                        continue
                    break
                except RuntimeError as exc:
                    if _is_cuda_oom(exc) and attempt_size > 1:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        attempt_size = max(1, attempt_size // 2)
                        continue
                    raise

                finally:
                    now = time.time()
                    if now >= next_eta_log_time or processed_images == total_images:
                        elapsed = max(1e-6, now - start_time)
                        images_per_sec = processed_images / elapsed
                        remaining_images = max(0, total_images - processed_images)
                        eta_seconds = remaining_images / max(1e-6, images_per_sec) if processed_images > 0 else 0.0
                        print(
                            (
                                "[extract] images "
                                f"{processed_images}/{total_images} "
                                f"({processed_images / max(1, total_images) * 100:.1f}%), "
                                f"images/s={images_per_sec:.2f}, "
                                f"elapsed={elapsed / 60.0:.1f}m, eta={eta_seconds / 60.0:.1f}m"
                            ),
                            flush=True,
                        )
                        next_eta_log_time = now + 60.0

    for idx in tqdm(required_indices, desc="Loading + bucketing images"):
        tensor = load_image_tensor(Path(index_to_path[idx]))
        tensor = resize_tensor_long_side(tensor, resize)
        shape = (int(tensor.shape[-2]), int(tensor.shape[-1]))
        bucket = pending_by_shape[shape]
        if not bucket:
            shape_bucket_count += 1
        bucket.append((idx, tensor))
        if len(bucket) >= rdd_batch_size:
            flush_entries(bucket)
            pending_by_shape[shape] = []

    for entries in pending_by_shape.values():
        flush_entries(entries)

    print(
        f"RDD extraction used {shape_bucket_count} shape buckets with rdd_batch_size={rdd_batch_size}",
        flush=True,
    )

    return feature_cache


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    device_name = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(device_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    use_batching = not args.disable_lg_batching
    rdd_batch_size = max(1, args.rdd_batch_size)
    lg_batch_size = max(1, args.lg_batch_size)

    index_to_path, sample_meta, lookups = build_metadata_index(args.train_data)
    path_to_index = {p: i for i, p in enumerate(index_to_path)}

    anchor_indices = list(range(len(index_to_path)))
    if args.max_anchors > 0:
        rng = random.Random(args.seed)
        anchor_indices = rng.sample(anchor_indices, min(args.max_anchors, len(anchor_indices)))

    print(f"Building pair-quality cache for {len(anchor_indices)} anchors...", flush=True)
    print(
        f"RDD batching: rdd_batch_size={rdd_batch_size} | "
        f"LightGlue batching: enabled={use_batching}, lg_batch_size={lg_batch_size}",
        flush=True,
    )
    rng = random.Random(args.seed)

    candidate_start_time = time.time()
    (
        positive_indices_by_anchor,
        negative_indices_by_anchor,
        total_pairs,
        required_feature_indices,
    ) = build_anchor_candidate_pools(
        anchor_indices,
        sample_meta,
        lookups,
        rng,
        max_positive_candidates_per_anchor=args.max_positive_candidates_per_anchor,
        max_negative_candidates_per_anchor=args.max_negative_candidates_per_anchor,
        sequence_aware_sampling=args.sequence_aware_sampling,
    )
    candidate_elapsed = time.time() - candidate_start_time
    print(
        f"Prepared candidate pools in {candidate_elapsed:.1f}s | "
        f"pairs_to_score={total_pairs} | unique_images={len(required_feature_indices)}",
        flush=True,
    )

    print("Phase 1/2: extracting RDD features for all required images...", flush=True)
    rdd = build_rdd(args.rdd_weights, device, args.top_k)
    feature_extract_start = time.time()
    feature_cache = preextract_feature_cache(
        index_to_path,
        required_feature_indices,
        rdd,
        device,
        args.resize,
        rdd_batch_size,
    )
    feature_extract_elapsed = time.time() - feature_extract_start
    print(
        f"Finished RDD feature extraction in {feature_extract_elapsed / 60.0:.1f}m",
        flush=True,
    )

    del rdd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Phase 2/2: scoring cached features with LightGlue...", flush=True)
    lg = build_masked_lg(device, weights=args.lg_weights)

    positive_candidates_by_anchor: dict[int, list[dict]] = {}
    negative_candidates_by_anchor: dict[int, list[dict]] = {}

    total_anchors = len(anchor_indices)
    processed_pairs = 0
    start_time = time.time()
    next_eta_log_time = start_time + 60.0

    for processed_anchors, anchor_idx in enumerate(
        tqdm(anchor_indices, desc="Scoring anchors"), start=1
    ):
        pos_indices = positive_indices_by_anchor[anchor_idx]
        neg_indices = negative_indices_by_anchor[anchor_idx]
        processed_pairs += len(pos_indices) + len(neg_indices)

        feat_a_dev = move_features_to_device(feature_cache[anchor_idx], device)

        pos_feats = [move_features_to_device(feature_cache[cand_idx], device) for cand_idx in pos_indices]
        pos_metrics = score_candidates_with_fallback(
            lg, feat_a_dev, pos_feats, lg_batch_size, use_batching
        )
        pos_records = _build_candidate_records(pos_indices, sample_meta, True, pos_metrics)
        pos_records = assign_quality_bands(pos_records)
        positive_candidates_by_anchor[anchor_idx] = pos_records

        neg_feats = [move_features_to_device(feature_cache[cand_idx], device) for cand_idx in neg_indices]
        neg_metrics = score_candidates_with_fallback(
            lg, feat_a_dev, neg_feats, lg_batch_size, use_batching
        )
        neg_records = _build_candidate_records(neg_indices, sample_meta, False, neg_metrics)
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
        "lg_batch_size": lg_batch_size,
        "rdd_batch_size": rdd_batch_size,
        "disable_lg_batching": args.disable_lg_batching,
        "num_anchors": len(anchor_indices),
        "num_feature_cache_images": len(required_feature_indices),
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
