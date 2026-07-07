from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class FrameFeat:
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray
    image_size: np.ndarray


@dataclass
class SequenceEntry:
    split: str
    lynx_id: str
    site: str
    sequence_id: str
    frame_paths: list[Path]

    @property
    def name(self) -> str:
        return f"{self.lynx_id}/{self.site}/{self.sequence_id}"


def list_sequences_from_split_dir(split_dir: Path, split_name: str = "train") -> list[SequenceEntry]:
    entries: list[SequenceEntry] = []
    for lynx_dir in sorted(split_dir.iterdir()):
        if not lynx_dir.is_dir():
            continue
        lynx_id = lynx_dir.name
        for site_dir in sorted(lynx_dir.iterdir()):
            if not site_dir.is_dir():
                continue
            site = site_dir.name
            for seq_dir in sorted(site_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                frames = sorted(seq_dir.glob("frame_*.png"))
                if not frames:
                    frames = sorted(seq_dir.glob("frame_*.jpg"))
                if not frames:
                    continue
                entries.append(
                    SequenceEntry(
                        split=split_name,
                        lynx_id=lynx_id,
                        site=site,
                        sequence_id=seq_dir.name,
                        frame_paths=frames,
                    )
                )
    return entries


def sample_frames(frame_paths: list[Path], max_frames: int) -> list[Path]:
    if len(frame_paths) <= max_frames:
        return frame_paths
    idxs = np.linspace(0, len(frame_paths) - 1, num=max_frames, dtype=int)
    return [frame_paths[i] for i in idxs]


def compute_average_precision(relevance: list[bool]) -> float:
    num_rel = sum(relevance)
    if num_rel == 0:
        return 0.0
    hit = 0
    precisions = []
    for idx, is_rel in enumerate(relevance, start=1):
        if is_rel:
            hit += 1
            precisions.append(hit / idx)
    return float(sum(precisions) / num_rel)


def compute_retrieval_metrics(
    query_records: list[dict[str, Any]],
    n_gallery: int,
) -> dict[str, float]:
    correct_top1 = 0
    correct_top5 = 0
    ap_sum = 0.0
    for rec in query_records:
        gt_id = rec["gt_id"]
        scores = rec["scores"]
        top1_id = scores[0]["lynx_id"] if scores else None
        top5_ids = [s["lynx_id"] for s in scores[:5]]
        correct_top1 += int(top1_id == gt_id)
        correct_top5 += int(gt_id in top5_ids)
        relevance = [s["lynx_id"] == gt_id for s in scores]
        ap_sum += compute_average_precision(relevance)
    n_queries = max(1, len(query_records))
    return {
        "top1_acc": correct_top1 / n_queries,
        "top5_acc": correct_top5 / n_queries,
        "mAP": ap_sum / n_queries,
        "n_queries": float(len(query_records)),
        "n_gallery": float(n_gallery),
    }


def compute_balanced_top1_accuracy(query_records: list[dict[str, Any]]) -> float:
    if not query_records:
        return 0.0
    per_id_counts: dict[str, dict[str, int]] = {}
    for rec in query_records:
        gt_id = rec["gt_id"]
        scores = rec["scores"]
        top1_id = scores[0]["lynx_id"] if scores else None
        if gt_id not in per_id_counts:
            per_id_counts[gt_id] = {"correct": 0, "total": 0}
        per_id_counts[gt_id]["total"] += 1
        if top1_id == gt_id:
            per_id_counts[gt_id]["correct"] += 1
    recalls = []
    for stats in per_id_counts.values():
        if stats["total"] > 0:
            recalls.append(stats["correct"] / stats["total"])
    if not recalls:
        return 0.0
    return float(sum(recalls) / len(recalls))


@torch.no_grad()
def score_pair_lightglue(lg, fa: FrameFeat, fb: FrameFeat, device: torch.device) -> tuple[float, int]:
    k0 = torch.from_numpy(fa.keypoints).to(device).unsqueeze(0)
    k1 = torch.from_numpy(fb.keypoints).to(device).unsqueeze(0)
    d0 = torch.from_numpy(fa.descriptors).to(device).unsqueeze(0)
    d1 = torch.from_numpy(fb.descriptors).to(device).unsqueeze(0)
    size0 = torch.tensor(fa.image_size[::-1].copy(), device=device).unsqueeze(0)
    size1 = torch.tensor(fb.image_size[::-1].copy(), device=device).unsqueeze(0)

    pred = lg(
        {
            "image0": {"keypoints": k0, "descriptors": d0, "image_size": size0},
            "image1": {"keypoints": k1, "descriptors": d1, "image_size": size1},
        }
    )
    if pred["scores"][0].numel() == 0:
        return 0.0, 0
    conf = pred["scores"][0]
    sum_conf = conf.sum().item()
    norm = min(max(1, fa.keypoints.shape[0]), max(1, fb.keypoints.shape[0]))
    score = sum_conf / norm
    return score, int((conf > 0).sum().item())


def aggregate_sequence_score(
    lg,
    q_frames: list[FrameFeat],
    g_frames: list[FrameFeat],
    device: torch.device,
    top_m: int,
) -> dict[str, float]:
    per_q_best: list[float] = []
    for qf in q_frames:
        best_score = 0.0
        for gf in g_frames:
            score, _ = score_pair_lightglue(lg, qf, gf, device)
            if score > best_score:
                best_score = score
        per_q_best.append(best_score)
    if not per_q_best:
        return {"score": 0.0}
    top_vals = sorted(per_q_best, reverse=True)[:top_m]
    seq_score = float(sum(top_vals) / len(top_vals))
    return {"score": seq_score}


def build_probe_subsets(
    train_data: Path,
    val_data: Path,
    num_queries: int,
    gallery_per_id: int,
    seed: int,
) -> tuple[list[SequenceEntry], list[SequenceEntry]]:
    rng = random.Random(seed)
    gallery_all = list_sequences_from_split_dir(train_data, "train")
    query_all = list_sequences_from_split_dir(val_data, "test")

    by_id: dict[str, list[SequenceEntry]] = {}
    for seq in gallery_all:
        by_id.setdefault(seq.lynx_id, []).append(seq)

    gallery: list[SequenceEntry] = []
    for seqs in by_id.values():
        rng.shuffle(seqs)
        gallery.extend(seqs[: max(1, gallery_per_id)])

    rng.shuffle(query_all)
    queries = query_all[: min(num_queries, len(query_all))]
    return gallery, queries


@torch.no_grad()
def extract_frame_feat(rdd, img: torch.Tensor, device: torch.device, top_k: int) -> FrameFeat:
    raw = rdd.module if hasattr(rdd, "module") else rdd
    raw.top_k = top_k
    raw.set_softdetect(top_k=top_k)
    img = img.to(device)
    out = raw.extract(img)[0]
    return FrameFeat(
        keypoints=out["keypoints"].cpu().numpy(),
        descriptors=out["descriptors"].cpu().numpy(),
        scores=out["scores"].cpu().numpy(),
        image_size=np.array(img.shape[-2:], dtype=np.int32),
    )


def load_image_tensor(path: Path, resize: int) -> torch.Tensor:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if resize > 0:
        w, h = img.size
        scale = resize / max(w, h)
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def run_retrieval_probe(
    rdd,
    lg,
    train_data: Path,
    val_data: Path,
    device: torch.device,
    *,
    num_queries: int,
    gallery_per_id: int,
    frames_per_seq: int,
    top_m_pool: int,
    resize: int,
    top_k: int,
    seed: int,
) -> dict[str, float]:
    gallery, queries = build_probe_subsets(
        train_data, val_data, num_queries, gallery_per_id, seed
    )
    if not gallery or not queries:
        return {
            "retrieval_probe/top1_acc": 0.0,
            "retrieval_probe/top5_acc": 0.0,
            "retrieval_probe/mAP": 0.0,
            "retrieval_probe/balanced_top1_acc": 0.0,
        }

    seq_feat_cache: dict[str, list[FrameFeat]] = {}

    def load_seq_feats(seq: SequenceEntry) -> list[FrameFeat]:
        cached = seq_feat_cache.get(seq.name)
        if cached is not None:
            return cached
        frames = sample_frames(seq.frame_paths, frames_per_seq)
        feats = []
        for fp in frames:
            img = load_image_tensor(fp, resize)
            feats.append(extract_frame_feat(rdd, img, device, top_k))
        seq_feat_cache[seq.name] = feats
        return feats

    query_records = []
    for q in queries:
        q_feats = load_seq_feats(q)
        scores = []
        for g in gallery:
            g_feats = load_seq_feats(g)
            agg = aggregate_sequence_score(lg, q_feats, g_feats, device, top_m_pool)
            scores.append({"score": agg["score"], "lynx_id": g.lynx_id, "name": g.name})
        scores.sort(key=lambda x: x["score"], reverse=True)
        query_records.append({"query": q.name, "gt_id": q.lynx_id, "scores": scores})

    metrics = compute_retrieval_metrics(query_records, n_gallery=len(gallery))
    balanced = compute_balanced_top1_accuracy(query_records)
    return {
        "retrieval_probe/top1_acc": metrics["top1_acc"],
        "retrieval_probe/top5_acc": metrics["top5_acc"],
        "retrieval_probe/mAP": metrics["mAP"],
        "retrieval_probe/balanced_top1_acc": balanced,
    }
