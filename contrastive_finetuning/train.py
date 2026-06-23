from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

from rdd.RDD.utils import to_pixel_coords
from contrastive_finetuning.loading import (
    BalancedBatchSampler,
    FixedTripletDataset,
    LabeledImageFolder,
    TripletImageFolder,
    get_loader,
)
from contrastive_finetuning.models import build_masked_lg, build_rdd
from contrastive_finetuning.pair_quality import PairMiningConfig, PairQualityCache, validate_cache_metadata
from contrastive_finetuning.process import align_tensors_to_max_length
from contrastive_finetuning.retrieval_probe import run_retrieval_probe


@dataclass
class LossBreakdown:
    total: torch.Tensor
    main: torch.Tensor
    coverage: torch.Tensor
    proxy: torch.Tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contrastive fine-tuning of RDD descriptor")
    p.add_argument("--train_data", type=Path, required=True)
    p.add_argument("--val_data", type=Path, required=True)
    p.add_argument("--rdd_weights", type=str, default="rdd/weights/RDD-v2.pth")
    p.add_argument("--lg_weights", type=str, default="rdd/weights/RDD_lg-v2.pth")
    p.add_argument("--output_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--project", type=str, default=None, help="wandb project name")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=0.5, help="Backward-compatible triplet margin")
    p.add_argument("--loss_margin", type=float, default=None)
    p.add_argument("--resize", type=int, default=512)
    p.add_argument("--top_k", type=int, default=512)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--batch_mode", choices=["triplet", "balanced"], default="triplet")
    p.add_argument("--n_classes", type=int, default=4)
    p.add_argument("--n_samples_per_class", type=int, default=2)
    p.add_argument("--loss_type", choices=["triplet", "batch_hard_topk"], default="triplet")
    p.add_argument("--topk_pos", type=int, default=32)
    p.add_argument("--topk_neg", type=int, default=32)
    p.add_argument("--coverage_weight", type=float, default=0.0)
    p.add_argument("--positive_sim_threshold", type=float, default=0.3)
    p.add_argument("--hard_negatives_per_anchor", type=int, default=1)
    p.add_argument("--use_match_proxy", action="store_true")
    p.add_argument("--match_proxy_weight", type=float, default=0.1)
    p.add_argument("--match_proxy_pairs_per_batch", type=int, default=2)
    p.add_argument("--aug_profile", choices=["none", "local_corr_v1"], default="none")
    p.add_argument("--use_center_bias_crop", action="store_true")
    p.add_argument("--save_eval_visuals", action="store_true")
    p.add_argument("--num_eval_visuals", type=int, default=4)
    p.add_argument("--eval_metrics_profile", choices=["basic", "extended"], default="basic")
    p.add_argument("--sequence_aware_sampling", type=lambda x: str(x).lower() in {"1", "true", "yes", "y", "on"}, default=True)

    p.add_argument("--use_pair_quality_mining", action="store_true")
    p.add_argument("--pair_quality_cache_dir", type=Path, default=None)
    p.add_argument("--positive_quality_mode", choices=["random", "ranked", "bucketed"], default="bucketed")
    p.add_argument("--positive_high_ratio", type=float, default=0.7)
    p.add_argument("--positive_medium_ratio", type=float, default=0.3)
    p.add_argument("--exclude_low_quality_positives", action="store_true", help="Exclude low-quality positives (default when mining enabled).")
    p.add_argument("--include_low_quality_positives", action="store_true", help="Allow low-quality positives during mining.")
    p.add_argument("--use_hard_negative_cache", action="store_true", help="Mix in cached hard negatives (default when mining enabled).")
    p.add_argument("--no_hard_negative_cache", action="store_true", help="Disable cached hard negatives.")
    p.add_argument("--hard_negative_ratio", type=float, default=0.5)
    p.add_argument("--max_positive_candidates_per_anchor", type=int, default=32)
    p.add_argument("--max_negative_candidates_per_anchor", type=int, default=32)

    p.add_argument("--use_retrieval_probe", action="store_true")
    p.add_argument("--retrieval_probe_num_queries", type=int, default=8)
    p.add_argument("--retrieval_probe_gallery_per_id", type=int, default=2)
    p.add_argument("--retrieval_probe_frames_per_seq", type=int, default=2)
    p.add_argument("--retrieval_probe_every_n_epochs", type=int, default=1)
    p.add_argument("--retrieval_probe_top_m_pool", type=int, default=3)
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


class CenterBiasedCrop:
    def __init__(self, scale_range: tuple[float, float] = (0.8, 1.0), jitter: float = 0.08) -> None:
        self.scale_range = scale_range
        self.jitter = jitter

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        crop_scale = random.uniform(*self.scale_range)
        crop_w = max(1, int(width * crop_scale))
        crop_h = max(1, int(height * crop_scale))

        jitter_x = int((width - crop_w) * self.jitter)
        jitter_y = int((height - crop_h) * self.jitter)
        cx = width // 2 + random.randint(-jitter_x, jitter_x) if jitter_x > 0 else width // 2
        cy = height // 2 + random.randint(-jitter_y, jitter_y) if jitter_y > 0 else height // 2

        left = min(max(cx - crop_w // 2, 0), max(width - crop_w, 0))
        top = min(max(cy - crop_h // 2, 0), max(height - crop_h, 0))
        return TF.crop(img, top, left, crop_h, crop_w)


class AddGaussianNoise:
    def __init__(self, std: float = 0.01) -> None:
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return (tensor + noise).clamp(0.0, 1.0)


class IdentityTransform:
    def __call__(self, img):
        return img


def build_train_transform(args: argparse.Namespace) -> transforms.Compose:
    ops: list = []
    if args.aug_profile == "local_corr_v1":
        if args.use_center_bias_crop:
            ops.append(CenterBiasedCrop(scale_range=(0.85, 1.0), jitter=0.08))
            ops.append(transforms.Resize((args.resize, args.resize), antialias=True))
        else:
            ops.append(
                transforms.RandomResizedCrop(
                    size=(args.resize, args.resize),
                    scale=(0.75, 1.0),
                    ratio=(0.9, 1.1),
                    antialias=True,
                )
            )
        ops.extend(
            [
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.03),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.2))], p=0.2),
                transforms.ToTensor(),
                AddGaussianNoise(std=0.01),
            ]
        )
    else:
        if args.use_center_bias_crop:
            ops.extend([CenterBiasedCrop(scale_range=(0.9, 1.0), jitter=0.04), transforms.Resize((args.resize, args.resize), antialias=True)])
        ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor()])


def get_loss_margin(args: argparse.Namespace) -> float:
    return args.margin if args.loss_margin is None else args.loss_margin


def resize_long_side(images: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = images.shape
    scale = size / max(h, w)
    new_h = max(32, int(h * scale) // 32 * 32)
    new_w = max(32, int(w * scale) // 32 * 32)
    return F.interpolate(images.float(), (new_h, new_w), mode="bilinear", align_corners=False)


def batch_features(feats: list[dict], image_h: int, image_w: int) -> dict:
    ks = [f["keypoints"] for f in feats]
    ds = [f["descriptors"] for f in feats]
    device = ks[0].device

    ks_pad, masks = align_tensors_to_max_length(ks)
    ds_pad, _ = align_tensors_to_max_length(ds)
    sizes = torch.tensor([image_w, image_h], device=device).unsqueeze(0).expand(len(feats), -1).contiguous()
    return {
        "keypoints": ks_pad,
        "descriptors": ds_pad,
        "image_size": sizes,
        "masks": masks.unsqueeze(1),
    }


def extract_train(rdd: torch.nn.Module, images: torch.Tensor) -> list[dict]:
    raw = _unwrap(rdd)
    b = images.shape[0]
    images_prep, rh, rw = raw.preprocess_tensor(images)
    _, _, h_p, w_p = images_prep.shape

    m1, k1, _ = rdd(images_prep)
    m1 = F.normalize(m1, dim=1)

    with torch.no_grad():
        kpts, kscores, _ = raw.softdetect(k1)
        kpts = torch.vstack([kpts[idx].unsqueeze(0) for idx in range(b)])
        kscores = torch.vstack([kscores[idx].unsqueeze(0) for idx in range(b)])
        kpts_px = to_pixel_coords(kpts, h_p, w_p)
        kpts_scaled = kpts_px * torch.tensor([rw, rh], device=images.device).view(1, -1)
        valid = kscores > raw.detection_threshold

    descs = raw.interpolator(m1, kpts_px, H=h_p, W=w_p)
    descs = F.normalize(descs, dim=-1)
    return [
        {
            "keypoints": kpts_scaled[idx][valid[idx]].detach(),
            "descriptors": descs[idx][valid[idx]],
        }
        for idx in range(b)
    ]


def descriptor_triplet_loss(
    feats_a: list[dict],
    feats_p: list[dict],
    feats_n: list[dict],
    margin: float,
) -> torch.Tensor:
    losses = []
    for fa, fp, fn in zip(feats_a, feats_p, feats_n):
        da, dp, dn = fa["descriptors"], fp["descriptors"], fn["descriptors"]
        if min(da.shape[0], dp.shape[0], dn.shape[0]) == 0:
            continue
        best_pos = (da @ dp.T).max(dim=1).values
        best_neg = (da @ dn.T).max(dim=1).values
        losses.append(F.relu(margin - best_pos + best_neg).mean())
    if not losses:
        device = feats_a[0]["descriptors"].device
        return torch.zeros(1, device=device, requires_grad=True).squeeze()
    return torch.stack(losses).mean()


def pairwise_best_scores(desc0: torch.Tensor, desc1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if min(desc0.shape[0], desc1.shape[0]) == 0:
        empty = desc0.new_zeros((0,))
        return empty, empty
    sim = desc0 @ desc1.T
    return sim.max(dim=1).values, sim.max(dim=0).values


def topk_mean(values: torch.Tensor, k: int) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    k = max(1, min(k, values.numel()))
    return values.topk(k).values.mean()


def pair_confidence_proxy(desc0: torch.Tensor, desc1: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if min(desc0.shape[0], desc1.shape[0]) == 0:
        return desc0.new_tensor(0.0)
    sim = desc0 @ desc1.T
    row_conf = F.softmax(sim / temperature, dim=1).max(dim=1).values
    col_conf = F.softmax(sim.T / temperature, dim=1).max(dim=1).values
    return 0.5 * (row_conf.mean() + col_conf.mean())


def mutual_nn_rate(desc0: torch.Tensor, desc1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if min(desc0.shape[0], desc1.shape[0]) == 0:
        zero = desc0.new_tensor(0.0)
        return zero, zero
    sim = desc0 @ desc1.T
    row_best = sim.argmax(dim=1)
    col_best = sim.argmax(dim=0)
    row_ids = torch.arange(sim.shape[0], device=sim.device)
    mutual = row_ids == col_best[row_best]
    rate = mutual.float().mean()
    if mutual.any():
        score = sim[row_ids[mutual], row_best[mutual]].mean()
    else:
        score = sim.new_tensor(0.0)
    return rate, score


def positive_pairs_from_labels(labels: torch.Tensor) -> list[tuple[int, int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        groups[int(label)].append(idx)
    pairs: list[tuple[int, int]] = []
    for indices in groups.values():
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                pairs.append((indices[left], indices[right]))
    return pairs


def directional_rank_loss(
    anchor_feat: dict,
    positive_feat: dict,
    negative_feats: Iterable[dict],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    desc_anchor = anchor_feat["descriptors"]
    desc_pos = positive_feat["descriptors"]
    if min(desc_anchor.shape[0], desc_pos.shape[0]) == 0:
        zero = desc_anchor.new_tensor(0.0)
        return zero, zero, zero, []

    pos_best, _ = pairwise_best_scores(desc_anchor, desc_pos)
    pos_score = topk_mean(pos_best, args.topk_pos)
    coverage = (pos_best > args.positive_sim_threshold).float().mean()

    neg_scores = []
    neg_order = []
    for neg_idx, neg_feat in enumerate(negative_feats):
        desc_neg = neg_feat["descriptors"]
        if desc_neg.shape[0] == 0:
            continue
        neg_best, _ = pairwise_best_scores(desc_anchor, desc_neg)
        neg_scores.append(topk_mean(neg_best, args.topk_neg))
        neg_order.append(neg_idx)

    if neg_scores:
        neg_scores_tensor = torch.stack(neg_scores)
        hard_k = max(1, min(args.hard_negatives_per_anchor, neg_scores_tensor.numel()))
        hard_values, hard_indices = neg_scores_tensor.topk(hard_k)
        neg_score = hard_values.mean()
        selected = [neg_order[idx] for idx in hard_indices.tolist()]
    else:
        neg_score = desc_anchor.new_tensor(0.0)
        selected = []

    main = F.relu(get_loss_margin(args) - pos_score + neg_score)
    coverage_loss = 1.0 - coverage
    total = main + args.coverage_weight * coverage_loss
    return total, main, coverage_loss, selected


def descriptor_triplet_loss_from_labeled_batch(
    feats: list[dict], labels: torch.Tensor, margin: float
) -> torch.Tensor:
    losses = []
    labels_list = labels.tolist()
    for anchor_idx, anchor_feat in enumerate(feats):
        pos_candidates = [idx for idx, label in enumerate(labels_list) if label == labels_list[anchor_idx] and idx != anchor_idx]
        neg_candidates = [idx for idx, label in enumerate(labels_list) if label != labels_list[anchor_idx]]
        if not pos_candidates or not neg_candidates:
            continue
        pos_idx = pos_candidates[0]
        desc_anchor = anchor_feat["descriptors"]
        desc_pos = feats[pos_idx]["descriptors"]
        if min(desc_anchor.shape[0], desc_pos.shape[0]) == 0:
            continue
        best_pos = (desc_anchor @ desc_pos.T).max(dim=1).values

        neg_image_scores = []
        for neg_idx in neg_candidates:
            desc_neg = feats[neg_idx]["descriptors"]
            if desc_neg.shape[0] == 0:
                continue
            best_neg = (desc_anchor @ desc_neg.T).max(dim=1).values
            neg_image_scores.append(best_neg.mean())
        if not neg_image_scores:
            continue
        hardest_neg = torch.stack(neg_image_scores).max()
        losses.append(F.relu(margin - best_pos.mean() + hardest_neg))

    if not losses:
        device = feats[0]["descriptors"].device
        return torch.zeros(1, device=device, requires_grad=True).squeeze()
    return torch.stack(losses).mean()


def batch_hard_topk_loss_balanced(
    feats: list[dict], labels: torch.Tensor, args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], list[tuple[int, int]]]:
    pos_pairs = positive_pairs_from_labels(labels)
    losses = []
    coverage_losses = []
    hard_negative_pairs: list[tuple[int, int]] = []
    labels_list = labels.tolist()

    for left, right in pos_pairs:
        negatives_left = [feats[idx] for idx, label in enumerate(labels_list) if label != labels_list[left]]
        total_lr, _, coverage_lr, selected_left = directional_rank_loss(feats[left], feats[right], negatives_left, args)
        losses.append(total_lr)
        coverage_losses.append(coverage_lr)
        neg_left_indices = [idx for idx, label in enumerate(labels_list) if label != labels_list[left]]
        for sel in selected_left:
            hard_negative_pairs.append((left, neg_left_indices[sel]))

        negatives_right = [feats[idx] for idx, label in enumerate(labels_list) if label != labels_list[right]]
        total_rl, _, coverage_rl, selected_right = directional_rank_loss(feats[right], feats[left], negatives_right, args)
        losses.append(total_rl)
        coverage_losses.append(coverage_rl)
        neg_right_indices = [idx for idx, label in enumerate(labels_list) if label != labels_list[right]]
        for sel in selected_right:
            hard_negative_pairs.append((right, neg_right_indices[sel]))

    if not losses:
        device = feats[0]["descriptors"].device
        zero = torch.zeros(1, device=device, requires_grad=True).squeeze()
        return zero, zero, [], []
    return torch.stack(losses).mean(), torch.stack(coverage_losses).mean(), pos_pairs, hard_negative_pairs


def batch_hard_topk_loss_triplet(
    feats_a: list[dict], feats_p: list[dict], feats_n: list[dict], args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], list[tuple[int, int]]]:
    losses = []
    coverage_losses = []
    positive_pairs = []
    negative_pairs = []

    for idx, (feat_a, feat_p) in enumerate(zip(feats_a, feats_p)):
        total_ap, _, coverage_ap, _ = directional_rank_loss(feat_a, feat_p, feats_n, args)
        total_pa, _, coverage_pa, _ = directional_rank_loss(feat_p, feat_a, feats_n, args)
        losses.extend([total_ap, total_pa])
        coverage_losses.extend([coverage_ap, coverage_pa])
        positive_pairs.append((idx, idx))
        negative_pairs.append((idx, idx))

    if not losses:
        device = feats_a[0]["descriptors"].device
        zero = torch.zeros(1, device=device, requires_grad=True).squeeze()
        return zero, zero, [], []
    return torch.stack(losses).mean(), torch.stack(coverage_losses).mean(), positive_pairs, negative_pairs


def select_proxy_pairs(
    positive_pairs: list[tuple[int, int]],
    negative_pairs: list[tuple[int, int]],
    max_pairs: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if max_pairs <= 0:
        return [], []
    pos_pairs = positive_pairs[:max_pairs]
    neg_pairs = negative_pairs[:max_pairs]
    return pos_pairs, neg_pairs


def lightglue_target_rate(
    lg: torch.nn.Module,
    feat0: dict,
    feat1: dict,
    image_h: int,
    image_w: int,
) -> torch.Tensor:
    with torch.no_grad():
        data0 = batch_features([feat0], image_h, image_w)
        data1 = batch_features([feat1], image_h, image_w)
        pred = lg({"image0": data0, "image1": data1})
        matches = pred["matches"][0]
        denom = max(int(feat0["keypoints"].shape[0]), 1)
        match_rate = matches.shape[0] / denom
        score_mean = pred["scores"][0].mean().item() if pred["scores"][0].numel() > 0 else 0.0
        return feat0["descriptors"].new_tensor(0.5 * match_rate + 0.5 * score_mean)


def compute_match_proxy_loss(
    lg: torch.nn.Module,
    feats_left: list[dict],
    feats_right: list[dict],
    positive_pairs: list[tuple[int, int]],
    negative_pairs: list[tuple[int, int]],
    image_h: int,
    image_w: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    if not args.use_match_proxy:
        return feats_left[0]["descriptors"].new_tensor(0.0)

    pos_pairs, neg_pairs = select_proxy_pairs(positive_pairs, negative_pairs, args.match_proxy_pairs_per_batch)
    losses = []
    for left_idx, right_idx in pos_pairs:
        desc_left = feats_left[left_idx]["descriptors"]
        desc_right = feats_right[right_idx]["descriptors"]
        if min(desc_left.shape[0], desc_right.shape[0]) == 0:
            continue
        target = lightglue_target_rate(lg, feats_left[left_idx], feats_right[right_idx], image_h, image_w)
        mutual_rate, _ = mutual_nn_rate(desc_left.detach(), desc_right.detach())
        proxy = pair_confidence_proxy(desc_left, desc_right)
        target = 0.5 * target + 0.5 * mutual_rate.detach()
        losses.append(F.mse_loss(proxy, target))

    for left_idx, right_idx in neg_pairs:
        desc_left = feats_left[left_idx]["descriptors"]
        desc_right = feats_right[right_idx]["descriptors"]
        if min(desc_left.shape[0], desc_right.shape[0]) == 0:
            continue
        target = lightglue_target_rate(lg, feats_left[left_idx], feats_right[right_idx], image_h, image_w)
        mutual_rate, _ = mutual_nn_rate(desc_left.detach(), desc_right.detach())
        proxy = pair_confidence_proxy(desc_left, desc_right)
        target = 0.5 * target + 0.5 * mutual_rate.detach()
        losses.append(F.mse_loss(proxy, target))

    if not losses:
        return feats_left[0]["descriptors"].new_tensor(0.0)
    return torch.stack(losses).mean()


def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    rgb = (tensor.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_matches(
    pts0: np.ndarray,
    pts1: np.ndarray,
    conf: np.ndarray,
    img0: np.ndarray,
    img1: np.ndarray,
    title: str,
    max_matches: int = 20,
) -> np.ndarray:
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.zeros((max(h0, h1) + 28, w0 + w1, 3), dtype=np.uint8)
    canvas[28 : 28 + h0, :w0] = img0
    canvas[28 : 28 + h1, w0:] = img1

    if len(pts0) > 0:
        idx = np.argsort(-conf)[:max_matches]
        pts0 = pts0[idx]
        pts1 = pts1[idx]
        conf = conf[idx]
        palette = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        for match_idx, ((x0, y0), (x1, y1)) in enumerate(zip(pts0, pts1)):
            color = palette[match_idx % len(palette)]
            pt0 = (int(x0), int(y0) + 28)
            pt1 = (int(x1) + w0, int(y1) + 28)
            cv2.circle(canvas, pt0, 3, color, -1)
            cv2.circle(canvas, pt1, 3, color, -1)
            cv2.line(canvas, pt0, pt1, color, 1)

    cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def unpack_matches(pred: dict, index: int, keypoints0: np.ndarray, keypoints1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matches = pred["matches"][index].detach().cpu().numpy()
    scores = pred["scores"][index].detach().cpu().numpy()
    if matches.size == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))
    valid = (matches[:, 0] < len(keypoints0)) & (matches[:, 1] < len(keypoints1))
    matches = matches[valid]
    scores = scores[valid]
    if matches.size == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))
    return keypoints0[matches[:, 0]], keypoints1[matches[:, 1]], scores


def maybe_save_eval_visuals(
    save_dir: Path,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    feats_a: list[dict],
    feats_p: list[dict],
    feats_n: list[dict],
    pred_pos: dict,
    pred_neg: dict,
    scale: np.ndarray,
    saved_so_far: int,
    max_visuals: int,
    prefix: str,
) -> int:
    if cv2 is None or saved_so_far >= max_visuals:
        return saved_so_far
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(min(len(feats_a), max_visuals - saved_so_far)):
        anchor_bgr = tensor_to_bgr(anchors[idx])
        pos_bgr = tensor_to_bgr(positives[idx])
        neg_bgr = tensor_to_bgr(negatives[idx])
        key_a = feats_a[idx]["keypoints"].detach().cpu().numpy() * scale
        key_p = feats_p[idx]["keypoints"].detach().cpu().numpy() * scale
        key_n = feats_n[idx]["keypoints"].detach().cpu().numpy() * scale
        pts0_p, pts1_p, scores_p = unpack_matches(pred_pos, idx, key_a, key_p)
        pts0_n, pts1_n, scores_n = unpack_matches(pred_neg, idx, key_a, key_n)

        pos_title = f"{prefix} pos | matches={len(scores_p)} | conf={scores_p.mean():.3f}" if len(scores_p) else f"{prefix} pos | matches=0 | conf=0.000"
        neg_title = f"{prefix} neg | matches={len(scores_n)} | conf={scores_n.mean():.3f}" if len(scores_n) else f"{prefix} neg | matches=0 | conf=0.000"
        cv2.imwrite(str(save_dir / f"pair_{saved_so_far:03d}_pos.jpg"), draw_matches(pts0_p, pts1_p, scores_p, anchor_bgr, pos_bgr, pos_title))
        cv2.imwrite(str(save_dir / f"pair_{saved_so_far:03d}_neg.jpg"), draw_matches(pts0_n, pts1_n, scores_n, anchor_bgr, neg_bgr, neg_title))
        saved_so_far += 1
        if saved_so_far >= max_visuals:
            break
    return saved_so_far


def compute_loss_breakdown(
    args: argparse.Namespace,
    lg: torch.nn.Module,
    batch,
    rdd: torch.nn.Module,
    device: torch.device,
) -> LossBreakdown:
    margin = get_loss_margin(args)

    if args.batch_mode == "balanced":
        images, labels = batch
        labels = labels.to(device)
        images_r = resize_long_side(images, args.resize).to(device)
        feats = extract_train(rdd, images_r)

        if args.loss_type == "triplet":
            main_loss = descriptor_triplet_loss_from_labeled_batch(feats, labels, margin)
            coverage_loss = main_loss.new_tensor(0.0)
            positive_pairs = positive_pairs_from_labels(labels)
            negative_pairs = []
            labels_list = labels.tolist()
            for left, _ in positive_pairs:
                neg_indices = [idx for idx, label in enumerate(labels_list) if label != labels_list[left]]
                if neg_indices:
                    negative_pairs.append((left, neg_indices[0]))
        else:
            main_loss, coverage_loss, positive_pairs, negative_pairs = batch_hard_topk_loss_balanced(feats, labels, args)

        proxy_loss = compute_match_proxy_loss(
            lg,
            feats,
            feats,
            positive_pairs,
            negative_pairs,
            int(images_r.shape[-2]),
            int(images_r.shape[-1]),
            args,
        )
        total_loss = main_loss + args.match_proxy_weight * proxy_loss
        return LossBreakdown(total=total_loss, main=main_loss, coverage=coverage_loss, proxy=proxy_loss)

    anchors, positives, negatives = batch
    anchors_r = resize_long_side(anchors, args.resize).to(device)
    positives_r = resize_long_side(positives, args.resize).to(device)
    negatives_r = resize_long_side(negatives, args.resize).to(device)
    feats_a = extract_train(rdd, anchors_r)
    feats_p = extract_train(rdd, positives_r)
    feats_n = extract_train(rdd, negatives_r)

    if args.loss_type == "triplet":
        main_loss = descriptor_triplet_loss(feats_a, feats_p, feats_n, margin)
        coverage_loss = main_loss.new_tensor(0.0)
        positive_pairs = [(idx, idx) for idx in range(len(feats_a))]
        negative_pairs = [(idx, idx) for idx in range(len(feats_a))]
    else:
        main_loss, coverage_loss, positive_pairs, negative_pairs = batch_hard_topk_loss_triplet(feats_a, feats_p, feats_n, args)

    proxy_loss = compute_match_proxy_loss(
        lg,
        feats_a,
        feats_p,
        positive_pairs,
        [],
        int(anchors_r.shape[-2]),
        int(anchors_r.shape[-1]),
        args,
    )
    if negative_pairs:
        proxy_loss = 0.5 * (
            proxy_loss
            + compute_match_proxy_loss(
                lg,
                feats_a,
                feats_n,
                [],
                negative_pairs,
                int(anchors_r.shape[-2]),
                int(anchors_r.shape[-1]),
                args,
            )
        )
    total_loss = main_loss + args.match_proxy_weight * proxy_loss
    return LossBreakdown(total=total_loss, main=main_loss, coverage=coverage_loss, proxy=proxy_loss)


def train_epoch(
    accelerator: Accelerator,
    rdd: torch.nn.Module,
    lg: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    mini_train_loader,
    mini_val_loader,
    epoch: int,
    total_epochs: int,
    args: argparse.Namespace,
    global_step: int,
) -> tuple[dict[str, float], int]:
    rdd.train()
    epoch_total = 0.0
    epoch_main = 0.0
    epoch_proxy = 0.0
    epoch_coverage = 0.0
    steps_per_epoch = len(loader)

    if accelerator.is_main_process:
        print(
            f"[epoch {epoch:02d}] starting training loop | steps={steps_per_epoch} | "
            f"batch_mode={args.batch_mode} | loss_type={args.loss_type}",
            flush=True,
        )

    for step, batch in enumerate(loader):
        step_start = time.perf_counter()
        if accelerator.is_main_process and step == 0:
            print(f"[epoch {epoch:02d}] first batch fetched, starting forward/loss", flush=True)

        breakdown = compute_loss_breakdown(args, lg, batch, rdd, accelerator.device)
        forward_time = time.perf_counter() - step_start

        optimizer.zero_grad()
        backward_start = time.perf_counter()
        accelerator.backward(breakdown.total)
        accelerator.clip_grad_norm_((p for p in _unwrap(rdd).parameters() if p.requires_grad), args.grad_clip)
        optimizer.step()
        backward_time = time.perf_counter() - backward_start
        step_time = time.perf_counter() - step_start

        epoch_total += breakdown.total.item()
        epoch_main += breakdown.main.item()
        epoch_proxy += breakdown.proxy.item()
        epoch_coverage += breakdown.coverage.item()
        global_step += 1
        progress = (epoch * steps_per_epoch + step + 1) / max(total_epochs * steps_per_epoch, 1)

        if accelerator.is_main_process:
            accelerator.log(
                {
                    "train/loss": breakdown.total.item(),
                    "train/main_loss": breakdown.main.item(),
                    "train/match_proxy_loss": breakdown.proxy.item(),
                    "train/coverage_loss": breakdown.coverage.item(),
                    "progress": progress,
                },
                step=global_step,
            )
            if step == 0:
                print(
                    f"  [epoch {epoch:02d} | step {step:04d}] first step done | "
                    f"forward+loss={forward_time:.2f}s backward+opt={backward_time:.2f}s total={step_time:.2f}s",
                    flush=True,
                )
            if step % 20 == 0:
                print(
                    f"  [epoch {epoch:02d} | step {step:04d} | progress={progress:.3f}] "
                    f"loss={breakdown.total.item():.4f} main={breakdown.main.item():.4f} proxy={breakdown.proxy.item():.4f} "
                    f"step_time={step_time:.2f}s"
                )

        if (step + 1) % 100 == 0:
            mini_train_m = eval_epoch(accelerator, rdd, lg, mini_train_loader, args, prefix="mini_train", epoch=epoch)
            mini_val_m = eval_epoch(accelerator, rdd, lg, mini_val_loader, args, prefix="mini_val", epoch=epoch)
            rdd.train()
            if accelerator.is_main_process:
                accelerator.log({**mini_train_m, **mini_val_m, "progress": progress}, step=global_step)
                print(
                    f"  [mini-eval @ progress={progress:.3f}] train ratio={mini_train_m['mini_train/match_ratio']:.3f} "
                    f"val ratio={mini_val_m['mini_val/match_ratio']:.3f}"
                )

    denom = max(steps_per_epoch, 1)
    return {
        "train/epoch_loss": epoch_total / denom,
        "train/epoch_main_loss": epoch_main / denom,
        "train/epoch_match_proxy_loss": epoch_proxy / denom,
        "train/epoch_coverage_loss": epoch_coverage / denom,
    }, global_step


@torch.no_grad()
def eval_epoch(
    accelerator: Accelerator,
    rdd: torch.nn.Module,
    lg: torch.nn.Module,
    loader,
    args: argparse.Namespace,
    prefix: str,
    epoch: int,
) -> dict:
    device = accelerator.device
    _unwrap(rdd).eval()
    total_pos = torch.zeros(1, device=device)
    total_neg = torch.zeros(1, device=device)
    total_pos_conf = torch.zeros(1, device=device)
    total_neg_conf = torch.zeros(1, device=device)
    total_participation = torch.zeros(1, device=device)
    total_mutual_pos = torch.zeros(1, device=device)
    total_mutual_neg = torch.zeros(1, device=device)
    total_hard_neg = torch.zeros(1, device=device)
    batch_count = torch.zeros(1, device=device)
    n = torch.zeros(1, device=device)
    saved_visuals = 0
    visuals_dir = args.output_dir / f"{prefix}_visuals" / f"epoch_{epoch:02d}"

    for anchors, positives, negatives in loader:
        anchors_r = resize_long_side(anchors, args.resize).to(device)
        positives_r = resize_long_side(positives, args.resize).to(device)
        negatives_r = resize_long_side(negatives, args.resize).to(device)
        h_r, w_r = anchors_r.shape[-2:]

        feats_a = extract_train(_unwrap(rdd), anchors_r)
        feats_p = extract_train(_unwrap(rdd), positives_r)
        feats_n = extract_train(_unwrap(rdd), negatives_r)

        data_a = batch_features(feats_a, h_r, w_r)
        data_p = batch_features(feats_p, h_r, w_r)
        data_n = batch_features(feats_n, h_r, w_r)

        pred_pos = lg({"image0": data_a, "image1": data_p})
        pred_neg = lg({"image0": data_a, "image1": data_n})

        pos_counts = []
        neg_counts = []
        pos_confs = []
        neg_confs = []
        participation = []
        mutual_pos = []
        mutual_neg = []
        hard_neg_this_batch = 0.0

        for idx, (feat_a, feat_p, feat_n) in enumerate(zip(feats_a, feats_p, feats_n)):
            pos_count = float(len(pred_pos["matches"][idx]))
            neg_count = float(len(pred_neg["matches"][idx]))
            pos_counts.append(pos_count)
            neg_counts.append(neg_count)
            hard_neg_this_batch = max(hard_neg_this_batch, neg_count)

            pos_scores = pred_pos["scores"][idx]
            neg_scores = pred_neg["scores"][idx]
            pos_confs.append(float(pos_scores.mean().item()) if pos_scores.numel() > 0 else 0.0)
            neg_confs.append(float(neg_scores.mean().item()) if neg_scores.numel() > 0 else 0.0)

            match_idx = pred_pos["matches"][idx]
            if match_idx.numel() > 0:
                matched_anchor = torch.unique(match_idx[:, 0]).numel()
                participation.append(matched_anchor / max(int(feat_a["keypoints"].shape[0]), 1))
            else:
                participation.append(0.0)

            pos_mutual, _ = mutual_nn_rate(feat_a["descriptors"], feat_p["descriptors"])
            neg_mutual, _ = mutual_nn_rate(feat_a["descriptors"], feat_n["descriptors"])
            mutual_pos.append(float(pos_mutual.item()))
            mutual_neg.append(float(neg_mutual.item()))

        total_pos += torch.tensor(sum(pos_counts), device=device)
        total_neg += torch.tensor(sum(neg_counts), device=device)
        total_pos_conf += torch.tensor(sum(pos_confs), device=device)
        total_neg_conf += torch.tensor(sum(neg_confs), device=device)
        total_participation += torch.tensor(sum(participation), device=device)
        total_mutual_pos += torch.tensor(sum(mutual_pos), device=device)
        total_mutual_neg += torch.tensor(sum(mutual_neg), device=device)
        total_hard_neg += torch.tensor(hard_neg_this_batch, device=device)
        batch_count += 1
        n += len(feats_a)

        if args.save_eval_visuals and accelerator.is_main_process:
            h_orig, w_orig = anchors.shape[-2:]
            scale = np.array([w_orig / w_r, h_orig / h_r], dtype=np.float32)
            saved_visuals = maybe_save_eval_visuals(
                visuals_dir,
                anchors,
                positives,
                negatives,
                feats_a,
                feats_p,
                feats_n,
                pred_pos,
                pred_neg,
                scale,
                saved_visuals,
                args.num_eval_visuals,
                prefix,
            )

    total_pos = accelerator.reduce(total_pos, reduction="sum")
    total_neg = accelerator.reduce(total_neg, reduction="sum")
    total_pos_conf = accelerator.reduce(total_pos_conf, reduction="sum")
    total_neg_conf = accelerator.reduce(total_neg_conf, reduction="sum")
    total_participation = accelerator.reduce(total_participation, reduction="sum")
    total_mutual_pos = accelerator.reduce(total_mutual_pos, reduction="sum")
    total_mutual_neg = accelerator.reduce(total_mutual_neg, reduction="sum")
    total_hard_neg = accelerator.reduce(total_hard_neg, reduction="sum")
    batch_count = accelerator.reduce(batch_count, reduction="sum")
    n = accelerator.reduce(n, reduction="sum")

    mean_pos = (total_pos / n.clamp(min=1)).item()
    mean_neg = (total_neg / n.clamp(min=1)).item()
    metrics = {
        f"{prefix}/mean_matches_pos": mean_pos,
        f"{prefix}/mean_matches_neg": mean_neg,
        f"{prefix}/match_ratio": mean_pos / max(mean_neg, 1e-6),
    }

    if args.eval_metrics_profile == "extended":
        mean_pos_conf = (total_pos_conf / n.clamp(min=1)).item()
        mean_neg_conf = (total_neg_conf / n.clamp(min=1)).item()
        metrics.update(
            {
                f"{prefix}/mutual_nn_precision_pos": (total_mutual_pos / n.clamp(min=1)).item(),
                f"{prefix}/mutual_nn_precision_neg": (total_mutual_neg / n.clamp(min=1)).item(),
                f"{prefix}/keypoint_participation_pos": (total_participation / n.clamp(min=1)).item(),
                f"{prefix}/confidence_separation": mean_pos_conf - mean_neg_conf,
                f"{prefix}/mean_hardest_negative_matches": (total_hard_neg / batch_count.clamp(min=1)).item(),
            }
        )
    return metrics


def build_pair_mining_config(args: argparse.Namespace) -> PairMiningConfig:
    if args.use_pair_quality_mining:
        exclude_low = not args.include_low_quality_positives
        use_hard = not args.no_hard_negative_cache
    else:
        exclude_low = args.exclude_low_quality_positives and not args.include_low_quality_positives
        use_hard = args.use_hard_negative_cache and not args.no_hard_negative_cache
    return PairMiningConfig(
        use_pair_quality_mining=args.use_pair_quality_mining,
        pair_quality_cache_dir=args.pair_quality_cache_dir,
        positive_quality_mode=args.positive_quality_mode,
        positive_high_ratio=args.positive_high_ratio,
        positive_medium_ratio=args.positive_medium_ratio,
        exclude_low_quality_positives=exclude_low,
        use_hard_negative_cache=use_hard,
        hard_negative_ratio=args.hard_negative_ratio,
        max_positive_candidates_per_anchor=args.max_positive_candidates_per_anchor,
        max_negative_candidates_per_anchor=args.max_negative_candidates_per_anchor,
    )


def load_pair_quality_cache_for_training(args: argparse.Namespace) -> PairQualityCache | None:
    if not args.use_pair_quality_mining:
        return None
    if args.pair_quality_cache_dir is None:
        raise ValueError("--use_pair_quality_mining requires --pair_quality_cache_dir")
    cache = PairQualityCache.load(args.pair_quality_cache_dir)
    validate_cache_metadata(
        cache.metadata,
        data_root=args.train_data,
        rdd_weights=args.rdd_weights,
        lg_weights=args.lg_weights,
        resize=args.resize,
        top_k=args.top_k,
    )
    return cache


def build_train_loader(
    args: argparse.Namespace,
    train_transform: transforms.Compose,
    pair_quality_cache: PairQualityCache | None = None,
    pair_mining_config: PairMiningConfig | None = None,
):
    if args.batch_mode == "balanced":
        dataset = LabeledImageFolder(
            args.train_data,
            transform=train_transform,
            sequence_aware_sampling=args.sequence_aware_sampling,
            pair_quality_cache=pair_quality_cache,
            pair_mining_config=pair_mining_config,
        )
        sampler = BalancedBatchSampler(dataset, n_classes=args.n_classes, n_samples=args.n_samples_per_class)
        loader = get_loader(
            dataset,
            batch_size=args.n_classes * args.n_samples_per_class,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
            persistent_workers=args.num_workers > 0,
            batch_sampler=sampler,
        )
        return dataset, loader

    dataset = TripletImageFolder(
        args.train_data,
        transform=train_transform,
        sequence_aware_sampling=args.sequence_aware_sampling,
        pair_quality_cache=pair_quality_cache,
        pair_mining_config=pair_mining_config,
    )
    loader = get_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        persistent_workers=args.num_workers > 0,
    )
    return dataset, loader


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    accelerator = Accelerator(log_with="wandb" if args.project else None)
    device = accelerator.device
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_transform = build_train_transform(args)
    eval_transform = build_eval_transform()

    pair_mining_config = build_pair_mining_config(args)
    pair_quality_cache = load_pair_quality_cache_for_training(args)
    train_train_ds, train_loader = build_train_loader(
        args, train_transform, pair_quality_cache, pair_mining_config
    )
    train_triplet_eval_ds = TripletImageFolder(args.train_data, transform=eval_transform)
    val_ds = TripletImageFolder(args.val_data, transform=eval_transform)

    train_eval_loader = get_loader(
        train_triplet_eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = get_loader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        persistent_workers=args.num_workers > 0,
    )

    mini_train_ds = FixedTripletDataset(train_triplet_eval_ds, n_samples=10 * args.batch_size, seed=args.seed)
    mini_val_ds = FixedTripletDataset(val_ds, n_samples=10 * args.batch_size, seed=args.seed)
    mini_train_loader = get_loader(
        mini_train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    mini_val_loader = get_loader(
        mini_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    rdd = build_rdd(args.rdd_weights, device, args.top_k)
    lg = build_masked_lg(device, weights=args.lg_weights)
    lg.eval()

    optimizer = torch.optim.Adam([p for p in rdd.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    rdd, optimizer, train_loader, train_eval_loader, val_loader, mini_train_loader, mini_val_loader = accelerator.prepare(
        rdd, optimizer, train_loader, train_eval_loader, val_loader, mini_train_loader, mini_val_loader
    )

    if args.project:
        accelerator.init_trackers(args.project, config=vars(args), init_kwargs={"wandb": {"name": args.run_name}})

    global_step = 0
    if accelerator.is_main_process:
        print(
            f"Starting training | train_data={args.train_data} | val_data={args.val_data} | "
            f"resize={args.resize} | top_k={args.top_k} | batch_mode={args.batch_mode} | loss_type={args.loss_type} | "
            f"pair_quality_mining={args.use_pair_quality_mining} | retrieval_probe={args.use_retrieval_probe}",
            flush=True,
        )
        if args.use_pair_quality_mining:
            print(f"  pair_quality_cache_dir={args.pair_quality_cache_dir}", flush=True)
    for epoch in range(args.epochs):
        epoch_metrics, global_step = train_epoch(
            accelerator,
            rdd,
            lg,
            optimizer,
            train_loader,
            mini_train_loader,
            mini_val_loader,
            epoch,
            args.epochs,
            args,
            global_step,
        )

        train_eval_metrics = eval_epoch(accelerator, rdd, lg, train_eval_loader, args, prefix="train_eval", epoch=epoch)
        val_metrics = eval_epoch(accelerator, rdd, lg, val_loader, args, prefix="val", epoch=epoch)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        metrics = {"epoch": epoch, "train/lr": lr, **epoch_metrics, **train_eval_metrics, **val_metrics}

        if (
            args.use_retrieval_probe
            and accelerator.is_main_process
            and (epoch + 1) % max(1, args.retrieval_probe_every_n_epochs) == 0
        ):
            _unwrap(rdd).eval()
            lg.eval()
            probe_metrics = run_retrieval_probe(
                rdd,
                lg,
                args.train_data,
                args.val_data,
                device,
                num_queries=args.retrieval_probe_num_queries,
                gallery_per_id=args.retrieval_probe_gallery_per_id,
                frames_per_seq=args.retrieval_probe_frames_per_seq,
                top_m_pool=args.retrieval_probe_top_m_pool,
                resize=args.resize,
                top_k=args.top_k,
                seed=args.seed,
            )
            rdd.train()
            metrics.update(probe_metrics)
            print(
                f"  retrieval_probe: top1={probe_metrics['retrieval_probe/top1_acc']:.3f} "
                f"top5={probe_metrics['retrieval_probe/top5_acc']:.3f} "
                f"mAP={probe_metrics['retrieval_probe/mAP']:.3f} "
                f"balanced_top1={probe_metrics['retrieval_probe/balanced_top1_acc']:.3f}",
                flush=True,
            )
        if accelerator.is_main_process:
            accelerator.log(metrics, step=epoch)
            print(
                f"Epoch {epoch:02d} | loss={epoch_metrics['train/epoch_loss']:.4f} | lr={lr:.2e}\n"
                f"  train_eval: pos={train_eval_metrics['train_eval/mean_matches_pos']:.1f} "
                f"neg={train_eval_metrics['train_eval/mean_matches_neg']:.1f} "
                f"ratio={train_eval_metrics['train_eval/match_ratio']:.3f}\n"
                f"  val:        pos={val_metrics['val/mean_matches_pos']:.1f} "
                f"neg={val_metrics['val/mean_matches_neg']:.1f} "
                f"ratio={val_metrics['val/match_ratio']:.3f}"
            )
            accelerator.save_state(str(args.output_dir / f"epoch_{epoch:02d}"))

    if args.project:
        accelerator.end_training()


if __name__ == "__main__":
    main()
