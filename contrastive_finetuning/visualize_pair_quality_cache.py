from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from contrastive_finetuning.build_pair_quality_cache import extract_features, resize_tensor_long_side
from contrastive_finetuning.models import build_lightglue, build_rdd
from contrastive_finetuning.pair_quality import PairQualityCache, candidate_quality_sort_key, validate_cache_metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize pair-quality cache selections as positive/negative match panels."
    )
    p.add_argument("--cache_dir", type=Path, required=True, help="Directory with metadata.json and pairs.pt")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--num_examples", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_matches", type=int, default=30)
    p.add_argument("--resize", type=int, default=None, help="Override cache resize (default: use cache metadata).")
    p.add_argument("--top_k", type=int, default=None, help="Override cache top_k (default: use cache metadata).")
    p.add_argument("--rdd_weights", type=str, default=None)
    p.add_argument("--lg_weights", type=str, default=None)
    p.add_argument(
        "--positive_selection",
        choices=["best_high", "best_composite", "worst_low"],
        default="best_high",
        help="Which cached positive candidate to visualize per anchor.",
    )
    p.add_argument(
        "--negative_selection",
        choices=["best_hard", "random_hard"],
        default="best_hard",
        help="Which cached hard negative candidate to visualize per anchor.",
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


def tensor_to_bgr_uint8(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim == 3:
        rgb = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    else:
        rgb = (tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_matches(
    image0_bgr: np.ndarray,
    image1_bgr: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    conf: np.ndarray,
    title: str,
    max_matches: int,
) -> np.ndarray:
    h0, w0 = image0_bgr.shape[:2]
    h1, w1 = image1_bgr.shape[:2]
    canvas = np.zeros((max(h0, h1) + 28, w0 + w1, 3), dtype=np.uint8)
    canvas[28 : 28 + h0, :w0] = image0_bgr
    canvas[28 : 28 + h1, w0:] = image1_bgr

    if len(conf) > 0:
        order = np.argsort(-conf)[:max_matches]
        kpts0 = kpts0[order]
        kpts1 = kpts1[order]
        palette = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        for idx, ((x0, y0), (x1, y1)) in enumerate(zip(kpts0, kpts1)):
            color = palette[idx % len(palette)]
            pt0 = (int(round(x0)), int(round(y0)) + 28)
            pt1 = (int(round(x1)) + w0, int(round(y1)) + 28)
            cv2.circle(canvas, pt0, 3, color, -1)
            cv2.circle(canvas, pt1, 3, color, -1)
            cv2.line(canvas, pt0, pt1, color, 1)

    cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


@torch.no_grad()
def match_pair_visual(
    lg: torch.nn.Module,
    feat_a: dict,
    feat_b: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    k0 = torch.from_numpy(feat_a["keypoints"]).to(device).unsqueeze(0)
    k1 = torch.from_numpy(feat_b["keypoints"]).to(device).unsqueeze(0)
    d0 = torch.from_numpy(feat_a["descriptors"]).to(device).unsqueeze(0)
    d1 = torch.from_numpy(feat_b["descriptors"]).to(device).unsqueeze(0)
    size0 = torch.tensor(feat_a["image_size"][::-1].copy(), device=device).unsqueeze(0)
    size1 = torch.tensor(feat_b["image_size"][::-1].copy(), device=device).unsqueeze(0)

    pred = lg(
        {
            "image0": {"keypoints": k0, "descriptors": d0, "image_size": size0},
            "image1": {"keypoints": k1, "descriptors": d1, "image_size": size1},
        }
    )
    matches = pred["matches"][0]
    scores = pred["scores"][0]
    if matches.numel() == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, np.empty((0,), dtype=np.float32), 0, 0.0

    kpts0 = feat_a["keypoints"][matches[:, 0].cpu().numpy()]
    kpts1 = feat_b["keypoints"][matches[:, 1].cpu().numpy()]
    conf = scores.detach().cpu().numpy()
    return kpts0, kpts1, conf, int(len(conf)), float(conf.mean())


def pick_positive(candidates: list[dict], mode: str, rng: random.Random) -> dict | None:
    if not candidates:
        return None
    if mode == "best_composite":
        return max(candidates, key=candidate_quality_sort_key)
    if mode == "worst_low":
        low = [c for c in candidates if c.get("quality_band") == "low"]
        pool = low or candidates
        return min(pool, key=candidate_quality_sort_key)
    high = [c for c in candidates if c.get("quality_band") == "high"]
    pool = high or candidates
    return max(pool, key=candidate_quality_sort_key)


def pick_negative(candidates: list[dict], mode: str, rng: random.Random) -> dict | None:
    if not candidates:
        return None
    if mode == "random_hard":
        top_k = min(8, len(candidates))
        return rng.choice(candidates[:top_k])
    return candidates[0]


def format_title(prefix: str, record: dict, live_matches: int, live_conf: float) -> str:
    band = record.get("quality_band", "?")
    composite = float(record.get("composite_score", 0.0))
    cached_matches = int(record.get("match_count", 0))
    participation = float(record.get("anchor_participation", 0.0))
    return (
        f"{prefix} | band={band} | composite={composite:.3f} | "
        f"cache_matches={cached_matches} live_matches={live_matches} live_conf={live_conf:.3f} | "
        f"participation={participation:.2f}"
    )


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device_name = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(device_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = PairQualityCache.load(args.cache_dir)
    metadata = cache.metadata
    data_root = Path(metadata["data_root"])
    resize = int(args.resize if args.resize is not None else metadata["resize"])
    top_k = int(args.top_k if args.top_k is not None else metadata["top_k"])
    rdd_weights = args.rdd_weights or metadata["rdd_weights"]
    lg_weights = args.lg_weights or metadata["lg_weights"]

    if (
        args.resize is None
        and args.top_k is None
        and args.rdd_weights is None
        and args.lg_weights is None
    ):
        validate_cache_metadata(
            metadata,
            data_root=data_root,
            rdd_weights=rdd_weights,
            lg_weights=lg_weights,
            resize=resize,
            top_k=top_k,
        )

    index_to_path: list[str] = cache.pairs["index_to_path"]
    anchor_indices = [
        idx
        for idx in cache.positive_candidates_by_anchor.keys()
        if cache.positive_candidates_by_anchor.get(idx) and cache.negative_candidates_by_anchor.get(idx)
    ]
    if not anchor_indices:
        raise ValueError(f"No scored anchors found in cache {args.cache_dir}")

    rng = random.Random(args.seed)
    chosen_anchors = rng.sample(anchor_indices, min(args.num_examples, len(anchor_indices)))

    rdd = build_rdd(rdd_weights, device, top_k)
    lg = build_lightglue(device, weights=lg_weights)

    feature_cache: dict[int, dict] = {}
    image_cache: dict[int, np.ndarray] = {}

    def get_features(idx: int) -> dict:
        if idx not in feature_cache:
            tensor = load_image_tensor(Path(index_to_path[idx]), resize)
            tensor = resize_tensor_long_side(tensor, resize)
            image_cache[idx] = tensor_to_bgr_uint8(tensor)
            feature_cache[idx] = extract_features(rdd, tensor, device)
        return feature_cache[idx]

    saved = 0
    for rank, anchor_idx in enumerate(chosen_anchors):
        pos_record = pick_positive(cache.positive_candidates_by_anchor[anchor_idx], args.positive_selection, rng)
        neg_record = pick_negative(cache.negative_candidates_by_anchor[anchor_idx], args.negative_selection, rng)
        if pos_record is None or neg_record is None:
            continue

        pos_idx = int(pos_record["candidate_index"])
        neg_idx = int(neg_record["candidate_index"])
        feat_a = get_features(anchor_idx)
        feat_p = get_features(pos_idx)
        feat_n = get_features(neg_idx)

        pos_k0, pos_k1, pos_conf, pos_n, pos_mean = match_pair_visual(lg, feat_a, feat_p, device)
        neg_k0, neg_k1, neg_conf, neg_n, neg_mean = match_pair_visual(lg, feat_a, feat_n, device)

        anchor_bgr = image_cache[anchor_idx]
        pos_bgr = image_cache[pos_idx]
        neg_bgr = image_cache[neg_idx]

        pos_title = format_title("cache pos", pos_record, pos_n, pos_mean)
        neg_title = format_title("cache neg", neg_record, neg_n, neg_mean)

        cv2.imwrite(
            str(args.output_dir / f"pair_{rank:03d}_pos.jpg"),
            draw_matches(anchor_bgr, pos_bgr, pos_k0, pos_k1, pos_conf, pos_title, args.max_matches),
        )
        cv2.imwrite(
            str(args.output_dir / f"pair_{rank:03d}_neg.jpg"),
            draw_matches(anchor_bgr, neg_bgr, neg_k0, neg_k1, neg_conf, neg_title, args.max_matches),
        )

        print(
            f"[{rank:03d}] anchor={anchor_idx} pos={pos_idx} band={pos_record.get('quality_band')} "
            f"composite={pos_record.get('composite_score', 0):.3f} | "
            f"neg={neg_idx} composite={neg_record.get('composite_score', 0):.3f}",
            flush=True,
        )
        saved += 1

    print(f"saved {saved} positive and {saved} negative panels to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
