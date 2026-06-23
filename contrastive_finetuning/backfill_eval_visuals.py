from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from contrastive_finetuning.loading import FixedTripletDataset, TripletImageFolder, get_loader
from contrastive_finetuning.models import build_masked_lg, build_rdd
from contrastive_finetuning.train import (
    batch_features,
    build_eval_transform,
    draw_matches,
    extract_train,
    resize_long_side,
    seed_all,
    tensor_to_bgr,
    unpack_matches,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill a few validation match visualizations for saved checkpoints")
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--val_data", type=Path, required=True)
    p.add_argument("--rdd_weights", type=str, default="rdd/weights/RDD-v2.pth")
    p.add_argument("--lg_weights", type=str, default="rdd/weights/RDD_lg-v2.pth")
    p.add_argument("--resize", type=int, default=128)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--num_eval_visuals", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def epoch_dirs(run_dir: Path) -> list[Path]:
    return sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("epoch_")])


def save_epoch_visuals(
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
    max_visuals: int,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(min(len(feats_a), max_visuals)):
        anchor_bgr = tensor_to_bgr(anchors[idx])
        pos_bgr = tensor_to_bgr(positives[idx])
        neg_bgr = tensor_to_bgr(negatives[idx])
        key_a = feats_a[idx]["keypoints"].detach().cpu().numpy() * scale
        key_p = feats_p[idx]["keypoints"].detach().cpu().numpy() * scale
        key_n = feats_n[idx]["keypoints"].detach().cpu().numpy() * scale

        pts0_p, pts1_p, scores_p = unpack_matches(pred_pos, idx, key_a, key_p)
        pts0_n, pts1_n, scores_n = unpack_matches(pred_neg, idx, key_a, key_n)

        pos_title = f"val pos | matches={len(scores_p)} | conf={scores_p.mean():.3f}" if len(scores_p) else "val pos | matches=0 | conf=0.000"
        neg_title = f"val neg | matches={len(scores_n)} | conf={scores_n.mean():.3f}" if len(scores_n) else "val neg | matches=0 | conf=0.000"

        cv2.imwrite(str(save_dir / f"pair_{idx:03d}_pos.jpg"), draw_matches(pts0_p, pts1_p, scores_p, anchor_bgr, pos_bgr, pos_title))
        cv2.imwrite(str(save_dir / f"pair_{idx:03d}_neg.jpg"), draw_matches(pts0_n, pts1_n, scores_n, anchor_bgr, neg_bgr, neg_title))


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_base_ds = TripletImageFolder(args.val_data, transform=build_eval_transform())
    val_ds = FixedTripletDataset(val_base_ds, n_samples=max(args.num_eval_visuals, args.batch_size), seed=args.seed)
    val_loader = get_loader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        persistent_workers=False,
    )
    anchors, positives, negatives = next(iter(val_loader))

    rdd = build_rdd(args.rdd_weights, device, args.top_k)
    lg = build_masked_lg(device, weights=args.lg_weights)
    lg.eval()

    anchors_r = resize_long_side(anchors, args.resize).to(device)
    positives_r = resize_long_side(positives, args.resize).to(device)
    negatives_r = resize_long_side(negatives, args.resize).to(device)
    h_r, w_r = anchors_r.shape[-2:]
    h_orig, w_orig = anchors.shape[-2:]
    scale = np.array([w_orig / w_r, h_orig / h_r], dtype=np.float32)

    for epoch_dir in epoch_dirs(args.run_dir):
        checkpoint_path = epoch_dir / "model.safetensors"
        if not checkpoint_path.is_file():
            print(f"skip {epoch_dir.name}: missing model.safetensors", flush=True)
            continue

        state = load_file(str(checkpoint_path))
        missing, unexpected = rdd.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"loaded {epoch_dir.name} with missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        else:
            print(f"loaded {epoch_dir.name}", flush=True)

        with torch.no_grad():
            feats_a = extract_train(rdd, anchors_r)
            feats_p = extract_train(rdd, positives_r)
            feats_n = extract_train(rdd, negatives_r)
            data_a = batch_features(feats_a, h_r, w_r)
            data_p = batch_features(feats_p, h_r, w_r)
            data_n = batch_features(feats_n, h_r, w_r)
            pred_pos = lg({"image0": data_a, "image1": data_p})
            pred_neg = lg({"image0": data_a, "image1": data_n})

        epoch_num = int(epoch_dir.name.split("_")[-1])
        save_dir = args.run_dir / "val_visuals" / f"epoch_{epoch_num:02d}"
        save_epoch_visuals(
            save_dir,
            anchors,
            positives,
            negatives,
            feats_a,
            feats_p,
            feats_n,
            pred_pos,
            pred_neg,
            scale,
            args.num_eval_visuals,
        )
        print(f"saved visuals for {epoch_dir.name} -> {save_dir}", flush=True)


if __name__ == "__main__":
    main()
