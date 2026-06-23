from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torchvision import transforms

from rdd.RDD.utils import to_pixel_coords
from contrastive_finetuning.loading import TripletImageFolder, FixedTripletDataset, get_loader
from contrastive_finetuning.models import build_rdd, build_masked_lg
from contrastive_finetuning.process import align_tensors_to_max_length


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contrastive fine-tuning of RDD descriptor")
    p.add_argument("--train_data",    type=Path, required=True)
    p.add_argument("--val_data",      type=Path, required=True)
    p.add_argument("--rdd_weights",   type=str,  default="rdd/weights/RDD-v2.pth")
    p.add_argument("--lg_weights",    type=str,  default="rdd/weights/RDD_lg-v2.pth")
    p.add_argument("--output_dir",    type=Path, default=Path("checkpoints"))
    p.add_argument("--project",       type=str,  default=None,  help="wandb project name")
    p.add_argument("--run_name",      type=str,  default=None)
    p.add_argument("--epochs",        type=int,  default=10)
    p.add_argument("--batch_size",    type=int,  default=8)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--margin",        type=float, default=0.5)
    p.add_argument("--resize",        type=int,  default=512)
    p.add_argument("--top_k",         type=int,  default=512)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--seed",          type=int,  default=0)
    p.add_argument("--num_workers",   type=int,  default=4)
    return p.parse_args()


# ── utils ─────────────────────────────────────────────────────────────────────
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return base model regardless of DDP / Accelerate wrapping."""
    return model.module if hasattr(model, "module") else model


def resize_long_side(images: torch.Tensor, size: int) -> torch.Tensor:
    """Resize so the long side == size and dims are div-by-32."""
    _, _, H, W = images.shape
    scale = size / max(H, W)
    new_H = int(H * scale) // 32 * 32
    new_W = int(W * scale) // 32 * 32
    return F.interpolate(images.float(), (new_H, new_W), mode="bilinear", align_corners=False)


def batch_features(feats: list[dict], image_h: int, image_w: int) -> dict:
    """
    Pack variable-length feature dicts into tensors for LightGlueMasked.

    Args:
        feats: list of B dicts with 'keypoints' (N_i, 2) and 'descriptors' (N_i, D)
        image_h, image_w: image dimensions for keypoint normalisation

    Returns dict with keypoints, descriptors, masks, image_size ready for LG.
    """
    ks = [f["keypoints"]   for f in feats]
    ds = [f["descriptors"] for f in feats]
    device = ks[0].device

    ks_pad, masks = align_tensors_to_max_length(ks)   # (B, M, 2), (B, M, 1)
    ds_pad, _     = align_tensors_to_max_length(ds)   # (B, M, D)

    # image_size as [W, H] — LightGlue normalize_keypoints convention
    sizes = torch.tensor(
        [image_w, image_h], device=device
    ).unsqueeze(0).expand(len(feats), -1).contiguous()

    return {
        "keypoints":   ks_pad,
        "descriptors": ds_pad,
        "image_size":  sizes,
        "masks":       masks.unsqueeze(1),  # (B, 1, M, 1) for masked attention
    }


# ── training-time feature extraction ─────────────────────────────────────────
def extract_train(rdd: torch.nn.Module, images: torch.Tensor) -> list[dict]:
    """
    One RDD forward pass with gradients flowing only through the descriptor.

    images must be div-by-32 aligned (use resize_long_side first).  Keypoint
    positions come from the frozen detector and are detached; descriptors at
    those positions retain their computation graph.

    Returns list of B dicts: {keypoints: Tensor(N,2), descriptors: Tensor(N,D)}
    """
    raw = _unwrap(rdd)
    B, _, H, W = images.shape

    # preprocess_tensor: dtype/device cast + div-by-32 resize (no-op since
    # images are already div-by-32)
    images_prep, rh, rw = raw.preprocess_tensor(images)
    _, _, H_p, W_p = images_prep.shape

    # Forward — gradients flow through descriptor (requires_grad=True),
    # not through detector (requires_grad=False)
    M1, K1, _ = rdd(images_prep)
    M1 = F.normalize(M1, dim=1)

    # Keypoint detection: frozen detector, no gradient needed
    with torch.no_grad():
        kpts, kscores, _ = raw.softdetect(K1)
        kpts    = torch.vstack([kpts[b].unsqueeze(0)    for b in range(B)])
        kscores = torch.vstack([kscores[b].unsqueeze(0) for b in range(B)])
        kpts_px     = to_pixel_coords(kpts, H_p, W_p)
        kpts_scaled = kpts_px * torch.tensor([rw, rh], device=images.device).view(1, -1)
        valid = kscores > raw.detection_threshold

    # Interpolate descriptor map at keypoints — F.grid_sample is differentiable
    descs = raw.interpolator(M1, kpts_px, H=H_p, W=W_p)
    descs = F.normalize(descs, dim=-1)

    return [
        {
            "keypoints":   kpts_scaled[b][valid[b]].detach(),
            "descriptors": descs[b][valid[b]],       # retains gradient
        }
        for b in range(B)
    ]


# ── loss ──────────────────────────────────────────────────────────────────────
def descriptor_triplet_loss(
    feats_a: list[dict],
    feats_p: list[dict],
    feats_n: list[dict],
    margin: float,
) -> torch.Tensor:
    """
    For each anchor descriptor, find its best match in the positive image and
    the negative image.  Encourages:
        similarity(anchor, positive) > similarity(anchor, negative) + margin

    loss_i = mean_over_keypoints( relu(margin - best_pos_sim + best_neg_sim) )
    loss   = mean_over_batch(loss_i)
    """
    losses = []
    for fa, fp, fn in zip(feats_a, feats_p, feats_n):
        da, dp, dn = fa["descriptors"], fp["descriptors"], fn["descriptors"]
        if min(da.shape[0], dp.shape[0], dn.shape[0]) == 0:
            continue
        best_pos = (da @ dp.T).max(dim=1).values   # (N_a,)
        best_neg = (da @ dn.T).max(dim=1).values   # (N_a,)
        losses.append(F.relu(margin - best_pos + best_neg).mean())

    if not losses:
        device = feats_a[0]["descriptors"].device
        return torch.zeros(1, device=device, requires_grad=True).squeeze()
    return torch.stack(losses).mean()


# ── training epoch ────────────────────────────────────────────────────────────
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
) -> tuple[float, int]:
    rdd.train()
    epoch_loss = 0.0
    steps_per_epoch = len(loader)

    for step, (anchors, positives, negatives) in enumerate(loader):
        device = accelerator.device
        anchors_r   = resize_long_side(anchors,   args.resize).to(device)
        positives_r = resize_long_side(positives, args.resize).to(device)
        negatives_r = resize_long_side(negatives, args.resize).to(device)

        feats_a = extract_train(rdd, anchors_r)
        feats_p = extract_train(rdd, positives_r)
        feats_n = extract_train(rdd, negatives_r)

        loss = descriptor_triplet_loss(feats_a, feats_p, feats_n, args.margin)

        optimizer.zero_grad()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(
            (p for p in _unwrap(rdd).parameters() if p.requires_grad),
            args.grad_clip,
        )
        optimizer.step()

        loss_val    = loss.item()
        epoch_loss += loss_val
        global_step += 1

        progress = (epoch * steps_per_epoch + step + 1) / (total_epochs * steps_per_epoch)

        if accelerator.is_main_process:
            accelerator.log({"train/loss": loss_val, "progress": progress}, step=global_step)
            if step % 20 == 0:
                print(f"  [epoch {epoch:02d} | step {step:04d} | progress={progress:.3f}] loss={loss_val:.4f}")

        if (step + 1) % 100 == 0:
            mini_train_m = eval_epoch(accelerator, rdd, lg, mini_train_loader, args, prefix="mini_train")
            mini_val_m   = eval_epoch(accelerator, rdd, lg, mini_val_loader,   args, prefix="mini_val")
            rdd.train()
            if accelerator.is_main_process:
                accelerator.log({**mini_train_m, **mini_val_m, "progress": progress}, step=global_step)
                print(
                    f"  [mini-eval @ progress={progress:.3f}]"
                    f"  train ratio={mini_train_m['mini_train/match_ratio']:.3f}"
                    f"  val ratio={mini_val_m['mini_val/match_ratio']:.3f}"
                )

    return epoch_loss / max(steps_per_epoch, 1), global_step


# ── validation epoch ──────────────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(
    accelerator: Accelerator,
    rdd: torch.nn.Module,
    lg: torch.nn.Module,
    loader,
    args: argparse.Namespace,
    prefix: str,
) -> dict:
    """
    Compute mean number of LightGlue matches for positive and negative pairs.
    A good descriptor should give many matches on positive pairs and few on
    negative pairs.  `prefix` is prepended to every returned metric key.
    """
    device = accelerator.device
    _unwrap(rdd).eval()
    total_pos = torch.zeros(1, device=device)
    total_neg = torch.zeros(1, device=device)
    n         = torch.zeros(1, device=device)

    for anchors, positives, negatives in loader:
        anchors_r   = resize_long_side(anchors,   args.resize).to(device)
        positives_r = resize_long_side(positives, args.resize).to(device)
        negatives_r = resize_long_side(negatives, args.resize).to(device)
        H_r, W_r = anchors_r.shape[-2:]

        feats_a = extract_train(_unwrap(rdd), anchors_r)
        feats_p = extract_train(_unwrap(rdd), positives_r)
        feats_n = extract_train(_unwrap(rdd), negatives_r)

        data_a = batch_features(feats_a, H_r, W_r)
        data_p = batch_features(feats_p, H_r, W_r)
        data_n = batch_features(feats_n, H_r, W_r)

        pred_pos = lg({"image0": data_a, "image1": data_p})
        pred_neg = lg({"image0": data_a, "image1": data_n})

        total_pos += sum(len(m) for m in pred_pos["matches"])
        total_neg += sum(len(m) for m in pred_neg["matches"])
        n += len(feats_a)

    # Sum counts across all processes before computing means
    total_pos = accelerator.reduce(total_pos, reduction="sum")
    total_neg = accelerator.reduce(total_neg, reduction="sum")
    n         = accelerator.reduce(n,         reduction="sum")

    mean_pos = (total_pos / n.clamp(min=1)).item()
    mean_neg = (total_neg / n.clamp(min=1)).item()
    return {
        f"{prefix}/mean_matches_pos": mean_pos,
        f"{prefix}/mean_matches_neg": mean_neg,
        f"{prefix}/match_ratio":      mean_pos / max(mean_neg, 1e-6),
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    accelerator = Accelerator(log_with="wandb" if args.project else None)
    device = accelerator.device

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──
    transform = transforms.ToTensor()
    train_ds = TripletImageFolder(args.train_data, transform=transform)
    val_ds   = TripletImageFolder(args.val_data,   transform=transform)

    train_loader = get_loader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, seed=args.seed,
    )
    train_eval_loader = get_loader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, seed=args.seed,
    )
    val_loader = get_loader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, seed=args.seed,
    )

    mini_train_ds = FixedTripletDataset(train_ds, n_samples=10 * args.batch_size, seed=args.seed)
    mini_val_ds   = FixedTripletDataset(val_ds,   n_samples=10 * args.batch_size, seed=args.seed)
    mini_train_loader = get_loader(
        mini_train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    mini_val_loader = get_loader(
        mini_val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )

    # ── models ──
    rdd = build_rdd(args.rdd_weights, device, args.top_k)
    lg  = build_masked_lg(device, weights=args.lg_weights)
    lg.eval()

    # Only descriptor parameters are trainable (detector frozen in build_rdd)
    optimizer = torch.optim.Adam(
        [p for p in rdd.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    rdd, optimizer, train_loader, train_eval_loader, val_loader, mini_train_loader, mini_val_loader = accelerator.prepare(
        rdd, optimizer, train_loader, train_eval_loader, val_loader, mini_train_loader, mini_val_loader
    )

    if args.project:
        accelerator.init_trackers(
            args.project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.run_name}},
        )

    # ── loop ──
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss, global_step = train_epoch(
            accelerator, rdd, lg, optimizer, train_loader,
            mini_train_loader, mini_val_loader,
            epoch, args.epochs, args, global_step,
        )

        train_eval_metrics = eval_epoch(accelerator, rdd, lg, train_eval_loader, args, prefix="train_eval")
        val_metrics        = eval_epoch(accelerator, rdd, lg, val_loader,        args, prefix="val")
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        metrics = {
            "epoch":            epoch,
            "train/epoch_loss": epoch_loss,
            "train/lr":         lr,
            **train_eval_metrics,
            **val_metrics,
        }

        if accelerator.is_main_process:
            accelerator.log(metrics, step=epoch)
            print(
                f"Epoch {epoch:02d} | loss={epoch_loss:.4f} | lr={lr:.2e}\n"
                f"  train_eval: pos={train_eval_metrics['train_eval/mean_matches_pos']:.1f}"
                f"  neg={train_eval_metrics['train_eval/mean_matches_neg']:.1f}"
                f"  ratio={train_eval_metrics['train_eval/match_ratio']:.3f}\n"
                f"  val:        pos={val_metrics['val/mean_matches_pos']:.1f}"
                f"  neg={val_metrics['val/mean_matches_neg']:.1f}"
                f"  ratio={val_metrics['val/match_ratio']:.3f}"
            )
            accelerator.save_state(str(args.output_dir / f"epoch_{epoch:02d}"))

    if args.project:
        accelerator.end_training()


if __name__ == "__main__":
    main()
