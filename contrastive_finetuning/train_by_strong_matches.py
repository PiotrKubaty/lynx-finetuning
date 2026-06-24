from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm
from torchvision import transforms

from torch.utils.data import Subset

from rdd.RDD.utils import to_pixel_coords
from contrastive_finetuning.loading import IndexAssignedTripletDataset, get_loader
from contrastive_finetuning.models import build_rdd, build_masked_lg
from contrastive_finetuning.process import align_tensors_to_max_length


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contrastive fine-tuning of RDD descriptor")
    p.add_argument("--train_index",   type=Path, required=True, help="JSON triplet index for training")
    p.add_argument("--val_index",     type=Path, required=True, help="JSON triplet index for validation")
    p.add_argument("--data_root",     type=Path, default=None,  help="Root prepended to relative paths in the index")
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
    p.add_argument("--eval_fraction", type=float, default=0.1, help="Fraction of dataset used for post-epoch eval")
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
def matched_descriptor_loss(
    feats_a: list[dict],
    feats_p: list[dict],
    feats_n: list[dict],
    lg: torch.nn.Module,
    image_h: int,
    image_w: int,
    margin: float,
) -> torch.Tensor:
    """
    Triplet loss using LightGlue matches as correspondence oracle.

    LightGlue runs under no_grad to produce match indices; the loss is then
    computed directly on those indexed descriptors (which retain their gradient
    from the RDD descriptor network).

    For each image in the batch:
      - pos_sims: dot products of LG-matched (anchor, positive) descriptor pairs
      - neg_sims: dot products of LG-matched (anchor, negative) descriptor pairs
      loss = relu(margin - mean(pos_sims) + mean(neg_sims))

    When LG finds no negative matches (ideal case), the loss reduces to
      relu(margin - pos_sims).mean()  — still encouraging tight pos pairs.
    """
    device = feats_a[0]["descriptors"].device

    with torch.no_grad():
        data_a = batch_features(feats_a, image_h, image_w)
        data_p = batch_features(feats_p, image_h, image_w)
        data_n = batch_features(feats_n, image_h, image_w)
        pred_pos = lg({"image0": data_a, "image1": data_p})
        pred_neg = lg({"image0": data_a, "image1": data_n})

    losses = []
    n_skipped = 0
    na_list, np_list, nn_list, pos_match_list, neg_match_list = [], [], [], [], []
    pos_sim_list, neg_sim_list = [], []

    for fa, fp, fn, m_pos, m_neg in zip(
        feats_a, feats_p, feats_n,
        pred_pos["matches"], pred_neg["matches"],
    ):
        da = fa["descriptors"]  # (Na, D) — gradient flows here
        dp = fp["descriptors"]  # (Np, D)
        dn = fn["descriptors"]  # (Nn, D)

        if m_pos.shape[0] == 0 or da.shape[0] == 0:
            n_skipped += 1
            continue

        na_list.append(da.shape[0])
        np_list.append(dp.shape[0])
        nn_list.append(dn.shape[0])
        pos_match_list.append(m_pos.shape[0])
        neg_match_list.append(m_neg.shape[0])

        pos_sims = (da[m_pos[:, 0]] * dp[m_pos[:, 1]]).sum(dim=-1)  # (M_pos,)
        pos_sim_list.append(pos_sims.mean().item())

        if m_neg.shape[0] > 0:
            neg_sims = (da[m_neg[:, 0]] * dn[m_neg[:, 1]]).sum(dim=-1)  # (M_neg,)
            neg_sim_list.append(neg_sims.mean().item())
            losses.append(F.relu(margin - pos_sims.mean() + neg_sims.mean()))
        else:
            losses.append(F.relu(margin - pos_sims).mean())

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    stats = {
        "n_skipped":        n_skipped,
        "mean_na":          _mean(na_list),
        "mean_np":          _mean(np_list),
        "mean_nn":          _mean(nn_list),
        "mean_pos_matches": _mean(pos_match_list),
        "mean_neg_matches": _mean(neg_match_list),
        "mean_pos_sim":     _mean(pos_sim_list),
        "mean_neg_sim":     _mean(neg_sim_list),
    }

    if not losses:
        stats["n_skipped"] += len(feats_a)
        return torch.zeros(1, device=device, requires_grad=True).squeeze(), stats
    return torch.stack(losses).mean(), stats


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
    epoch_loss     = 0.0
    epoch_skipped  = 0
    epoch_images   = 0
    mini_eval_time = 0.0
    steps_per_epoch = len(loader)
    t_epoch_start = time.perf_counter()

    pbar = tqdm(
        enumerate(loader),
        total=steps_per_epoch,
        desc=f"Epoch {epoch:02d}",
        disable=not accelerator.is_main_process,
    )
    for step, (anchors, positives, negatives) in pbar:
        device = accelerator.device
        anchors_r   = resize_long_side(anchors,   args.resize).to(device)
        positives_r = resize_long_side(positives, args.resize).to(device)
        negatives_r = resize_long_side(negatives, args.resize).to(device)

        H_r, W_r = anchors_r.shape[-2:]
        feats_a = extract_train(rdd, anchors_r)
        feats_p = extract_train(rdd, positives_r)
        feats_n = extract_train(rdd, negatives_r)

        loss, stats = matched_descriptor_loss(
            feats_a, feats_p, feats_n, lg, H_r, W_r, args.margin
        )
        epoch_skipped += stats["n_skipped"]
        epoch_images  += len(feats_a)

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
        pbar.set_postfix(loss=f"{loss_val:.4f}", skip=stats["n_skipped"])

        if accelerator.is_main_process:
            accelerator.log(
                {
                    "train/loss":    loss_val,
                    "train/mean_na": stats["mean_na"],
                    "train/mean_np": stats["mean_np"],
                    "train/mean_nn": stats["mean_nn"],
                    "progress":      progress,
                },
                step=global_step,
            )
            accelerator.log(
                {
                    "matches/mean_pos":     stats["mean_pos_matches"],
                    "matches/mean_neg":     stats["mean_neg_matches"],
                    "matches/mean_pos_sim": stats["mean_pos_sim"],
                    "matches/mean_neg_sim": stats["mean_neg_sim"],
                },
                step=global_step,
            )

        if (step + 1) % 100 == 0:
            t_mini_start = time.perf_counter()
            mini_train_m = eval_epoch(accelerator, rdd, lg, mini_train_loader, args, prefix="mini_train")
            mini_val_m   = eval_epoch(accelerator, rdd, lg, mini_val_loader,   args, prefix="mini_val")
            mini_eval_time += time.perf_counter() - t_mini_start
            rdd.train()
            if accelerator.is_main_process:
                accelerator.log({**mini_train_m, **mini_val_m, "progress": progress}, step=global_step)

    epoch_total_time = time.perf_counter() - t_epoch_start
    epoch_train_time = epoch_total_time - mini_eval_time
    skip_rate = epoch_skipped / max(epoch_images, 1)

    if accelerator.is_main_process:
        accelerator.log(
            {
                "train/skip_rate":    skip_rate,
                "train/n_skipped":    epoch_skipped,
                "time/train_s":       epoch_train_time,
                "time/mini_eval_s":   mini_eval_time,
            },
            step=global_step,
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
    }


# ── pseudo-accuracy eval ──────────────────────────────────────────────────────
@torch.no_grad()
def eval_pseudo_accuracy(
    accelerator: Accelerator,
    rdd: torch.nn.Module,
    lg: torch.nn.Module,
    dataset_subset,
    args: argparse.Namespace,
    prefix: str,
) -> dict:
    """
    For each query in the subset, run LG against every positive and every
    negative candidate listed in the JSON index.  The candidate with the most
    matches wins; the prediction is correct when that winner is a positive.

    Also returns mean match counts over all pos/neg pairs as a byproduct.
    """
    device = accelerator.device
    _unwrap(rdd).eval()

    if isinstance(dataset_subset, Subset):
        base_ds = dataset_subset.dataset
        entries = [base_ds._entries[i] for i in dataset_subset.indices]
    else:
        base_ds = dataset_subset
        entries = base_ds._entries

    accuracies: list[float] = []
    best_pos_scores: list[float] = []
    best_neg_scores: list[float] = []

    for entry in tqdm(entries, desc=f"{prefix}", leave=False, disable=not accelerator.is_main_process):
        query_img = base_ds._loader(base_ds._full_path(entry["query_frame"]))
        if base_ds.query_transform is not None:
            query_img = base_ds.query_transform(query_img)
        query_r = resize_long_side(query_img.unsqueeze(0).to(device), args.resize)
        H_q, W_q = query_r.shape[-2:]
        feats_q = extract_train(_unwrap(rdd), query_r)

        def _score(rel_path: str) -> float:
            cand_img = base_ds._loader(base_ds._full_path(rel_path))
            if base_ds.transform is not None:
                cand_img = base_ds.transform(cand_img)
            cand_r = resize_long_side(cand_img.unsqueeze(0).to(device), args.resize)
            H_c, W_c = cand_r.shape[-2:]
            feats_c = extract_train(_unwrap(rdd), cand_r)
            pred = lg({
                "image0": batch_features(feats_q, H_q, W_q),
                "image1": batch_features(feats_c, H_c, W_c),
            })
            s = pred["scores"][0]
            return s.mean().item() if s.numel() > 0 else 0.0

        score_pos = max(_score(p) for p in entry["positives"])
        score_neg = max(_score(n) for n in entry["negatives"])

        if score_pos > score_neg:
            accuracies.append(1.0)
        elif score_pos == score_neg:
            accuracies.append(0.5)
        else:
            accuracies.append(0.0)

        best_pos_scores.append(score_pos)
        best_neg_scores.append(score_neg)

    n = len(entries)
    return {
        f"{prefix}/pseudo_accuracy": sum(accuracies)    / max(n, 1),
        f"{prefix}/mean_score_pos":  sum(best_pos_scores) / max(n, 1),
        f"{prefix}/mean_score_neg":  sum(best_neg_scores) / max(n, 1),
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
    train_ds = IndexAssignedTripletDataset(
        args.train_index, root=args.data_root, transform=transform,
    )
    val_ds = IndexAssignedTripletDataset(
        args.val_index, root=args.data_root, transform=transform,
    )

    train_loader = get_loader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, seed=args.seed,
    )

    _rng = random.Random(args.seed)

    def _fixed_subset(ds, fraction):
        n = max(1, int(len(ds) * fraction))
        return Subset(ds, _rng.sample(range(len(ds)), min(n, len(ds))))

    mini_train_loader = get_loader(
        _fixed_subset(train_ds, 10 * args.batch_size / max(len(train_ds), 1)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    mini_val_loader = get_loader(
        _fixed_subset(val_ds, 10 * args.batch_size / max(len(val_ds), 1)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    eval_train_subset = _fixed_subset(train_ds, args.eval_fraction)
    eval_val_subset   = _fixed_subset(val_ds,   args.eval_fraction)

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

    rdd, optimizer, train_loader, mini_train_loader, mini_val_loader = accelerator.prepare(
        rdd, optimizer, train_loader, mini_train_loader, mini_val_loader
    )

    if args.project:
        accelerator.init_trackers(
            args.project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.run_name}},
        )

    # ── baseline eval (before any training) ──
    global_step = 0
    baseline_train = eval_pseudo_accuracy(accelerator, rdd, lg, eval_train_subset, args, prefix="train_eval")
    baseline_val   = eval_pseudo_accuracy(accelerator, rdd, lg, eval_val_subset,   args, prefix="val")
    rdd.train()
    if accelerator.is_main_process:
        accelerator.log({**baseline_train, **baseline_val, "epoch": -1}, step=global_step)

    # ── loop ──
    for epoch in range(args.epochs):
        epoch_loss, global_step = train_epoch(
            accelerator, rdd, lg, optimizer, train_loader,
            mini_train_loader, mini_val_loader,
            epoch, args.epochs, args, global_step,
        )

        t_eval_start = time.perf_counter()
        train_eval_metrics = eval_pseudo_accuracy(accelerator, rdd, lg, eval_train_subset, args, prefix="train_eval")
        val_metrics        = eval_pseudo_accuracy(accelerator, rdd, lg, eval_val_subset,   args, prefix="val")
        epoch_eval_time = time.perf_counter() - t_eval_start
        rdd.train()

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        metrics = {
            "epoch":              epoch,
            "train/epoch_loss":   epoch_loss,
            "train/lr":           lr,
            "time/epoch_eval_s":  epoch_eval_time,
            **train_eval_metrics,
            **val_metrics,
        }

        if accelerator.is_main_process:
            accelerator.log(metrics, step=global_step)
            accelerator.save_state(str(args.output_dir / f"epoch_{epoch:02d}"))

    if args.project:
        accelerator.end_training()


if __name__ == "__main__":
    main()
