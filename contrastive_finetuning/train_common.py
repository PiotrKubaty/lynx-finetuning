from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm

from torch.utils.data import Subset

from rdd.RDD.utils import to_pixel_coords
from contrastive_finetuning.keypoint_cache import is_cached_batch, unpad_cached_features
from contrastive_finetuning.loading import PseudoAccuracyDataset, get_loader
from contrastive_finetuning.process import align_tensors_to_max_length


# ── CLI ───────────────────────────────────────────────────────────────────────
def add_common_args(p: argparse.ArgumentParser) -> None:
    """Arguments shared by all training scripts.

    Loss-specific hyperparameters (e.g. --lg_margin) are added by each
    script's own parse_args().
    """
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
    p.add_argument("--resize",        type=int,  default=512)
    p.add_argument("--top_k",         type=int,  default=512)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--seed",          type=int,  default=0)
    p.add_argument("--num_workers",   type=int,  default=4)
    p.add_argument("--eval_every_epochs", type=int, default=10)
    p.add_argument(
        "--eval_batch_size", type=int, default=4,
        help="Queries per DataLoader batch in eval_pseudo_accuracy (controls CPU "
             "decode/prefetch parallelism). Candidate images (eval_batch_size * "
             "(n_pos + n_neg) per DataLoader batch) are chunked to --batch_size "
             "before RDD's deformable attention, since that scales steeply with "
             "images-per-call and OOMs on larger top_k/top_m indices otherwise",
    )
    p.add_argument(
        "--keypoint_cache", type=Path, default=None,
        help="Directory holding a prebuilt RDD keypoint cache (see "
             "`python -m contrastive_finetuning.build_keypoint_cache`). When set, "
             "every RDD detection — training steps, both eval paths, and the "
             "pre-training measurement passes — is replaced by a lookup of "
             "precomputed keypoints/descriptors, and no image is decoded at all. "
             "Only valid with a FROZEN RDD (--trained_model lg) and a fixed input: "
             "--augment is rejected, since it changes the image RDD would have seen. "
             "The cache records the RDD weights hash, --resize, --top_k and the "
             "detection threshold it was built with, and refuses to open against a "
             "run that disagrees.",
    )
    p.add_argument(
        "--wandb_tags", type=str, default="",
        help="Comma-separated wandb tags for this run",
    )
    p.add_argument(
        "--mode_b_top_k", type=int, default=0,
        help="K: candidates kept per query frame by the ground-truth-blind preselection "
             "behind val/video_accuracy_hybrid (see eval_pseudo_accuracy). 0 means the "
             "index's own top_k, i.e. the number of positives per entry, which mirrors "
             "the index's negative pool; pass 2*top_k to match the index's total "
             "per-frame budget instead.",
    )
    p.add_argument(
        "--trained_model", type=str, default="lg", choices=["lg", "rdd", "lg+rdd"],
        help="Which model(s) are unfrozen and receive gradient: 'lg' (default) "
             "freezes RDD and trains only LightGlue; 'rdd' freezes LightGlue and "
             "trains only RDD; 'lg+rdd' trains both.",
    )


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


def resolve_trained_models(trained_model: str) -> tuple[bool, bool]:
    """Splits `--trained_model` ('lg' | 'rdd' | 'lg+rdd') into (train_rdd, train_lg)."""
    models = trained_model.split("+")
    return "rdd" in models, "lg" in models


def resize_long_side(images: torch.Tensor, size: int) -> torch.Tensor:
    """Resize so the long side == size and dims are div-by-32."""
    _, _, H, W = images.shape
    scale = size / max(H, W)
    new_H = int(H * scale) // 32 * 32
    new_W = int(W * scale) // 32 * 32
    return F.interpolate(images.float(), (new_H, new_W), mode="bilinear", align_corners=False)


def build_wandb_tags(args: argparse.Namespace) -> list[str]:
    """`--wandb_tags` (comma-separated) as a list; falls back to ["run"] if empty."""
    extra = getattr(args, "wandb_tags", "") or ""
    tags = [t.strip() for t in extra.split(",") if t.strip()]
    return tags or ["run"]


def log_code_to_wandb(accelerator: Accelerator, root: Path | None = None) -> None:
    """Snapshot every .py file under `root` into the run's wandb code artifact.

    Lets any run be traced back to the exact source that produced it, including
    uncommitted edits (wandb only records the git SHA + diff otherwise). Defaults
    to the repo root, i.e. the parent of this package, so `rdd/` and `rdd_patch/`
    are captured alongside `contrastive_finetuning/`.

    No-op on non-main processes and when no wandb tracker is active.
    """
    if not accelerator.is_main_process:
        return
    try:
        run = accelerator.get_tracker("wandb", unwrap=True)
    except ValueError:  # trackers configured, but none of them is wandb
        return
    # A blank GeneralTracker is what accelerate hands back when trackers only
    # exist on the main process; it has no wandb run behind it.
    if run is None or not hasattr(run, "log_code"):
        return

    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    run.log_code(str(root), include_fn=lambda path: path.endswith(".py"))


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
    One RDD forward pass. Whether RDD is frozen is decided by `--trained_model`
    in train_by_lg_matches.py; this stays a plain forward either way — the
    caller wraps it in torch.no_grad() when it wants to skip building the graph.

    images must be div-by-32 aligned (use resize_long_side first).

    Returns list of B dicts: {keypoints: Tensor(N,2), descriptors: Tensor(N,D)}
    """
    raw = _unwrap(rdd)
    B, _, H, W = images.shape

    # preprocess_tensor: dtype/device cast + div-by-32 resize (no-op since
    # images are already div-by-32)
    images_prep, rh, rw = raw.preprocess_tensor(images)
    _, _, H_p, W_p = images_prep.shape

    M1, K1, _ = rdd(images_prep)
    M1 = F.normalize(M1, dim=1)

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
            "descriptors": descs[b][valid[b]],
        }
        for b in range(B)
    ]


# ── LG matching ────────────────────────────────────────────────────────────────
def run_lg_matching_grad(
    lg: torch.nn.Module,
    feats_a: list[dict],
    feats_p: list[dict],
    feats_n: list[dict],
    image_h: int,
    image_w: int,
) -> tuple[dict, dict, dict, dict, dict]:
    """
    Run LightGlue WITHOUT a no_grad wrapper, so LG's own parameters receive
    gradient from any loss computed on the returned `scores` /
    `matching_scores0`. Returns the full prediction dicts (not just match
    indices) — used by train_by_lg_matches.py.

    Also returns the batch_features dicts (data_a/data_p/data_n): callers
    that want keypoint-coverage-normalized scores (see _lg_scores) need
    their `masks`.

    Note LightGlueMasked detaches its descriptor *inputs* by default (see
    `detach_descriptors` in rdd_patch/lightglue_masked.py), so gradient
    normally only reaches LG's own weights. train_by_lg_matches.py builds LG
    with `detach_descriptors=False` when `--trained_model` includes 'rdd', so
    gradient can also flow back into the network that produced feats_*.
    """
    data_a = batch_features(feats_a, image_h, image_w)
    data_p = batch_features(feats_p, image_h, image_w)
    data_n = batch_features(feats_n, image_h, image_w)
    pred_pos = lg({"image0": data_a, "image1": data_p})
    pred_neg = lg({"image0": data_a, "image1": data_n})
    return pred_pos, pred_neg, data_a, data_p, data_n


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
    A good matcher should give many matches on positive pairs and few on
    negative pairs.  `prefix` is prepended to every returned metric key.
    """
    device = accelerator.device
    _unwrap(rdd).eval()
    total_pos = torch.zeros(1, device=device)
    total_neg = torch.zeros(1, device=device)
    n         = torch.zeros(1, device=device)

    for anchors, positives, negatives in loader:
        feats_a, H_r, W_r = features_from_batch(anchors,   _unwrap(rdd), args.resize, device)
        feats_p, _,   _   = features_from_batch(positives, _unwrap(rdd), args.resize, device)
        feats_n, _,   _   = features_from_batch(negatives, _unwrap(rdd), args.resize, device)

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
def _video_id(rel_path: str) -> str:
    """Index paths look like `{split}/{lynx_id}/{location}/{video_id}/{frame}.jpg`."""
    return str(Path(rel_path).parent)


def _lynx_id(rel_path: str) -> str:
    return Path(rel_path).parts[1]


def features_from_batch(
    batch,
    rdd: torch.nn.Module,
    resize: int,
    device: torch.device,
    chunk_size: int | None = None,
) -> tuple[list[dict], int, int]:
    """Features for one DataLoader element, from the cache or from RDD.

    Returns `(feats, H, W)` where `feats` is the list-of-dicts shape
    `extract_train` produces and `batch_features` consumes, and `(H, W)` is the
    resized image size those keypoint coordinates live in.

    `batch` is either a stacked image tensor (the normal path, resized here and
    pushed through RDD) or the collated output of `KeypointCache.load_padded`
    (`--keypoint_cache`, where RDD never runs and no image was ever decoded).
    Every extraction site goes through this, so the two paths cannot drift
    apart — in particular the cached branch flattens leading batch dims the
    same way the image branch's explicit `.view(B * n, ...)` does, so a
    pseudo-accuracy candidate pool lines up identically.

    Pass `chunk_size` to bound images-per-RDD-call (see `_extract_chunked`); it
    is irrelevant to the cached branch, which has no such forward.
    """
    if is_cached_batch(batch):
        return unpad_cached_features(batch, device)
    images = resize_long_side(batch, resize).to(device)
    h, w = images.shape[-2:]
    if chunk_size is None:
        return extract_train(rdd, images), h, w
    return _extract_chunked(rdd, images, chunk_size), h, w


def _extract_chunked(rdd: torch.nn.Module, images: torch.Tensor, chunk_size: int) -> list[dict]:
    """extract_train, chunked along the batch dim.

    RDD's deformable attention scales steeply with images-per-forward-call, so
    peak GPU memory needs to be bounded independent of how many candidate
    images a DataLoader batch happens to bring along (which varies with the
    index's top_k/top_m).
    """
    feats: list[dict] = []
    for i in range(0, images.shape[0], chunk_size):
        feats.extend(extract_train(rdd, images[i:i + chunk_size]))
    return feats


def _lg_scores(pred: dict, q_data: dict, g_data: dict, device: torch.device) -> torch.Tensor:
    """Per-pair score = sum(match confidence) / min(valid keypoints in query, in candidate).

    Normalizing by keypoint coverage instead of averaging confidence over
    however many matches were found keeps a couple of lucky high-confidence
    matches from outscoring a pair that's genuinely well-matched throughout.

    Stays lazy/on-device throughout (no `.item()`) — this is also called from
    the training loss every step, where a GPU sync per batch element would
    actually cost something, unlike in eval.
    """
    B = q_data["keypoints"].shape[0]
    sums = torch.stack([
        pred["scores"][i].sum() if pred["scores"][i].numel() > 0 else torch.zeros((), device=device)
        for i in range(B)
    ])
    n_q = q_data["masks"].squeeze(1).squeeze(-1).sum(dim=1).clamp(min=1)
    n_g = g_data["masks"].squeeze(1).squeeze(-1).sum(dim=1).clamp(min=1)
    return sums / torch.minimum(n_q, n_g)


def _pseudo_batch_dims(cand_batch) -> tuple[int, int]:
    """(queries, candidates per query) for a PseudoAccuracyDataset batch."""
    if is_cached_batch(cand_batch):
        return tuple(cand_batch["n_keypoints"].shape[:2])
    return int(cand_batch.shape[0]), int(cand_batch.shape[1])


def _flatten_candidates(cand_batch):
    """Fold the per-query candidate dim into the batch dim, for either payload.

    Images need an explicit view; cached features are flattened downstream by
    `unpad_cached_features`, which handles any number of leading dims, so they
    pass through untouched.
    """
    if is_cached_batch(cand_batch):
        return cand_batch
    B, n_cand, C, H, W = cand_batch.shape
    return cand_batch.view(B * n_cand, C, H, W)


def _score_candidate_pool(
    rdd: torch.nn.Module,
    lg: torch.nn.Module,
    query_batch: torch.Tensor,
    cand_batch: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    """Score every query in a PseudoAccuracyDataset batch against its own candidates.

    `query_batch` is (B, C, H, W) and `cand_batch` is (B, n_cand, C, H, W) as
    they come off the loader — or, under `--keypoint_cache`, the collated
    feature dicts standing in for both; returns a (B, n_cand) score matrix
    either way.

    The single extraction site for the whole eval, so the mode-B pool and the
    scores it is applied to always come out of the same forward.
    """
    B, n_cand = _pseudo_batch_dims(cand_batch)

    feats_q, H_q, W_q = features_from_batch(
        query_batch, _unwrap(rdd), args.resize, device, chunk_size=args.batch_size)
    feats_c, H_c, W_c = features_from_batch(
        _flatten_candidates(cand_batch), _unwrap(rdd), args.resize, device,
        chunk_size=args.batch_size)

    # Each query's features are matched against its own n_cand candidates
    # positionally, so repeat them to line up as one flat (B * n_cand)
    # -sized batch for LG.
    feats_q_rep = [f for f in feats_q for _ in range(n_cand)]
    data_q = batch_features(feats_q_rep, H_q, W_q)
    data_c = batch_features(feats_c,     H_c, W_c)
    pred = lg({"image0": data_q, "image1": data_c})
    return _lg_scores(pred, data_q, data_c, device).view(B, n_cand)


def build_pseudo_accuracy_loader(
    accelerator: Accelerator,
    dataset_subset,
    args: argparse.Namespace,
):
    """
    Build the (Accelerate-prepared) DataLoader that eval_pseudo_accuracy runs on.

    Kept separate from eval_pseudo_accuracy, and called once per split before
    the training loop, for two reasons: the PseudoAccuracyDataset scan and the
    worker pool are then paid for once instead of on every evaluation, and —
    more importantly — `accelerator.prepare` shards the queries across
    processes, so an N-GPU run actually splits the work N ways instead of every
    rank redundantly scoring the whole index.

    `dataset_subset` is an IndexAssignedTripletDataset or a Subset of one; the
    returned loader's `.dataset` is the derived PseudoAccuracyDataset (which
    eval_pseudo_accuracy reads `n_pos`/`entries` off).
    """
    if isinstance(dataset_subset, Subset):
        base_ds = dataset_subset.dataset
        entries = [base_ds._entries[i] for i in dataset_subset.indices]
    else:
        base_ds = dataset_subset
        entries = base_ds._entries

    ds = PseudoAccuracyDataset(
        entries,
        root=base_ds.root,
        transform=base_ds.transform,
        query_transform=base_ds.query_transform,
        loader=base_ds._loader,
        feature_cache=base_ds.feature_cache,
    )
    loader = get_loader(
        ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    return accelerator.prepare(loader)


# ── mode-B candidate pool ─────────────────────────────────────────────────────
def diverse_topk_cols(scores: list[float], videos: list[str], k: int) -> list[int]:
    """Index port of lynx_find_strong_matches_in_parallel.select_diverse_topk.

    Picks up to k candidates by descending score while spreading picks across
    videos: candidates are grouped by video and sorted within each group, then
    taken round-robin over videos (best video first, by its top score), one
    frame per round. Same grouping, same (stable) sort order and same
    round-robin as the mining script — it just carries positions in the
    candidate list instead of dicts.
    """
    if k <= 0 or not scores:
        return []

    by_video: dict[str, list[int]] = defaultdict(list)
    for c in range(len(scores)):
        by_video[videos[c]].append(c)
    for v in by_video:
        by_video[v].sort(key=lambda c: scores[c], reverse=True)
    video_order = sorted(by_video.keys(), key=lambda v: scores[by_video[v][0]], reverse=True)

    selected: list[int] = []
    rank = 0
    while len(selected) < k:
        added_any = False
        for v in video_order:
            if rank < len(by_video[v]):
                selected.append(by_video[v][rank])
                added_any = True
                if len(selected) == k:
                    break
        if not added_any:
            break
        rank += 1
    return selected


def mode_b_pool_rows(
    scores: torch.Tensor, entries: list[dict], idxs: list[int], k: int
) -> torch.Tensor:
    """Mode-B pool of one batch: (B, n_cand) bool, row r for entry `idxs[r]`.

    `scores[r]` are entry idxs[r]'s candidate scores in `positives +
    negatives` order; the pool is one `select_diverse_topk` over all of them
    pooled together, i.e. with no slots reserved for the correct lynx_id.
    """
    rows = scores.cpu().tolist()  # one sync for the whole batch
    pool = torch.zeros(len(idxs), scores.shape[1], dtype=torch.bool)
    for r, i in enumerate(idxs):
        entry = entries[i]
        videos = [_video_id(p) for p in entry["positives"] + entry["negatives"]]
        pool[r, diverse_topk_cols(rows[r], videos, k)] = True
    return pool


def _log_mode_b_pool(accelerator: Accelerator, prefix: str, ds, mask: torch.Tensor) -> None:
    """One-off summary of the pool that was just frozen.

    `videos_with_positive` is the ceiling on video_accuracy_hybrid for this
    index and this preselect checkpoint: a video whose true lynx_id survives
    preselection on no query frame at all cannot be predicted correctly by any
    stage-2 checkpoint, however good.
    """
    kept_pos = mask[:, :ds.n_pos].any(dim=1)
    videos_with_positive = {
        _video_id(e["query_frame"]) for e, keeps in zip(ds.entries, kept_pos.tolist()) if keeps
    }
    videos = {_video_id(e["query_frame"]) for e in ds.entries}
    n = max(len(ds.entries), 1)
    accelerator.print(
        f"[{prefix}/hybrid] froze mode-B pool: {int(mask.sum())} candidates over "
        f"{len(ds.entries)} query frames "
        f"({int(mask[:, :ds.n_pos].sum()) / n:.2f} of the true lynx_id per frame); "
        f"true lynx_id survives on {int(kept_pos.sum())}/{len(ds.entries)} query frames "
        f"and in {len(videos_with_positive)}/{len(videos)} videos "
        f"({len(videos_with_positive) / max(len(videos), 1):.1%} — the ceiling on "
        f"{prefix}/video_accuracy_hybrid)"
    )


def _update_video_scores(
    store: dict[str, dict], video_id: str, true_lynx: str, query_frame: str,
    cand_paths: list[str], scores: list[float], keep: list[int] | None = None,
) -> None:
    """Fold one query frame's candidate scores into its video's record.

    `keep`, when given, is a per-candidate 0/1 mask restricting the pool to the
    mode-B preselection. The record accumulates the best score *per candidate
    lynx_id* over the video's whole pool — that ranking is what both the top-1
    and the top-5 prediction are read off — plus the single winning pair, for
    the verbose report.
    """
    rec = store.setdefault(video_id, {
        "true_lynx": true_lynx, "best_score": -float("inf"), "best_lynx": None,
        "best_query_frame": None, "best_cand_path": None, "by_lynx": {},
    })
    by_lynx = rec["by_lynx"]
    for c, (path, score) in enumerate(zip(cand_paths, scores)):
        if keep is not None and not keep[c]:
            continue
        lynx = _lynx_id(path)
        if score > by_lynx.get(lynx, -float("inf")):
            by_lynx[lynx] = score
        if score > rec["best_score"]:
            rec["best_score"] = score
            rec["best_lynx"] = lynx
            rec["best_query_frame"] = query_frame
            rec["best_cand_path"] = path


def _ranked_lynx(rec: dict) -> list[str]:
    """Candidate lynx_ids of one video, best score first.

    Ties keep insertion (first-seen) order, the same tie-break `best_lynx`
    applies, so `_ranked_lynx(rec)[0] == rec["best_lynx"]` always holds.
    """
    return sorted(rec["by_lynx"], key=lambda lynx: rec["by_lynx"][lynx], reverse=True)


def _video_accuracy(store: dict[str, dict], top_n: int = 1) -> float:
    """Fraction of videos whose true lynx_id is among the top_n predicted ones.

    With top_n=1 this is "the strongest pair in the pool carries the right
    identity". Larger top_n asks the softer question the retrieval benchmark's
    top-5 asks: for a video whose pool holds no frame of the true lynx_id at
    all (possible in the mode-B pool, never in the index's own), no top_n can
    make it correct.
    """
    correct = [
        1.0 if rec["true_lynx"] in _ranked_lynx(rec)[:top_n] else 0.0 for rec in store.values()
    ]
    return sum(correct) / max(len(correct), 1)


def _print_video_mismatches(accelerator: Accelerator, prefix: str, pool: str, store: dict[str, dict]) -> None:
    n_wrong = 0
    for video_id, rec in sorted(store.items()):
        if rec["best_lynx"] == rec["true_lynx"]:
            continue
        n_wrong += 1
        ranked = _ranked_lynx(rec)
        rank = ranked.index(rec["true_lynx"]) + 1 if rec["true_lynx"] in rec["by_lynx"] else None
        accelerator.print(
            f"[{prefix}/{pool}] MISMATCH video={video_id} true_lynx={rec['true_lynx']} "
            f"predicted_lynx={rec['best_lynx']} score={rec['best_score']:.4f} "
            f"true_lynx_rank={rank if rank else f'absent (of {len(ranked)} in pool)'} "
            f"query_frame={rec['best_query_frame']} matched_candidate={rec['best_cand_path']}"
        )
    accelerator.print(f"[{prefix}/{pool}] {n_wrong}/{len(store)} videos misclassified")


@torch.no_grad()
def eval_pseudo_accuracy(
    accelerator: Accelerator,
    rdd: torch.nn.Module,
    lg: torch.nn.Module,
    loader,
    args: argparse.Namespace,
    prefix: str,
    verbose: bool = False,
) -> dict:
    """
    For each query in the subset, run LG against every positive and every
    negative candidate listed in the JSON index.  The candidate with the most
    matches wins; the prediction is correct when that winner is a positive.

    Also returns mean match counts over all pos/neg pairs as a byproduct, and
    video-level accuracies. Paths look like
    ``{split}/{lynx_id}/{location}/{video_id}/{frame}.jpg``, so all query
    frames sharing a parent directory belong to the same video/individual.
    Each video's candidate lynx_ids are ranked by their best score anywhere in
    the video's pool (every query frame × every candidate); the video is
    correct at top-1 when the winner is its own lynx_id, and at top-5
    (``..._top5``) when its lynx_id is anywhere in the leading five. A pool
    that holds no frame of the true lynx_id at all is wrong at every top_n —
    which the index's own pool makes impossible and mode B's does not. The two
    pools differ only in which candidates the ranking runs over:

    ``video_accuracy_index``
        the whole pool the index lists for the query frame. The index reserves
        top_k slots for the correct lynx_id and makes every *other* identity
        share the other top_k, so the correct answer is guaranteed to be on the
        ballot — this number is optimistic by construction and is only
        comparable across checkpoints scored on the same index.

    ``video_accuracy_hybrid``
        only the candidates that survive the ground-truth-blind preselection
        ``lynx_query_two_stage_in_parallel.py`` calls mode B: one diverse top-k
        (`--mode_b_top_k`) over all identities pooled together, with no slots
        reserved for the correct one. The correct lynx_id has to earn its place
        like any other, so a video is only counted correct when its identity
        both survives preselection and then wins on score.

        That pool is computed here, on the fly, from the scores this function
        already needs — nothing is read from or written to the index file. It
        is computed **once**, on the first pass over a given dataset, and then
        frozen on ``PseudoAccuracyDataset.mode_b_mask`` and reused verbatim by
        every later pass, so the pool stays tied to the checkpoint that was
        loaded at that moment. In training that first pass is the baseline
        evaluation, before any optimizer step, i.e. exactly the pretrained
        checkpoint mode B's stage 1 is supposed to use; the pool then does not
        drift with the model being judged. (A standalone run such as
        eval_video_accuracy.py has only one pass, so there preselection and
        scoring are the same checkpoint by construction.)

        Reconstructing mode B from the index's own candidates is exact:
        `select_diverse_topk` picks round-robin over videos, so with a budget
        far below the number of gallery videos it only ever takes each chosen
        video's best frame, and a video in the global top-k is necessarily in
        the top-k of its own class (positives or negatives) — its best frame is
        therefore already in the entry. The mining script's top_m is likewise
        already applied to the entries, under the same ranking key mode B uses.

    `loader` comes from build_pseudo_accuracy_loader — a prepared DataLoader
    over a PseudoAccuracyDataset, so each process only scores its own shard of
    the queries and every query's full candidate pool goes through a single
    RDD+LG forward pass instead of being scored candidate by candidate. Every
    batch's results are gathered across processes (see gather_for_metrics
    below), so all ranks return identical, whole-split metrics — and, on the
    first pass, identical mode-B pools.

    If `verbose`, prints one line per misclassified video (wrong predicted
    lynx_id) with the winning query frame, its score, and the matched
    candidate frame, once per pool — see
    contrastive_finetuning/eval_video_accuracy.py.
    """
    device = accelerator.device
    _unwrap(rdd).eval()

    ds = loader.dataset
    entries = ds.entries
    # First pass over this dataset: derive the mode-B pool from the scores
    # below and freeze it. Every rank takes this branch together (they all hold
    # a dataset in the same state), which is what keeps the gather below
    # symmetric.
    freezing_pool = ds.mode_b_mask is None
    mode_b_top_k = getattr(args, "mode_b_top_k", 0) or ds.n_pos
    frozen_mask = (
        torch.zeros(len(entries), ds.n_pos + ds.n_neg, dtype=torch.bool) if freezing_pool else None
    )

    accuracies: list[float] = []
    best_pos_scores: list[float] = []
    best_neg_scores: list[float] = []
    # video_id -> {"true_lynx": str, "best_score": float, "best_lynx": str, ...}
    videos: dict[str, dict] = {}         # over the index's own pool
    videos_hybrid: dict[str, dict] = {}  # over the mode-B preselected pool

    for query_batch, cand_batch, idx_batch in tqdm(
        loader, desc=f"{prefix}", leave=False, disable=not accelerator.is_main_process
    ):
        if not is_cached_batch(query_batch):
            # Moved before the resize inside _score_candidate_pool, as this path
            # always has: interpolating this many candidate images is worth doing
            # on the GPU. The cached path has nothing to resize and moves its
            # tensors in unpad_cached_features.
            query_batch = query_batch.to(device)
            cand_batch  = cand_batch.to(device)

        scores = _score_candidate_pool(rdd, lg, query_batch, cand_batch, args, device)

        score_pos, _ = scores[:, :ds.n_pos].max(dim=1)
        score_neg, _ = scores[:, ds.n_pos:].max(dim=1)

        # idx_batch is still this rank's own shard here (the gather below is
        # what turns it into the all-ranks version); .cpu() because the prepared
        # loader may already have moved it to the device, while the pool lives
        # on CPU with the dataset.
        shard_idxs = idx_batch.cpu()
        if freezing_pool:
            pool = mode_b_pool_rows(scores, entries, shard_idxs.tolist(), mode_b_top_k)
        else:
            pool = ds.mode_b_mask[shard_idxs]

        # Collect this batch's results from every process before touching
        # Python: the loader is sharded, so a rank only ever sees a slice of
        # the queries, and the frame/video aggregation below needs the whole
        # split. gather_for_metrics (rather than plain gather) drops the
        # duplicate samples Accelerate pads the last batches with to keep
        # shard sizes equal. The whole score row travels, not just its argmax,
        # because the top-5 prediction ranks every candidate lynx_id; so does
        # the pool, so that every rank freezes the same whole-split mask.
        (score_pos, score_neg, scores, pool, idx_batch) = accelerator.gather_for_metrics(
            (score_pos, score_neg, scores,
             pool.to(device=device, dtype=torch.uint8), idx_batch.to(device))
        )
        if freezing_pool:
            frozen_mask[idx_batch.cpu()] = pool.cpu().bool()

        # One sync per batch (instead of one per query, let alone per
        # candidate) to pull the whole batch's results back to Python.
        for sp, sn, row, keep, idx in zip(
            score_pos.tolist(), score_neg.tolist(), scores.tolist(),
            pool.cpu().tolist(), idx_batch.tolist(),
        ):
            if sp > sn:
                accuracies.append(1.0)
            elif sp == sn:
                accuracies.append(0.5)
            else:
                accuracies.append(0.0)
            best_pos_scores.append(sp)
            best_neg_scores.append(sn)

            entry = ds.entries[idx]
            video_id  = _video_id(entry["query_frame"])
            true_lynx = _lynx_id(entry["query_frame"])
            query_frame = entry["query_frame"]
            cand_paths = entry["positives"] + entry["negatives"]

            _update_video_scores(videos, video_id, true_lynx, query_frame, cand_paths, row)
            _update_video_scores(
                videos_hybrid, video_id, true_lynx, query_frame, cand_paths, row, keep=keep)

    if freezing_pool:
        # diverse_topk_cols always returns at least one candidate, so an
        # all-False row means the entry was never scored — which would make the
        # frozen pool disagree with the metric just computed from it.
        unfilled = int((~frozen_mask.any(dim=1)).sum())
        if unfilled:
            raise RuntimeError(
                f"mode-B pool: {unfilled}/{len(entries)} query frames were never scored; "
                "refusing to freeze a partial pool"
            )
        ds.mode_b_mask = frozen_mask
        _log_mode_b_pool(accelerator, prefix, ds, frozen_mask)

    n = len(entries)

    if verbose:
        _print_video_mismatches(accelerator, prefix, "index", videos)
        _print_video_mismatches(accelerator, prefix, "hybrid", videos_hybrid)

    return {
        f"{prefix}/frame_accuracy":             sum(accuracies)      / max(n, 1),
        f"{prefix}/mean_score_pos":             sum(best_pos_scores) / max(n, 1),
        f"{prefix}/mean_score_neg":             sum(best_neg_scores) / max(n, 1),
        f"{prefix}/video_accuracy_index":       _video_accuracy(videos),
        f"{prefix}/video_accuracy_index_top5":  _video_accuracy(videos, top_n=5),
        f"{prefix}/video_accuracy_hybrid":      _video_accuracy(videos_hybrid),
        f"{prefix}/video_accuracy_hybrid_top5": _video_accuracy(videos_hybrid, top_n=5),
    }
