"""
Dense query x gallery LightGlue score matrix for one split, with annotations.

What this is for
----------------
Training mines its negatives from a fixed index, so the model only ever sees
the negatives that the *pretrained* matcher found hard. Deciding what to
replace that with (see the re-mining discussion) needs to know what the full
pairwise score landscape actually looks like under the *finetuned* matcher:
which queries score high against which gallery frames, how much of that is
within-video redundancy, and how much signal survives if you only score a
sparse subset of pairs. This script produces that landscape once, offline, so
the analysis can be done on a matrix instead of on a 15-hour GPU job per idea.

The frames are sampled exactly like `lynx_find_strong_matches_in_parallel.py`
does (`np.linspace` over each video's sorted frames, `--frames_per_video`
of them), so rows and columns line up with the frames the mined index was
built from. Scores come from `train_common._lg_scores` — the same
sum(match confidence) / min(keypoints) used by the training loss and by
`eval_pseudo_accuracy` — so a number here is directly comparable to one there.

Cost
----
The train split is 224 videos, so at the default 20 frames/video this is
~4.5k x 4.5k = ~20M LightGlue calls. Two things keep that tractable:

  * `--keypoint_cache` removes RDD from the inner loop entirely — every
    frame's features are loaded once, up front, and live on the GPU for the
    whole run (~1.2 GB at 4.5k frames x 512 keypoints x 256 dims fp16).
  * the gallery is walked in order of keypoint count, so each LightGlue call
    pads to roughly the count its own chunk needs rather than to the global
    max. Attention cost is quadratic in the padded length, and this split runs
    99..512 keypoints/frame (mean 390), so the ordering alone is worth ~40%.

`--num_shards`/`--shard` split the query rows; shards are independent
processes (one per GPU), each writing its own row block, and `--merge`
stitches them into the final matrix. Shard files are the resume granularity —
`--resume` skips the ones already on disk.

Output (in `--out_dir`)
-----------------------
    shard_000of016.npz   per-shard: scores, n_matches, rows
    scores.npy           (Nq, Ng) float32   -- written by --merge
    n_matches.npy        (Nq, Ng) int32     -- matches with confidence > 0
    axis.json            per-frame annotations for both axes + run metadata

`axis.json` carries, for every row and column: the dataset-relative path,
`lynx_id`, `video` (the `split/lynx/site/sequence` id), the frame name, and
the frame's keypoint count — which is the cheapest available proxy for "is
this frame sharp enough to match anything", and is worth having next to the
scores when deciding which frames deserve a matcher call at all.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from contrastive_finetuning.keypoint_cache import open_cache_for_run
from contrastive_finetuning.models import build_masked_lg, build_rdd
from contrastive_finetuning.train_common import (
    _lg_scores, _lynx_id, _video_id, batch_features, extract_train,
    resize_long_side, seed_all,
)

SHARD_GLOB = "shard_*of*.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, required=True,
                   help="Dataset root containing train/ and test/.")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Where shard files and the merged matrix go.")
    p.add_argument("--split", type=str, default="train", choices=["train", "test"],
                   help="Split used for both axes unless --query_split/--gallery_split override it.")
    p.add_argument("--query_split", type=str, default=None, choices=["train", "test"])
    p.add_argument("--gallery_split", type=str, default=None, choices=["train", "test"])
    p.add_argument("--frames_per_video", type=int, default=20,
                   help="N frames sampled uniformly per video, as in the mining pipeline. "
                        "Videos with fewer frames contribute all of them.")
    p.add_argument("--rdd_weights", type=str, default="rdd/weights/RDD-v2.pth")
    p.add_argument("--lg_weights", type=str, default="rdd/weights/RDD_lg-v2.pth",
                   help="Matcher whose scores are measured: a pretrained .pth or a "
                        "finetuning run's model.safetensors.")
    p.add_argument("--keypoint_cache", type=Path, default=None,
                   help="Prebuilt RDD cache to read features from instead of running RDD "
                        "(contrastive_finetuning.build_keypoint_cache). Strongly recommended: "
                        "RDD is frozen here by definition, and the cache is verified against "
                        "--rdd_weights/--resize/--top_k on open.")
    p.add_argument("--resize", type=int, default=512)
    p.add_argument("--top_k", type=int, default=512)
    p.add_argument("--pair_batch", type=int, default=128,
                   help="Gallery frames per LightGlue call. Throughput is flat from 64 to 512 "
                        "on an A100 (the forward is compute-bound and scales linearly), so this "
                        "is a memory knob, not a speed one.")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                   help="Allow TF32 matmuls in LightGlue. Measured on an A100: 1.64x faster for "
                        "a mean score change of 1.4e-5 (max 9.6e-4) — an order of magnitude below "
                        "the 1.4e-4 run-to-run noise the fp16 descriptor cache already carries, "
                        "and the top-20 ranking of a scored chunk is unchanged. Pass --no-tf32 to "
                        "reproduce training-path arithmetic exactly.")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Split the query rows into this many independent row blocks.")
    p.add_argument("--shard", type=int, default=0, help="Which row block this process computes.")
    p.add_argument("--resume", action="store_true",
                   help="Skip this shard if its output file already exists.")
    p.add_argument("--symmetry_probe", type=int, default=256,
                   help="Score this many random pairs in both orders at startup and report the "
                        "disagreement. score(i,j) == score(j,i) would mean only the upper "
                        "triangle needs computing, halving the run. 0 disables the probe.")
    p.add_argument("--merge", action="store_true",
                   help="Don't score anything: stitch the shard files in --out_dir into "
                        "scores.npy / n_matches.npy and exit.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.query_split = args.query_split or args.split
    args.gallery_split = args.gallery_split or args.split
    if not (0 <= args.shard < args.num_shards):
        p.error(f"--shard must be in [0, {args.num_shards})")
    return args


# ── frame enumeration ─────────────────────────────────────────────────────────
def list_frames(root: Path, split: str, frames_per_video: int) -> list[str]:
    """Dataset-relative paths of the sampled frames, in a stable sorted order.

    Mirrors `lynx_benchmark.list_sequences` + `sample_frames`: videos are walked
    lynx/site/sequence in sorted order, and each contributes `frames_per_video`
    frames picked by `np.linspace` over its sorted frame list. Reproducing that
    exactly is what makes this matrix's axes the same frames the mined index
    drew from.
    """
    rels: list[str] = []
    for lynx_dir in sorted((root / split).iterdir()):
        if not lynx_dir.is_dir():
            continue
        for site_dir in sorted(lynx_dir.iterdir()):
            if not site_dir.is_dir():
                continue
            for seq_dir in sorted(site_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                frames = sorted(seq_dir.glob("frame_*.jpg"))
                if not frames:
                    continue
                if len(frames) > frames_per_video:
                    idxs = np.linspace(0, len(frames) - 1, num=frames_per_video, dtype=int)
                    frames = [frames[i] for i in idxs]
                rels.extend(f.relative_to(root).as_posix() for f in frames)
    return rels


def annotate(rels: list[str], counts: np.ndarray) -> list[dict]:
    """Per-frame records for one axis of the matrix."""
    return [
        {
            "path": rel,
            "lynx_id": _lynx_id(rel),
            "video": _video_id(rel),
            "frame": Path(rel).stem,
            "n_keypoints": int(n),
        }
        for rel, n in zip(rels, counts)
    ]


# ── features ──────────────────────────────────────────────────────────────────
def load_features(
    rels: list[str], root: Path, cache, rdd, resize: int, top_k: int,
    device: torch.device, desc: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """All frames' features, padded to `top_k` and resident on `device`.

    Returns (keypoints (N, top_k, 2) fp32, descriptors (N, top_k, D) fp16,
    counts (N,) int64 on CPU, H, W). Descriptors stay fp16 in storage and are
    cast per chunk in the inner loop — the whole point of holding them on the
    GPU is that the 20M-call loop never touches the filesystem.
    """
    kps, descs, counts, hws = [], [], [], set()
    if cache is not None:
        for rel in tqdm(rels, desc=f"load {desc}"):
            item = cache.load_padded(rel)
            kps.append(item["keypoints"])
            descs.append(item["descriptors"])
            counts.append(int(item["n_keypoints"]))
            hws.add(tuple(item["image_hw"].tolist()))
    else:
        # One image per RDD call: the fallback path runs once over ~4.5k frames
        # (a couple of minutes), and batching would need every frame in a batch
        # to share its *original* resolution, which nothing guarantees.
        for rel in tqdm(rels, desc=f"extract {desc}"):
            img = to_tensor(Image.open(root / rel).convert("RGB")).unsqueeze(0)
            img = resize_long_side(img, resize).to(device)
            with torch.no_grad():
                feat = extract_train(rdd, img)[0]
            n, d_dim = feat["keypoints"].shape[0], feat["descriptors"].shape[-1]
            kp = torch.zeros(top_k, 2, dtype=torch.float32)
            de = torch.zeros(top_k, d_dim, dtype=torch.float16)
            kp[:n] = feat["keypoints"].cpu()
            de[:n] = feat["descriptors"].half().cpu()
            kps.append(kp)
            descs.append(de)
            counts.append(n)
            hws.add(tuple(img.shape[-2:]))

    if len(hws) != 1:
        raise ValueError(
            f"frames resize to more than one size ({sorted(hws)}); batch_features "
            "normalizes a whole LightGlue call against a single image size, so a "
            "mixed-resolution axis would silently misplace keypoints"
        )
    h, w = hws.pop()
    return (
        torch.stack(kps).to(device),
        torch.stack(descs).to(device),
        torch.tensor(counts, dtype=torch.long),
        int(h), int(w),
    )


def _ragged(kp: torch.Tensor, de: torch.Tensor, counts: torch.Tensor,
            idxs) -> list[dict]:
    """Padded GPU tensors -> the list-of-dicts `batch_features` consumes (views, no copy)."""
    return [
        {"keypoints": kp[i, :counts[i]], "descriptors": de[i, :counts[i]].float()}
        for i in idxs
    ]


# ── scoring ───────────────────────────────────────────────────────────────────
@torch.inference_mode()
def score_pairs(lg, data_q: dict, g_feats: list[dict], h: int, w: int,
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """One LightGlue call over len(g_feats) pairs -> (scores, match counts).

    `data_q` is already `batch_features`-packed (callers reuse it across chunks);
    the gallery side is packed here because its padding follows its own chunk.
    """
    data_g = batch_features(g_feats, h, w)
    pred = lg({"image0": data_q, "image1": data_g})
    scores = _lg_scores(pred, data_q, data_g, device)
    matches = torch.stack([(c > 0).sum() for c in pred["scores"]])
    return scores, matches


@torch.inference_mode()
def score_rows(lg, rows, q_pack, g_pack, pair_batch: int, device: torch.device):
    """Score every gallery frame against each query row in `rows`.

    The gallery is visited in ascending keypoint-count order so that a chunk
    pads to its own maximum rather than the global one; results are scattered
    back to natural gallery order, so the returned rows are indexed by gallery
    position exactly as `axis.json` lists them.
    """
    kq, dq, nq, h, w = q_pack
    kg, dg, ng, _, _ = g_pack
    n_gallery = kg.shape[0]
    order = torch.argsort(ng)

    scores = np.empty((len(rows), n_gallery), dtype=np.float32)
    matches = np.empty((len(rows), n_gallery), dtype=np.int32)
    chunks = [(c, c.tolist()) for c in order.split(pair_batch)]

    pbar = tqdm(total=len(rows) * n_gallery, desc="pairs", unit="pair", unit_scale=True)
    for r, qi in enumerate(rows):
        row_s = torch.empty(n_gallery, device=device)
        row_m = torch.empty(n_gallery, dtype=torch.long, device=device)
        q_feat = _ragged(kq, dq, nq, [qi])[0]
        # The query side pads to its own keypoint count, which doesn't depend on
        # which gallery chunk it is paired with — so its batch_features output is
        # built once per distinct chunk length (two, in practice: the full chunk
        # and the remainder) instead of once per chunk.
        q_data_by_len: dict[int, dict] = {}
        for chunk, chunk_idx in chunks:
            if len(chunk_idx) not in q_data_by_len:
                q_data_by_len[len(chunk_idx)] = batch_features([q_feat] * len(chunk_idx), h, w)
            g_feats = _ragged(kg, dg, ng, chunk_idx)
            s, m = score_pairs(lg, q_data_by_len[len(chunk_idx)], g_feats, h, w, device)
            row_s[chunk] = s
            row_m[chunk] = m
            pbar.update(len(chunk_idx))
        scores[r] = row_s.cpu().numpy()
        matches[r] = row_m.cpu().numpy()
    pbar.close()
    return scores, matches


@torch.inference_mode()
def symmetry_probe(lg, pack, n_pairs: int, pair_batch: int, device: torch.device,
                   rng: np.random.Generator) -> dict:
    """Measure how far score(i, j) is from score(j, i) on random pairs.

    LightGlue's cross-attention is symmetric in principle, but image0 and
    image1 are not interchangeable in the implementation. If the disagreement
    is negligible the whole matrix is determined by its upper triangle, which
    halves a run this size — worth knowing before spending the GPU-hours, and
    cheap enough (a few hundred pairs) to check every time.
    """
    kp, de, n, h, w = pack
    total = kp.shape[0]
    i = rng.integers(0, total, size=n_pairs)
    j = rng.integers(0, total, size=n_pairs)
    fwd, bwd = [], []
    for s in range(0, n_pairs, pair_batch):
        a = _ragged(kp, de, n, i[s:s + pair_batch].tolist())
        b = _ragged(kp, de, n, j[s:s + pair_batch].tolist())
        fwd.append(score_pairs(lg, batch_features(a, h, w), b, h, w, device)[0].cpu())
        bwd.append(score_pairs(lg, batch_features(b, h, w), a, h, w, device)[0].cpu())
    fwd, bwd = torch.cat(fwd), torch.cat(bwd)
    delta = (fwd - bwd).abs()
    return {
        "pairs": int(n_pairs),
        "mean_abs_delta": float(delta.mean()),
        "max_abs_delta": float(delta.max()),
        "mean_score": float(fwd.mean()),
        "rel_mean_abs_delta": float(delta.mean() / max(fwd.mean().item(), 1e-9)),
    }


# ── merge ─────────────────────────────────────────────────────────────────────
def merge(out_dir: Path) -> None:
    """Stitch shard row blocks into the full matrix."""
    shards = sorted(out_dir.glob(SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(f"no {SHARD_GLOB} in {out_dir}")
    axis = json.loads((out_dir / "axis.json").read_text())
    n_q, n_g = len(axis["query"]), len(axis["gallery"])

    scores = np.full((n_q, n_g), np.nan, dtype=np.float32)
    matches = np.full((n_q, n_g), -1, dtype=np.int32)
    for path in tqdm(shards, desc="merge"):
        with np.load(path) as z:
            rows = z["rows"]
            scores[rows] = z["scores"]
            matches[rows] = z["n_matches"]

    missing = int(np.isnan(scores[:, 0]).sum())
    if missing:
        raise RuntimeError(
            f"{missing} of {n_q} query rows are not covered by the shard files in "
            f"{out_dir} — a shard is missing or was written for a different "
            "--num_shards. Rerun the missing shards before merging."
        )
    np.save(out_dir / "scores.npy", scores)
    np.save(out_dir / "n_matches.npy", matches)
    print(f"merged {len(shards)} shards -> {out_dir}/scores.npy  {scores.shape} "
          f"({scores.nbytes / 1e6:.0f} MB)")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(args.out_dir)
        return

    out_path = args.out_dir / f"shard_{args.shard:03d}of{args.num_shards:03d}.npz"
    if args.resume and out_path.exists():
        print(f"shard {args.shard}: {out_path} exists, skipping")
        return

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    q_rels = list_frames(args.data_root, args.query_split, args.frames_per_video)
    g_rels = (q_rels if args.gallery_split == args.query_split
              else list_frames(args.data_root, args.gallery_split, args.frames_per_video))
    print(f"query {args.query_split}: {len(q_rels)} frames | "
          f"gallery {args.gallery_split}: {len(g_rels)} frames | "
          f"{len(q_rels) * len(g_rels) / 1e6:.1f}M pairs total")

    cache = (open_cache_for_run(args.keypoint_cache, args.rdd_weights, args.resize, args.top_k)
             if args.keypoint_cache is not None else None)
    rdd = None if cache is not None else build_rdd(args.rdd_weights, device, args.top_k)
    lg = build_masked_lg(device, weights=args.lg_weights)
    lg.eval()

    q_pack = load_features(q_rels, args.data_root, cache, rdd, args.resize, args.top_k,
                           device, desc="query")
    g_pack = (q_pack if g_rels is q_rels
              else load_features(g_rels, args.data_root, cache, rdd, args.resize, args.top_k,
                                 device, desc="gallery"))

    # Both axes are fully known here, so every shard writes the same axis.json;
    # the merge step needs it and the last writer wins harmlessly.
    axis = {
        "query": annotate(q_rels, q_pack[2].numpy()),
        "gallery": annotate(g_rels, g_pack[2].numpy()),
        "meta": {
            "data_root": str(args.data_root),
            "query_split": args.query_split,
            "gallery_split": args.gallery_split,
            "frames_per_video": args.frames_per_video,
            "lg_weights": str(args.lg_weights),
            "rdd_weights": str(args.rdd_weights),
            "keypoint_cache": str(args.keypoint_cache) if args.keypoint_cache else None,
            "resize": args.resize,
            "top_k": args.top_k,
            "tf32": args.tf32,
            "image_hw": [q_pack[3], q_pack[4]],
            "score": "sum(match confidence) / min(keypoints) — train_common._lg_scores",
        },
    }
    (args.out_dir / "axis.json").write_text(json.dumps(axis, indent=1, ensure_ascii=False))

    if args.symmetry_probe:
        probe = symmetry_probe(lg, q_pack, args.symmetry_probe, args.pair_batch, device,
                               np.random.default_rng(args.seed))
        print(f"symmetry probe: mean |s(i,j) - s(j,i)| = {probe['mean_abs_delta']:.2e} "
              f"({probe['rel_mean_abs_delta']:.2%} of mean score {probe['mean_score']:.4f}), "
              f"max {probe['max_abs_delta']:.2e} over {probe['pairs']} pairs")
        (args.out_dir / "symmetry_probe.json").write_text(json.dumps(probe, indent=1))

    rows = np.array_split(np.arange(len(q_rels)), args.num_shards)[args.shard]
    print(f"shard {args.shard}/{args.num_shards}: rows {rows[0]}..{rows[-1]} "
          f"({len(rows)} queries x {len(g_rels)} gallery)")

    t0 = time.time()
    scores, matches = score_rows(lg, rows.tolist(), q_pack, g_pack, args.pair_batch, device)
    dt = time.time() - t0

    # Temp file + rename, so an interrupted shard never leaves a truncated .npz
    # that --resume would then skip.
    tmp = out_path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, scores=scores, n_matches=matches, rows=rows)
    tmp.replace(out_path)

    n_pairs = len(rows) * len(g_rels)
    print(f"shard {args.shard}: {n_pairs / 1e6:.2f}M pairs in {dt / 60:.1f} min "
          f"({n_pairs / dt:.0f} pairs/s) -> {out_path}")


if __name__ == "__main__":
    main()
