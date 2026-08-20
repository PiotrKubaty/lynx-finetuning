"""
Rebuilds a training index's candidate pools with the CURRENT matcher.

Why
---
The index is mined once, with the pretrained matcher, and then frozen for the
whole run. Measured on the dense score matrix after 300 epochs (see
`score_matrix.py` and reports/), that is the dominant failure mode: the model
drives the 10,900 specific index negatives down to 0.0062 mean score while the
18.9M different-individual pairs it never saw stay flat (0.0134 -> 0.0123). Its
loss on the frozen index falls to 0.0403, but on negatives re-mined with itself
it is 0.5543 -- higher than the 0.4676 the run started from. Only ~9.6% of
training steps still carry a gradient. Re-mining is what closes that gap.

Cost, and why the gallery is thinned rather than shortlisted
------------------------------------------------------------
An exact re-mining is 2180 query frames x 4461 gallery frames = 9.72M matcher
calls, ~65 min on 4 GPUs at the measured 2513 pairs/s -- as expensive as the
entire 300-epoch training, so it cannot run often at full size.

The obvious fix -- a cheap coarse stage that shortlists "strong candidate
videos" and only expands those -- was simulated against the ground-truth matrix
and does NOT pay off here. The reason is in the data: the hardest negatives are
spread over 222 of 224 videos, and the finetuned model returns exactly zero for
89% of pairs, so a video's two or three representatives usually score 0 and rank
their video by noise. What does work is simpler -- keep the search exact and
shrink the gallery uniformly. Because the selection keeps one frame per video
anyway, and hard negatives are abundant rather than rare (the 50th hardest
negative still scores 0.349), a thinned gallery loses little hardness. Mean
score of the five selected negatives, against 0.0062 for the frozen index:

    gallery       calls    selected-negative score    vs exact
    20 fr/video   9.72M    0.6358                     100.0%
    10 fr/video   4.88M    0.6104                      96.0%
     5 fr/video   2.44M    0.5681                      89.4%
     3 fr/video   1.46M    0.5178                      81.4%

Even the cheapest setting is two orders of magnitude harder than what training
currently sees. For comparison, a coarse pass with 3 representatives expanding
the top 40 videos reached 93.4% at 3.0x -- no better than simply thinning to
~7 frames/video, for a great deal more machinery. The coarse stage is therefore
available (`--coarse_videos`) but off by default, and it only ever *adds*
videos to the thinned gallery; it never restricts the exact search.

Budget
------
At 2513 pairs/s on 4 GPUs, with ~65 min of training inside a 20 h envelope:

    gallery       per round     rounds in 19 h     re-mine every
    20 fr/video   65 min         17                 ~17 epochs
    10 fr/video   32 min         35                  ~8 epochs
     5 fr/video   16 min         71                  ~4 epochs
     3 fr/video   10 min        113                  ~2.7 epochs

`--fraction` refreshes only 1/k of the queries per round (round-robin), which
divides the per-round cost by k without changing the total -- the ANCE pattern
of a continuously, partially refreshed index rather than rare full rebuilds.
Staleness is the thing to buy down: with the pretrained model standing in for a
maximally stale one, its top-500 list (11% of the gallery) contains only 41% of
what the current model considers its five hardest negatives, and diffusion on a
kNN graph built from those stale scores scored *worse* than the stale ranking
itself at every setting tried. Nothing cheap substitutes for re-scoring; only
re-scoring less of the gallery, more often.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm

from contrastive_finetuning.keypoint_cache import open_cache_for_run
from contrastive_finetuning.models import build_masked_lg, build_rdd
from contrastive_finetuning.score_matrix import list_frames, load_features, _ragged, score_pairs
from contrastive_finetuning.train_common import (
    _lynx_id, _video_id, batch_features, diverse_topk_cols,
)


def add_remine_args(p: argparse.ArgumentParser) -> None:
    """Flags shared by the standalone script and the in-training re-miner."""
    p.add_argument(
        "--frames_per_video", type=int, default=20,
        help="Frames per video the gallery is sampled from, matching the original "
             "mining. --remine_gallery_per_video picks which of these get scored.",
    )
    p.add_argument(
        "--remine_pair_batch", type=int, default=128,
        help="Gallery frames per matcher call during re-mining. Throughput is flat "
             "from 64 to 512, so this is a memory knob; it is separate from the "
             "training --batch_size, which is far too small to be efficient here.",
    )
    p.add_argument(
        "--remine_gallery_per_video", type=int, default=5,
        help="Frames per video actually scored during re-mining. The search is exact "
             "within that set; this is the cost/hardness dial. 5 keeps 89.4%% of the "
             "hardness a full 20-frame gallery finds, at 4x less compute.",
    )
    p.add_argument(
        "--remine_top_k", type=int, default=0,
        help="Positives and negatives kept per query frame. 0 keeps whatever the "
             "existing entry has, so the index schema does not change.",
    )
    p.add_argument(
        "--coarse_videos", type=int, default=0,
        help="Optional coarse stage: rank whole videos by --coarse_reps frames each, "
             "then also score EVERY frame of the top this-many videos. 0 disables it. "
             "Measured no better than spending the same compute on a denser uniform "
             "gallery, so it is off by default; it only ever adds candidates.",
    )
    p.add_argument(
        "--coarse_reps", type=int, default=2,
        help="Representative frames per video in the coarse stage (evenly spaced).",
    )
    p.add_argument(
        "--carry_over", action=argparse.BooleanOptionalAction, default=True,
        help="Also score the entry's previous candidates, so a round can never return "
             "a weaker pool than it started with. ~10 extra pairs per query (0.1%% of "
             "a round); measured a negligible gain on its own, kept as cheap insurance.",
    )


@torch.inference_mode()
def _coarse_shortlist(lg, q_feats, pack, rep_cols: dict[str, list[int]],
                      query_video: str, top_videos: int, h: int, w: int,
                      pair_batch: int, device: torch.device) -> set[str]:
    """Videos whose representative frames score highest against one query frame."""
    kp, de, counts = pack[0], pack[1], pack[2]
    cols, vids = [], []
    for v, idxs in rep_cols.items():
        if v == query_video:
            continue
        cols.extend(idxs)
        vids.extend([v] * len(idxs))
    best: dict[str, float] = {}
    for s in range(0, len(cols), pair_batch):
        chunk = cols[s:s + pair_batch]
        sc, _ = score_pairs(lg, batch_features([q_feats] * len(chunk), h, w),
                            _ragged(kp, de, counts, chunk), h, w, device)
        for v, val in zip(vids[s:s + pair_batch], sc.tolist()):
            if val > best.get(v, float("-inf")):
                best[v] = val
    return set(sorted(best, key=lambda v: best[v], reverse=True)[:top_videos])


@torch.inference_mode()
def remine_entries(
    lg, entries: list[dict], gallery_rels: list[str], pack, query_feats,
    thin_cols: list[int], rep_cols: dict[str, list[int]],
    device: torch.device, args, rows: list[int], desc: str = "remine",
    quiet: bool = False,
) -> dict[int, dict]:
    """Fresh positives/negatives for `rows` of `entries`, scored by `lg`.

    Returns {entry index -> new entry} for the handled rows only, so a
    multi-process caller merges the shards itself.

    `pack` holds features for the WHOLE gallery (all frames of the split's
    sampling) -- they are cheap to keep resident. `thin_cols` is what actually
    gets scored; the coarse stage and carry-over add columns on top of it. The
    only approximation is therefore which columns are scored, never how.
    """
    kp_g, de_g, n_g, h, w = pack
    g_lynx = [_lynx_id(r) for r in gallery_rels]
    g_video = [_video_id(r) for r in gallery_rels]
    col_of = {rel: i for i, rel in enumerate(gallery_rels)}
    pair_batch = args.remine_pair_batch
    # Scoring order follows keypoint count so each matcher call pads to its own
    # chunk rather than the gallery-wide maximum (see score_matrix).
    kp_rank = {c: r for r, c in enumerate(torch.argsort(n_g).tolist())}

    out: dict[int, dict] = {}
    for r in tqdm(rows, desc=desc, disable=quiet):
        entry = entries[r]
        q_rel = entry["query_frame"]
        q_lynx, q_video = _lynx_id(q_rel), _video_id(q_rel)
        q_feats = query_feats(q_rel)

        cand = set(thin_cols)
        if args.carry_over:
            cand |= {col_of[p] for p in entry.get("positives", []) + entry.get("negatives", [])
                     if p in col_of}
        if args.coarse_videos:
            keep = _coarse_shortlist(lg, q_feats, pack, rep_cols, q_video,
                                     args.coarse_videos, h, w, pair_batch, device)
            cand |= {c for c, v in enumerate(g_video) if v in keep}
        cols = sorted((c for c in cand if g_video[c] != q_video), key=kp_rank.get)

        # The query side pads to its own keypoint count whatever it is paired
        # with, so its batch_features output is built once per distinct chunk
        # length (two in practice) instead of once per chunk.
        q_data: dict[int, dict] = {}
        scores: dict[int, float] = {}
        for s in range(0, len(cols), pair_batch):
            chunk = cols[s:s + pair_batch]
            if len(chunk) not in q_data:
                q_data[len(chunk)] = batch_features([q_feats] * len(chunk), h, w)
            sc, _ = score_pairs(lg, q_data[len(chunk)],
                                _ragged(kp_g, de_g, n_g, chunk), h, w, device)
            scores.update(zip(chunk, sc.tolist()))

        k = args.remine_top_k or len(entry["positives"])

        def pick(cand_cols: list[int]) -> list[str]:
            if not cand_cols:
                return []
            sub = [scores[c] for c in cand_cols]
            vids = [g_video[c] for c in cand_cols]
            return [gallery_rels[cand_cols[i]] for i in diverse_topk_cols(sub, vids, k)]

        new = dict(entry)
        # Falling back to the old pool keeps the entry valid if a class came up
        # empty (a lynx with a single video has no cross-video positives here).
        new["positives"] = pick([c for c in cols if g_lynx[c] == q_lynx]) or entry["positives"]
        new["negatives"] = pick([c for c in cols if g_lynx[c] != q_lynx]) or entry["negatives"]
        out[r] = new
    return out


def remine_index(
    lg, entries: list[dict], data_root: Path, split: str, cache, rdd,
    device: torch.device, args, rows: list[int] | None = None,
    accelerator: Accelerator | None = None, packs: dict | None = None,
) -> list[dict]:
    """One round: build the gallery, re-mine `rows`, return the updated entries.

    Rows outside `rows` keep their pools untouched, which is what makes
    `--fraction` a cheap, continuous alternative to rare full rebuilds.

    `packs` is an optional dict the caller keeps across rounds: the gallery and
    query features are loaded into it once and reused. Reading 4.5k cache files
    takes about as long as a quarter-index round actually scores, so a training
    run that re-mines every couple of epochs would otherwise spend more time on
    the filesystem than on the matcher. Costs ~1.8 GB of GPU memory held for the
    whole run (4461 gallery + 2180 query frames at 512 keypoints, fp16).
    """
    full_rels = list_frames(data_root, split, args.frames_per_video)
    prev = {p for e in entries for p in e.get("positives", []) + e.get("negatives", [])}
    query_rels = sorted({e["query_frame"] for e in entries})
    # In the normal case `prev` is already a subset of `full_rels` (the index was
    # mined from the same sampling), so this union is a no-op and the gallery is
    # identical every round — which is what makes `packs` reusable.
    gallery_rels = sorted(set(full_rels) | prev)
    thin = set(list_frames(data_root, split, args.remine_gallery_per_video))

    quiet = accelerator is not None and not accelerator.is_main_process
    store = packs if packs is not None else {}
    g_key = ("gallery", len(gallery_rels), gallery_rels[0], gallery_rels[-1])
    if g_key not in store:
        store.clear()
        store[g_key] = load_features(gallery_rels, data_root, cache, rdd, args.resize,
                                     args.top_k, device, desc="remine gallery")
    pack = store[g_key]
    col_of = {rel: i for i, rel in enumerate(gallery_rels)}
    thin_cols = sorted(col_of[r] for r in thin if r in col_of)

    rep_cols: dict[str, list[int]] = {}
    if args.coarse_videos:
        by_video: dict[str, list[int]] = {}
        for rel, c in col_of.items():
            by_video.setdefault(_video_id(rel), []).append(c)
        for v, idxs in by_video.items():
            idxs.sort()
            step = max(1, len(idxs) // max(1, args.coarse_reps))
            rep_cols[v] = idxs[::step][:args.coarse_reps]

    q_key = ("query", len(query_rels), query_rels[0], query_rels[-1])
    if q_key not in store:
        store[q_key] = load_features(query_rels, data_root, cache, rdd, args.resize,
                                     args.top_k, device, desc="remine queries")
    q_pack = store[q_key]
    q_pos = {rel: i for i, rel in enumerate(query_rels)}
    def query_feats(rel: str):
        return _ragged(q_pack[0], q_pack[1], q_pack[2], [q_pos[rel]])[0]

    rows = list(range(len(entries))) if rows is None else rows
    if accelerator is not None and accelerator.num_processes > 1:
        shard = rows[accelerator.process_index::accelerator.num_processes]
        mine = remine_entries(lg, entries, gallery_rels, pack, query_feats, thin_cols,
                              rep_cols, device, args, shard,
                              desc=f"remine[{accelerator.process_index}]", quiet=quiet)
        merged: dict[int, dict] = {}
        for part in gather_object([mine]):
            merged.update(part)
    else:
        merged = remine_entries(lg, entries, gallery_rels, pack, query_feats, thin_cols,
                                rep_cols, device, args, rows, quiet=quiet)

    return [merged.get(i, e) for i, e in enumerate(entries)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, required=True, help="Index whose pools are refreshed")
    p.add_argument("--out", type=Path, required=True, help="Where the new index JSON is written")
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--split", type=str, default="train", choices=["train", "test"],
                   help="Split the candidates are drawn from.")
    p.add_argument("--rdd_weights", type=str, default="rdd/weights/RDD-v2.pth")
    p.add_argument("--lg_weights", type=str, required=True,
                   help="Matcher doing the mining: the CURRENT checkpoint, .pth or .safetensors.")
    p.add_argument("--keypoint_cache", type=Path, default=None)
    p.add_argument("--resize", type=int, default=512)
    p.add_argument("--top_k", type=int, default=512)
    p.add_argument("--fraction", type=int, default=1,
                   help="Refresh only every f-th entry (round-robin with --round).")
    p.add_argument("--round", type=int, default=0, help="Which residue class --fraction refreshes.")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    add_remine_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    entries = json.loads(args.index.read_text())
    cache = (open_cache_for_run(args.keypoint_cache, args.rdd_weights, args.resize, args.top_k)
             if args.keypoint_cache is not None else None)
    rdd = None if cache is not None else build_rdd(args.rdd_weights, device, args.top_k)
    lg = build_masked_lg(device, weights=args.lg_weights)
    lg.eval()
    for param in lg.parameters():
        param.requires_grad_(False)

    rows = [i for i in range(len(entries)) if i % args.fraction == args.round % args.fraction]
    accelerator.print(f"re-mining {len(rows)}/{len(entries)} entries, scoring "
                      f"{args.remine_gallery_per_video} frames/video")
    t0 = time.time()
    new_entries = remine_index(lg, entries, args.data_root, args.split, cache, rdd,
                               device, args, rows, accelerator)
    dt = time.time() - t0

    if accelerator.is_main_process:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(new_entries, indent=1, ensure_ascii=False))
        changed = sum(1 for a, b in zip(entries, new_entries) if a["negatives"] != b["negatives"])
        accelerator.print(f"re-mined in {dt / 60:.1f} min; negatives changed in "
                          f"{changed}/{len(entries)} entries -> {args.out}")


if __name__ == "__main__":
    main()
