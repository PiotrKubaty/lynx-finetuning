from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from torchvision import transforms

from contrastive_finetuning.keypoint_cache import open_cache_for_run
from contrastive_finetuning.loading import IndexAssignedTripletDataset
from contrastive_finetuning.models import build_rdd, build_masked_lg
from contrastive_finetuning.train_common import (
    build_pseudo_accuracy_loader, eval_pseudo_accuracy, seed_all,
)

"""
Evaluates frame- and video-level pseudo-accuracy on a single checkpoint —
no training. For every misclassified video, prints the query frame that
produced the winning score, the score itself, and the (wrong) candidate frame
that won instead of the correct lynx, once for the index's own candidate pool
and once for the ground-truth-blind mode-B pool (see eval_pseudo_accuracy).

`--lg_weights` accepts a finetuning run's `model.safetensors` as readily as a
pretrained `.pth`. Pass `--preselect_lg_weights` to split the two roles the way
`lynx_query_two_stage_in_parallel.py` does — the pretrained checkpoint picks the
mode-B pool (stage 1), the checkpoint under test only scores it (stage 2) — so
`video_accuracy_hybrid` measures a finetuned checkpoint against candidates it
had no hand in choosing. Without that flag both roles fall to `--lg_weights`.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index",       type=Path, required=True, help="JSON index to evaluate (query_frame/positives/negatives)")
    parser.add_argument("--data_root",   type=Path, default=None,  help="Root prepended to relative paths in the index")
    parser.add_argument("--rdd_weights", type=str,  default="rdd/weights/RDD-v2.pth")
    parser.add_argument("--lg_weights",  type=str,  default="rdd/weights/RDD_lg-v2.pth",
                        help="Checkpoint under test: a pretrained .pth or a finetuning run's "
                             "accelerate model.safetensors.")
    parser.add_argument(
        "--preselect_lg_weights", type=str, default=None,
        help="Checkpoint used ONLY to pick the mode-B pool behind video_accuracy_hybrid — "
             "stage 1 in lynx_query_two_stage_in_parallel.py. Point it at the pretrained "
             "checkpoint the index was mined with to judge a finetuned --lg_weights against "
             "a pool it did not choose for itself. Costs one extra pass over the index, "
             "whose own metrics are reported under the 'preselect/' prefix. Defaults to "
             "--lg_weights (the checkpoint under test preselects for itself).",
    )
    parser.add_argument(
        "--preselect_rdd_weights", type=str, default=None,
        help="RDD side of --preselect_lg_weights; defaults to --rdd_weights, which is "
             "normally right since finetuning runs with --trained_model lg leave RDD alone.",
    )
    parser.add_argument("--resize",      type=int,  default=512)
    parser.add_argument("--top_k",       type=int,  default=512)
    parser.add_argument("--num_workers", type=int,  default=4)
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Queries per DataLoader batch, and the max number of images per RDD "
             "forward call (RDD's deformable attention scales steeply with "
             "images-per-call, so the candidate pool is chunked to this size)",
    )
    parser.add_argument(
        "--mode_b_top_k", type=int, default=0,
        help="K: candidates kept per query frame by the ground-truth-blind preselection "
             "behind video_accuracy_hybrid (see eval_pseudo_accuracy). 0 means the index's "
             "own top_k, i.e. the number of positives per entry.",
    )
    parser.add_argument(
        "--keypoint_cache", type=Path, default=None,
        help="Prebuilt RDD keypoint cache to read detections from instead of "
             "running RDD (see contrastive_finetuning.build_keypoint_cache). RDD is "
             "frozen on this path by definition, so the cache is always valid here "
             "as long as it was built with the same --rdd_weights/--resize/--top_k, "
             "which it verifies on open.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    # eval_pseudo_accuracy/build_pseudo_accuracy_loader read these two
    # separately (training tunes them independently); here one flag drives both.
    args.eval_batch_size = args.batch_size
    return args


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    accelerator = Accelerator()
    device = accelerator.device

    transform = transforms.ToTensor()
    feature_cache = None
    if args.keypoint_cache is not None:
        feature_cache = open_cache_for_run(
            args.keypoint_cache, args.rdd_weights, args.resize, args.top_k)
    ds = IndexAssignedTripletDataset(
        args.index, root=args.data_root, transform=transform, feature_cache=feature_cache)
    loader = build_pseudo_accuracy_loader(accelerator, ds, args)

    def build(rdd_weights: str, lg_weights: str):
        rdd = build_rdd(rdd_weights, device, args.top_k)
        lg = build_masked_lg(device, weights=lg_weights)
        for p in list(rdd.parameters()) + list(lg.parameters()):
            p.requires_grad_(False)
        lg.eval()
        # Neither model is prepared: both are frozen, so there is nothing for DDP
        # to synchronise (and wrapping a fully-frozen module raises). The loader
        # above is what splits the queries across processes when launched
        # multi-GPU.
        return rdd, lg

    metrics = {}
    if args.preselect_lg_weights or args.preselect_rdd_weights:
        # eval_pseudo_accuracy freezes the mode-B pool on its first pass over a
        # dataset, so running the preselect checkpoint first is what hands the
        # checkpoint under test a pool chosen by someone else. This pass's own
        # metrics are the preselect checkpoint's baseline, worth having next to
        # the numbers below.
        pre_rdd, pre_lg = build(args.preselect_rdd_weights or args.rdd_weights,
                                args.preselect_lg_weights or args.lg_weights)
        metrics.update(
            eval_pseudo_accuracy(accelerator, pre_rdd, pre_lg, loader, args, prefix="preselect"))
        del pre_rdd, pre_lg
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rdd, lg = build(args.rdd_weights, args.lg_weights)
    metrics.update(
        eval_pseudo_accuracy(accelerator, rdd, lg, loader, args, prefix="eval", verbose=True))

    if accelerator.is_main_process:
        for k, v in metrics.items():
            accelerator.print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
