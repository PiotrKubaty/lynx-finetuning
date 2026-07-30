from __future__ import annotations

import argparse
from pathlib import Path

from accelerate import Accelerator
from torchvision import transforms

from contrastive_finetuning.loading import IndexAssignedTripletDataset
from contrastive_finetuning.models import build_rdd, build_masked_lg
from contrastive_finetuning.train_common import (
    build_pseudo_accuracy_loader, eval_pseudo_accuracy, seed_all,
)

"""
Evaluates frame- and video-level pseudo-accuracy on a single checkpoint —
no training. For every misclassified video, prints the query frame that
produced the winning score, the score itself, and the (wrong) candidate frame
that won instead of the correct lynx.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index",       type=Path, required=True, help="JSON index to evaluate (query_frame/positives/negatives)")
    parser.add_argument("--data_root",   type=Path, default=None,  help="Root prepended to relative paths in the index")
    parser.add_argument("--rdd_weights", type=str,  default="rdd/weights/RDD-v2.pth")
    parser.add_argument("--lg_weights",  type=str,  default="rdd/weights/RDD_lg-v2.pth")
    parser.add_argument("--resize",      type=int,  default=512)
    parser.add_argument("--top_k",       type=int,  default=512)
    parser.add_argument("--num_workers", type=int,  default=4)
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Queries per DataLoader batch, and the max number of images per RDD "
             "forward call (RDD's deformable attention scales steeply with "
             "images-per-call, so the candidate pool is chunked to this size)",
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
    ds = IndexAssignedTripletDataset(args.index, root=args.data_root, transform=transform)
    loader = build_pseudo_accuracy_loader(accelerator, ds, args)

    rdd = build_rdd(args.rdd_weights, device, args.top_k)
    lg  = build_masked_lg(device, weights=args.lg_weights)
    for p in rdd.parameters():
        p.requires_grad_(False)
    for p in lg.parameters():
        p.requires_grad_(False)
    lg.eval()
    # Neither model is prepared: both are frozen, so there is nothing for DDP to
    # synchronise (and wrapping a fully-frozen module raises). The loader above
    # is what splits the queries across processes when launched multi-GPU.

    metrics = eval_pseudo_accuracy(accelerator, rdd, lg, loader, args, prefix="eval", verbose=True)

    if accelerator.is_main_process:
        for k, v in metrics.items():
            accelerator.print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
