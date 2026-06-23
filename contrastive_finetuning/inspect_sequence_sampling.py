from __future__ import annotations

import argparse
import random
from pathlib import Path

from contrastive_finetuning.loading import LabeledImageFolder, TripletImageFolder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect sequence-aware positive sampling")
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--num_examples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sequence_aware_sampling", type=lambda x: str(x).lower() in {"1", "true", "yes", "y", "on"}, default=True)
    p.add_argument("--compare_modes", action="store_true")
    return p.parse_args()


def describe_positive(anchor_meta, pos_meta, enabled: bool) -> str:
    if not enabled:
        return "identity-only random positive"
    if pos_meta.source_id != anchor_meta.source_id:
        return "different source"
    if pos_meta.sequence_id != anchor_meta.sequence_id:
        return "different sequence"
    if pos_meta.index != anchor_meta.index:
        return "same sequence fallback"
    return "no alternative positive available"


def print_pair_block(prefix: str, anchor_meta, pos_meta, neg_meta, enabled: bool) -> None:
    print(f"  {prefix} positive: {pos_meta.path} | source={pos_meta.source_id} | sequence={pos_meta.sequence_id}")
    print(f"  {prefix} negative: {neg_meta.path} | source={neg_meta.source_id} | sequence={neg_meta.sequence_id}")
    print(f"  {prefix} reason: {describe_positive(anchor_meta, pos_meta, enabled)}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    triplet_ds = TripletImageFolder(
        args.data_root,
        transform=None,
        sequence_aware_sampling=args.sequence_aware_sampling,
    )
    labeled_ds = LabeledImageFolder(
        args.data_root,
        transform=None,
        sequence_aware_sampling=args.sequence_aware_sampling,
    )

    print(f"dataset={args.data_root}")
    print(f"sequence_aware_sampling={args.sequence_aware_sampling}")
    print(f"num_identities={len(triplet_ds.classes)} num_images={len(triplet_ds)}")
    print()

    compare_triplet_ds = None
    if args.compare_modes:
        compare_triplet_ds = TripletImageFolder(args.data_root, transform=None, sequence_aware_sampling=not args.sequence_aware_sampling)
        print(f"compare_modes=True | comparison_sequence_aware_sampling={not args.sequence_aware_sampling}")
        print()

    chosen = rng.sample(range(len(triplet_ds)), min(args.num_examples, len(triplet_ds)))
    for rank, anchor_idx in enumerate(chosen, start=1):
        anchor_meta = triplet_ds._sample_meta[anchor_idx]
        pos_idx = triplet_ds._sample_positive_index(anchor_idx, rng=rng)
        neg_idx = triplet_ds._sample_negative_index(anchor_meta.label, rng=rng)
        pos_meta = triplet_ds._sample_meta[pos_idx]
        neg_meta = triplet_ds._sample_meta[neg_idx]

        print(f"[{rank:02d}] anchor_label={anchor_meta.identity}")
        print(f"  anchor: {anchor_meta.path} | source={anchor_meta.source_id} | sequence={anchor_meta.sequence_id}")
        print_pair_block("current", anchor_meta, pos_meta, neg_meta, args.sequence_aware_sampling)

        if compare_triplet_ds is not None:
            compare_pos_idx = compare_triplet_ds._sample_positive_index(anchor_idx, rng=random.Random(args.seed + rank))
            compare_neg_idx = compare_triplet_ds._sample_negative_index(anchor_meta.label, rng=random.Random(args.seed + 1000 + rank))
            compare_pos_meta = compare_triplet_ds._sample_meta[compare_pos_idx]
            compare_neg_meta = compare_triplet_ds._sample_meta[compare_neg_idx]
            print_pair_block("compare", anchor_meta, compare_pos_meta, compare_neg_meta, compare_triplet_ds.sequence_aware_sampling)
        print()

    print("Balanced batch preview:")
    labels = sorted(set(labeled_ds.targets))
    chosen_labels = labels[: min(3, len(labels))]
    for label in chosen_labels:
        picks = labeled_ds._sequence_diverse_indices(label, 2, rng=rng)
        metas = [labeled_ds._sample_meta[idx] for idx in picks]
        print(f"  identity={metas[0].identity}")
        for item in metas:
            print(f"    {item.path} | source={item.source_id} | sequence={item.sequence_id}")


if __name__ == "__main__":
    main()
