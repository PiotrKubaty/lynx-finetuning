from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from contrastive_finetuning.pair_quality import PairMiningConfig, PairQualityCache


class RandomLike(Protocol):
    def choice(self, seq): ...
    def sample(self, population, k: int): ...
    def shuffle(self, x): ...
    def choices(self, population, k: int): ...
    def random(self) -> float: ...


@dataclass(frozen=True)
class SampleMeta:
    index: int
    label: int
    identity: str
    source_id: str
    sequence_id: str
    path: Path


class _SequenceAwareImageFolderBase:
    pair_quality_cache: PairQualityCache | None = None
    pair_mining_config: PairMiningConfig | None = None

    def _init_sequence_metadata(self, root: str | Path, sequence_aware_sampling: bool) -> None:
        self._root = Path(root)
        self.sequence_aware_sampling = sequence_aware_sampling
        self._sample_meta: list[SampleMeta] = []
        self._class_to_indices: dict[int, list[int]] = defaultdict(list)
        self._class_to_source_to_indices: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        self._class_to_source_sequence_to_indices: dict[int, dict[tuple[str, str], list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for idx, (sample_path, label) in enumerate(self.samples):
            path = Path(sample_path)
            rel_parts = path.relative_to(self._root).parts
            if len(rel_parts) < 4:
                raise ValueError(
                    "Expected dataset hierarchy identity/source/sequence/frame, "
                    f"but got path '{path}' relative to '{self._root}'."
                )
            identity, source_id, sequence_id = rel_parts[0], rel_parts[1], rel_parts[2]
            meta = SampleMeta(
                index=idx,
                label=label,
                identity=identity,
                source_id=source_id,
                sequence_id=sequence_id,
                path=path,
            )
            self._sample_meta.append(meta)
            self._class_to_indices[label].append(idx)
            self._class_to_source_to_indices[label][source_id].append(idx)
            self._class_to_source_sequence_to_indices[label][(source_id, sequence_id)].append(idx)

    def _sample_positive_index(self, index: int, rng: RandomLike | random.Random = random) -> int:
        if (
            self.pair_quality_cache is not None
            and self.pair_mining_config is not None
            and self.pair_mining_config.use_pair_quality_mining
        ):
            mined = self.pair_quality_cache.sample_positive_index(
                index, self, rng, self.pair_mining_config
            )
            if mined is not None:
                return mined

        label = self.targets[index]
        if not self.sequence_aware_sampling:
            pos_pool = self._class_to_indices[label]
            pos_idx = index
            while pos_idx == index and len(pos_pool) > 1:
                pos_idx = rng.choice(pos_pool)
            return pos_idx

        meta = self._sample_meta[index]
        source_groups = self._class_to_source_to_indices[label]
        sequence_groups = self._class_to_source_sequence_to_indices[label]

        cross_source = [
            candidate
            for source_id, indices in source_groups.items()
            if source_id != meta.source_id
            for candidate in indices
        ]
        if cross_source:
            return rng.choice(cross_source)

        cross_sequence = [
            candidate
            for (source_id, sequence_id), indices in sequence_groups.items()
            if source_id == meta.source_id and sequence_id != meta.sequence_id
            for candidate in indices
        ]
        if cross_sequence:
            return rng.choice(cross_sequence)

        same_sequence = [
            candidate
            for candidate in sequence_groups[(meta.source_id, meta.sequence_id)]
            if candidate != index
        ]
        if same_sequence:
            return rng.choice(same_sequence)
        return index

    def _sample_negative_index(self, label: int, rng: RandomLike | random.Random = random, anchor_idx: int | None = None) -> int:
        if (
            anchor_idx is not None
            and self.pair_quality_cache is not None
            and self.pair_mining_config is not None
            and self.pair_mining_config.use_pair_quality_mining
        ):
            mined = self.pair_quality_cache.sample_negative_index(
                anchor_idx, label, self, rng, self.pair_mining_config
            )
            if mined is not None:
                return mined

        neg_label = label
        while neg_label == label:
            neg_label = rng.choice(list(self._class_to_indices.keys()))
        return rng.choice(self._class_to_indices[neg_label])

    def _sequence_diverse_indices(self, label: int, n_samples: int, rng: RandomLike | random.Random = random) -> list[int]:
        if (
            self.pair_quality_cache is not None
            and self.pair_mining_config is not None
            and self.pair_mining_config.use_pair_quality_mining
            and n_samples >= 2
        ):
            seed_pool = self._class_to_indices[label]
            seed_idx = rng.choice(seed_pool)
            mined_group = self.pair_quality_cache.sample_balanced_group_indices(
                seed_idx, label, n_samples, self, rng, self.pair_mining_config
            )
            if mined_group is not None and len(mined_group) == n_samples:
                return mined_group

        if not self.sequence_aware_sampling:
            pool = self._class_to_indices[label]
            return rng.choices(pool, k=n_samples) if len(pool) < n_samples else rng.sample(pool, n_samples)

        source_groups = {
            source_id: indices.copy() for source_id, indices in self._class_to_source_to_indices[label].items()
        }
        sequence_groups = {
            key: indices.copy() for key, indices in self._class_to_source_sequence_to_indices[label].items()
        }
        for indices in source_groups.values():
            rng.shuffle(indices)
        for indices in sequence_groups.values():
            rng.shuffle(indices)

        selected: list[int] = []
        used_sequences: set[tuple[str, str]] = set()

        source_keys = list(source_groups.keys())
        rng.shuffle(source_keys)
        for source_id in source_keys:
            if len(selected) >= n_samples:
                break
            indices = source_groups[source_id]
            if indices:
                choice = indices.pop()
                selected.append(choice)
                meta = self._sample_meta[choice]
                used_sequences.add((meta.source_id, meta.sequence_id))

        sequence_keys = list(sequence_groups.keys())
        rng.shuffle(sequence_keys)
        for key in sequence_keys:
            if len(selected) >= n_samples:
                break
            if key in used_sequences:
                continue
            indices = sequence_groups[key]
            remaining = [idx for idx in indices if idx not in selected]
            if remaining:
                choice = remaining.pop()
                selected.append(choice)
                meta = self._sample_meta[choice]
                used_sequences.add((meta.source_id, meta.sequence_id))

        unique_remaining = [idx for idx in self._class_to_indices[label] if idx not in selected]
        rng.shuffle(unique_remaining)
        while unique_remaining and len(selected) < n_samples:
            selected.append(unique_remaining.pop())

        if len(selected) < n_samples:
            pool = self._class_to_indices[label]
            selected.extend(rng.choices(pool, k=n_samples - len(selected)))
        return selected


def attach_pair_quality_cache(
    dataset: _SequenceAwareImageFolderBase,
    cache: PairQualityCache | None,
    config: PairMiningConfig | None,
) -> None:
    dataset.pair_quality_cache = cache
    dataset.pair_mining_config = config


class TripletImageFolder(_SequenceAwareImageFolderBase, Dataset):
    """ImageFolder wrapper that returns (anchor, positive, negative) triplets."""

    def __init__(
        self,
        root: str | Path,
        transform: transforms.Compose | None = None,
        anchor_transform: transforms.Compose | None = None,
        sequence_aware_sampling: bool = False,
        pair_quality_cache: PairQualityCache | None = None,
        pair_mining_config: PairMiningConfig | None = None,
    ) -> None:
        self._base = ImageFolder(root=str(root), transform=None)
        self.transform = transform
        self.anchor_transform = anchor_transform or transform
        self.samples = self._base.samples
        self.targets = self._base.targets
        self.classes = self._base.classes
        self.class_to_idx = self._base.class_to_idx
        self._init_sequence_metadata(root, sequence_aware_sampling=sequence_aware_sampling)
        attach_pair_quality_cache(self, pair_quality_cache, pair_mining_config)

    def __len__(self) -> int:
        return len(self._base)

    def _load(self, idx: int):
        path, _ = self._base.samples[idx]
        return self._base.loader(path)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_label = self._base.targets[index]
        pos_idx = self._sample_positive_index(index)
        neg_idx = self._sample_negative_index(anchor_label, anchor_idx=index)

        anchor_img = self._load(index)
        pos_img = self._load(pos_idx)
        neg_img = self._load(neg_idx)

        if self.anchor_transform is not None:
            anchor_img = self.anchor_transform(anchor_img)
        if self.transform is not None:
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        return anchor_img, pos_img, neg_img


class FixedTripletDataset(Dataset):
    """Deterministic triplet dataset: triplets are pre-sampled once at construction."""

    def __init__(self, base: TripletImageFolder, n_samples: int, seed: int = 42) -> None:
        rng = random.Random(seed)
        self._base = base
        anchor_indices = rng.sample(range(len(base)), min(n_samples, len(base)))
        self._triplets: list[tuple[int, int, int]] = []
        for anchor_idx in anchor_indices:
            anchor_label = base._base.targets[anchor_idx]
            pos_idx = base._sample_positive_index(anchor_idx, rng=rng)
            neg_idx = base._sample_negative_index(anchor_label, rng=rng, anchor_idx=anchor_idx)
            self._triplets.append((anchor_idx, pos_idx, neg_idx))

    def __len__(self) -> int:
        return len(self._triplets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_idx, pos_idx, neg_idx = self._triplets[index]
        base = self._base
        anchor_img = base._load(anchor_idx)
        pos_img = base._load(pos_idx)
        neg_img = base._load(neg_idx)
        if base.anchor_transform is not None:
            anchor_img = base.anchor_transform(anchor_img)
        if base.transform is not None:
            pos_img = base.transform(pos_img)
            neg_img = base.transform(neg_img)
        return anchor_img, pos_img, neg_img


class LabeledImageFolder(_SequenceAwareImageFolderBase, ImageFolder):
    """Plain ImageFolder that returns (image, label) pairs."""

    def __init__(
        self,
        root: str | Path,
        transform: transforms.Compose | None = None,
        sequence_aware_sampling: bool = False,
        pair_quality_cache: PairQualityCache | None = None,
        pair_mining_config: PairMiningConfig | None = None,
    ) -> None:
        super().__init__(root=str(root), transform=transform)
        self._init_sequence_metadata(root, sequence_aware_sampling=sequence_aware_sampling)
        attach_pair_quality_cache(self, pair_quality_cache, pair_mining_config)


class BalancedBatchSampler(Sampler[list[int]]):
    """Yields batches with exactly `n_classes` classes and `n_samples` per class."""

    def __init__(
        self,
        dataset: LabeledImageFolder,
        n_classes: int,
        n_samples: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples
        self._classes = list(dataset._class_to_indices.keys())
        if len(self._classes) < n_classes:
            raise ValueError(
                f"Dataset has only {len(self._classes)} classes, but n_classes={n_classes} was requested."
            )
        if n_samples < 2:
            raise ValueError("BalancedBatchSampler requires n_samples >= 2 to form positive pairs.")
        self._n_batches = max(len(dataset.targets) // self.batch_size, 1)

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self):
        for _ in range(self._n_batches):
            chosen_classes = random.sample(self._classes, self.n_classes)
            batch: list[int] = []
            for cls in chosen_classes:
                batch.extend(self.dataset._sequence_diverse_indices(cls, self.n_samples))
            yield batch


def get_loader(
    data: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 16,
    pin: bool = True,
    persistent_workers: bool = True,
    seed: int | None = None,
    batch_sampler: Sampler[list[int]] | None = None,
):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    loader_kwargs = {
        "dataset": data,
        "pin_memory": pin,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers and num_workers > 0,
        "generator": generator,
    }
    if batch_sampler is not None:
        loader_kwargs["batch_sampler"] = batch_sampler
    else:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["shuffle"] = shuffle
    return DataLoader(**loader_kwargs)
