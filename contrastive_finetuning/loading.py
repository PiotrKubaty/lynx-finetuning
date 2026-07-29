from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision import transforms

class IndexAssignedTripletDataset(Dataset):
    """Triplet dataset driven by a pre-built JSON index.

    Each entry in the index describes one query frame together with a pool of
    positives and a pool of negatives.  At every call to ``__getitem__`` one
    positive and one negative are drawn at random from their respective pools.

    Expected JSON schema (list of objects)::

        [
          {
            "query_frame": "relative/path/to/query.jpg",
            "positives": ["rel/pos1.jpg", ...],
            "negatives": ["rel/neg1.jpg", ...]
          },
          ...
        ]

    Args:
        index_path: Path to the JSON index file.
        root: Root directory that is prepended to every relative path in the
            index.  If *None*, paths are used as-is (must then be absolute).
        transform: Transform applied to all three images (query, positive,
            negative).
        query_transform: Optional separate transform for the query image; falls
            back to *transform* when not provided.
        loader: Callable that loads a PIL image from a path.
        random_negative_prob: Probability of replacing the index-mined
            negative with a random image of a *different* lynx, drawn from
            the whole candidate pool (every lynx dir under the same split the
            index's own positives/negatives come from) instead of just this
            entry's top_m negatives. 0 (default) keeps the original
            index-only behavior. Requires `root` to be set, since the
            candidate pool is built by scanning the filesystem.
    """

    def __init__(
        self,
        index_path: str | Path,
        root: str | Path | None = None,
        transform: transforms.Compose | None = None,
        query_transform: transforms.Compose | None = None,
        loader=None,
        random_negative_prob: float = 0.0,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.transform = transform
        self.query_transform = query_transform or transform
        self._loader = loader or default_loader
        self.random_negative_prob = random_negative_prob

        with open(index_path) as f:
            self._entries: list[dict] = json.load(f)

        for entry in self._entries:
            if not entry.get("positives"):
                raise ValueError(f"Entry for {entry['query_frame']} has no positives.")
            if not entry.get("negatives"):
                raise ValueError(f"Entry for {entry['query_frame']} has no negatives.")

        self._lynx_pool: dict[str, list[str]] = {}
        self._lynx_ids: list[str] = []
        if random_negative_prob > 0:
            if self.root is None:
                raise ValueError("random_negative_prob > 0 requires `root` to be set")
            self._lynx_pool = self._scan_lynx_pool()
            self._lynx_ids = list(self._lynx_pool)

    def _scan_lynx_pool(self) -> dict[str, list[str]]:
        """Every image under the split that positives/negatives are drawn
        from (e.g. 'train/'), grouped by lynx id — the same split used by
        every entry's own positives/negatives, whatever the query's split.
        """
        cand_split = Path(self._entries[0]["positives"][0]).parts[0]
        cand_root = self.root / cand_split
        pool: dict[str, list[str]] = defaultdict(list)
        for lynx_dir in sorted(cand_root.iterdir()):
            if not lynx_dir.is_dir():
                continue
            for img_path in lynx_dir.rglob("*.jpg"):
                pool[lynx_dir.name].append(str(img_path.relative_to(self.root)))
        return dict(pool)

    def _lynx_id(self, rel_path: str) -> str:
        return Path(rel_path).parts[1]

    def __len__(self) -> int:
        return len(self._entries)

    def _full_path(self, rel: str) -> Path:
        return self.root / rel if self.root is not None else Path(rel)

    def _sample_negative(self, entry: dict) -> str:
        if self.random_negative_prob > 0 and random.random() < self.random_negative_prob:
            query_lynx = self._lynx_id(entry["query_frame"])
            neg_lynx = query_lynx
            while neg_lynx == query_lynx:
                neg_lynx = random.choice(self._lynx_ids)
            return random.choice(self._lynx_pool[neg_lynx])
        return random.choice(entry["negatives"])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entry = self._entries[index]

        query_path = self._full_path(entry["query_frame"])
        pos_path   = self._full_path(random.choice(entry["positives"]))
        neg_path   = self._full_path(self._sample_negative(entry))

        query_img = self._loader(query_path)
        pos_img   = self._loader(pos_path)
        neg_img   = self._loader(neg_path)

        if self.query_transform is not None:
            query_img = self.query_transform(query_img)
        if self.transform is not None:
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return query_img, pos_img, neg_img


class PseudoAccuracyDataset(Dataset):
    """Query + its FULL candidate pool, for pseudo-accuracy eval.

    Unlike IndexAssignedTripletDataset (one random positive + one random
    negative per item), pseudo-accuracy needs every positive and every
    negative for a query to find the best-scoring one. Every entry must carry
    the same number of positives and the same number of negatives (true for
    indices built with a fixed top_k/top_m) so candidates stack into a
    uniform (n_pos + n_neg, C, H, W) tensor per item and batch across queries
    via the default collate_fn — same DataLoader/num_workers path as
    training, instead of loading images one at a time in the main process.

    Args:
        entries: List of index entries (query_frame/positives/negatives dicts).
        root: Root directory prepended to every relative path in the index.
        transform: Transform applied to every candidate image.
        query_transform: Optional separate transform for the query image;
            falls back to *transform* when not provided.
        loader: Callable that loads a PIL image from a path.
    """

    def __init__(
        self,
        entries: list[dict],
        root: str | Path | None = None,
        transform: transforms.Compose | None = None,
        query_transform: transforms.Compose | None = None,
        loader=None,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.transform = transform
        self.query_transform = query_transform or transform
        self._loader = loader or default_loader
        self.entries = entries

        n_pos = {len(e["positives"]) for e in entries}
        n_neg = {len(e["negatives"]) for e in entries}
        if len(n_pos) > 1 or len(n_neg) > 1:
            raise ValueError(
                "PseudoAccuracyDataset requires every entry to have the same "
                f"number of positives/negatives; got positive counts {n_pos} "
                f"and negative counts {n_neg}"
            )
        self.n_pos = next(iter(n_pos)) if n_pos else 0
        self.n_neg = next(iter(n_neg)) if n_neg else 0

    def __len__(self) -> int:
        return len(self.entries)

    def _full_path(self, rel: str) -> Path:
        return self.root / rel if self.root is not None else Path(rel)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        entry = self.entries[index]

        query_img = self._loader(self._full_path(entry["query_frame"]))
        if self.query_transform is not None:
            query_img = self.query_transform(query_img)

        cand_paths = list(entry["positives"]) + list(entry["negatives"])
        cand_imgs = [self._loader(self._full_path(p)) for p in cand_paths]
        if self.transform is not None:
            cand_imgs = [self.transform(img) for img in cand_imgs]

        return query_img, torch.stack(cand_imgs), index


def get_loader(
    data: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 16,
    pin: bool = True,
    persistent_workers=True,
    seed: int | None = None,
):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        generator=generator,
    )
    return loader
