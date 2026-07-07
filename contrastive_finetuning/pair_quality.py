from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

CACHE_VERSION = 1
QUALITY_BANDS = ("high", "medium", "low")


@dataclass
class PairMiningConfig:
    use_pair_quality_mining: bool = False
    pair_quality_cache_dir: Path | None = None
    positive_quality_mode: str = "bucketed"
    positive_high_ratio: float = 0.7
    positive_medium_ratio: float = 0.3
    exclude_low_quality_positives: bool = True
    use_hard_negative_cache: bool = True
    hard_negative_ratio: float = 0.5
    max_positive_candidates_per_anchor: int = 32
    max_negative_candidates_per_anchor: int = 32


def normalize_path_str(path: str | Path) -> str:
    return str(Path(path).resolve())


def compute_composite_score(
    normalized_sum_conf: float,
    anchor_participation: float,
    mutual_nn_rate: float,
) -> float:
    return (
        0.5 * float(normalized_sum_conf)
        + 0.3 * float(anchor_participation)
        + 0.2 * float(mutual_nn_rate)
    )


def candidate_quality_sort_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(record.get("composite_score", 0.0)),
        float(record.get("normalized_sum_conf", 0.0)),
        float(record.get("anchor_participation", 0.0)),
        float(record.get("mutual_nn_rate", 0.0)),
    )


def assign_quality_bands(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=candidate_quality_sort_key, reverse=True)
    count = len(sorted_candidates)
    high_cut = max(1, int(round(count * 0.25)))
    medium_cut = max(high_cut + 1, int(round(count * 0.60))) if count >= 3 else high_cut
    medium_cut = min(medium_cut, count)

    for idx, record in enumerate(sorted_candidates):
        if idx < high_cut:
            record["quality_band"] = "high"
        elif idx < medium_cut:
            record["quality_band"] = "medium"
        else:
            record["quality_band"] = "low"

        if record.get("match_count", 0) == 0 or float(record.get("anchor_participation", 0.0)) < 0.05:
            record["quality_band"] = "low"

    nonzero_indices = [
        i for i, rec in enumerate(sorted_candidates) if float(rec.get("composite_score", 0.0)) > 0.0
    ]
    if nonzero_indices and all(rec["quality_band"] != "high" for rec in sorted_candidates):
        sorted_candidates[nonzero_indices[0]]["quality_band"] = "high"
    if count >= 3 and all(rec["quality_band"] != "medium" for rec in sorted_candidates[1:]):
        medium_idx = min(1, count - 1)
        if sorted_candidates[medium_idx]["quality_band"] == "low":
            sorted_candidates[medium_idx]["quality_band"] = "medium"

    return sorted_candidates


def compute_mutual_nn_rate(desc0: np.ndarray, desc1: np.ndarray) -> float:
    if desc0.shape[0] == 0 or desc1.shape[0] == 0:
        return 0.0
    sim = desc0 @ desc1.T
    row_best = sim.argmax(axis=1)
    col_best = sim.argmax(axis=0)
    row_ids = np.arange(sim.shape[0])
    mutual = row_ids == col_best[row_best]
    return float(mutual.mean())


def validate_cache_metadata(
    metadata: dict[str, Any],
    *,
    data_root: Path,
    rdd_weights: str | Path,
    lg_weights: str | Path,
    resize: int,
    top_k: int,
) -> None:
    if int(metadata.get("cache_version", -1)) != CACHE_VERSION:
        raise ValueError(
            f"Unsupported pair-quality cache version {metadata.get('cache_version')} "
            f"(expected {CACHE_VERSION}). Rebuild the cache."
        )

    expected = {
        "data_root": normalize_path_str(data_root),
        "rdd_weights": normalize_path_str(rdd_weights),
        "lg_weights": normalize_path_str(lg_weights),
        "resize": int(resize),
        "top_k": int(top_k),
    }
    mismatches: list[str] = []
    for key, value in expected.items():
        cached = metadata.get(key)
        if key.endswith("_weights"):
            if normalize_path_str(cached) != value:
                mismatches.append(f"{key}: cache={cached!r} run={value!r}")
        elif cached != value:
            mismatches.append(f"{key}: cache={cached!r} run={value!r}")

    if mismatches:
        raise ValueError(
            "Pair-quality cache metadata does not match current run settings:\n  "
            + "\n  ".join(mismatches)
        )


def load_pair_quality_cache(cache_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_root = Path(cache_dir)
    metadata_path = cache_root / "metadata.json"
    pairs_path = cache_root / "pairs.pt"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing pair-quality metadata: {metadata_path}")
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Missing pair-quality pair cache: {pairs_path}")

    metadata = json.loads(metadata_path.read_text())
    pairs = torch.load(pairs_path, map_location="cpu", weights_only=False)
    return metadata, pairs


class PairQualityCache:
    """In-memory wrapper around a loaded pair-quality cache."""

    def __init__(self, metadata: dict[str, Any], pairs: dict[str, Any]) -> None:
        self.metadata = metadata
        self.pairs = pairs
        self.positive_candidates_by_anchor: dict[int, list[dict[str, Any]]] = pairs[
            "positive_candidates_by_anchor"
        ]
        self.negative_candidates_by_anchor: dict[int, list[dict[str, Any]]] = pairs[
            "negative_candidates_by_anchor"
        ]

    @classmethod
    def load(cls, cache_dir: str | Path) -> PairQualityCache:
        metadata, pairs = load_pair_quality_cache(cache_dir)
        return cls(metadata, pairs)

    def _filter_candidates(
        self,
        candidates: list[dict[str, Any]],
        *,
        exclude_low: bool,
        bands: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        filtered = candidates
        if bands is not None:
            filtered = [c for c in filtered if c.get("quality_band") in bands]
        if exclude_low:
            filtered = [c for c in filtered if c.get("quality_band") != "low"]
        return filtered

    def _pick_from_candidates(
        self,
        candidates: list[dict[str, Any]],
        rng: random.Random,
        config: PairMiningConfig,
    ) -> dict[str, Any] | None:
        if not candidates:
            return None

        mode = config.positive_quality_mode
        exclude_low = config.exclude_low_quality_positives

        if mode == "random":
            pool = self._filter_candidates(candidates, exclude_low=exclude_low)
            return rng.choice(pool) if pool else None

        high = self._filter_candidates(candidates, exclude_low=False, bands=("high",))
        medium = self._filter_candidates(candidates, exclude_low=False, bands=("medium",))
        low = self._filter_candidates(candidates, exclude_low=False, bands=("low",))

        if mode == "ranked":
            pool = high or medium or (low if not exclude_low else [])
            return pool[0] if pool else None

        # bucketed
        roll = rng.random()
        if roll < config.positive_high_ratio and high:
            return rng.choice(high)
        if roll < config.positive_high_ratio + config.positive_medium_ratio and medium:
            return rng.choice(medium)
        if not exclude_low and low:
            return rng.choice(low)
        pool = high or medium
        return rng.choice(pool) if pool else None

    def _structural_positive_candidates(
        self,
        anchor_idx: int,
        dataset_base: Any,
        *,
        tier_name: str,
        bands: tuple[str, ...] | None = None,
        exclude_low: bool = False,
        used_indices: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        anchor_meta = dataset_base._sample_meta[anchor_idx]
        all_candidates = self.positive_candidates_by_anchor.get(anchor_idx, [])
        used_indices = used_indices or set()

        def match_band(c: dict[str, Any]) -> bool:
            candidate_idx = int(c.get("candidate_index", -1))
            if candidate_idx == anchor_idx or candidate_idx in used_indices:
                return False
            if bands is not None and c.get("quality_band") not in bands:
                return False
            if exclude_low and c.get("quality_band") == "low":
                return False
            if tier_name == "cross_source":
                return c.get("source_id") != anchor_meta.source_id
            if tier_name == "cross_sequence":
                return c.get("source_id") == anchor_meta.source_id and c.get("sequence_id") != anchor_meta.sequence_id
            if tier_name == "same_sequence":
                return c.get("source_id") == anchor_meta.source_id and c.get("sequence_id") == anchor_meta.sequence_id
            raise ValueError(f"unknown positive tier {tier_name!r}")

        return [c for c in all_candidates if match_band(c)]

    def sample_positive_index(
        self,
        anchor_idx: int,
        dataset_base: Any,
        rng: random.Random,
        config: PairMiningConfig,
    ) -> int | None:
        exclude_low = config.exclude_low_quality_positives
        ordered_groups = [
            ("cross_source", ("high",)),
            ("cross_sequence", ("high",)),
            ("cross_source", ("medium",)),
            ("cross_sequence", ("medium",)),
            ("same_sequence", ("high", "medium")),
        ]
        if not exclude_low:
            ordered_groups.extend(
                [
                    ("cross_source", ("low",)),
                    ("cross_sequence", ("low",)),
                    ("same_sequence", ("low",)),
                ]
            )

        for tier_name, bands in ordered_groups:
            tier_candidates = self._structural_positive_candidates(
                anchor_idx,
                dataset_base,
                tier_name=tier_name,
                bands=bands,
                exclude_low=exclude_low,
            )
            picked = self._pick_from_candidates(tier_candidates, rng, config)
            if picked is not None:
                return int(picked["candidate_index"])
        return None

    def sample_negative_index(
        self,
        anchor_idx: int,
        label: int,
        dataset_base: Any,
        rng: random.Random,
        config: PairMiningConfig,
    ) -> int | None:
        use_hard = config.use_hard_negative_cache and rng.random() < config.hard_negative_ratio
        if use_hard:
            hard_candidates = self.negative_candidates_by_anchor.get(anchor_idx, [])
            hard_candidates = [c for c in hard_candidates if not c.get("same_identity", True)]
            if hard_candidates:
                picked = hard_candidates[0] if config.positive_quality_mode == "ranked" else rng.choice(
                    hard_candidates[: max(1, min(8, len(hard_candidates)))]
                )
                return int(picked["candidate_index"])
        return None

    def sample_balanced_group_indices(
        self,
        seed_idx: int,
        label: int,
        n_samples: int,
        dataset_base: Any,
        rng: random.Random,
        config: PairMiningConfig,
    ) -> list[int] | None:
        if n_samples <= 1:
            return [seed_idx]

        selected = [seed_idx]
        used = {seed_idx}
        exclude_low = config.exclude_low_quality_positives
        ordered_groups = [
            ("cross_source", ("high",)),
            ("cross_sequence", ("high",)),
            ("cross_source", ("medium",)),
            ("cross_sequence", ("medium",)),
            ("same_sequence", ("high", "medium")),
        ]
        if not exclude_low:
            ordered_groups.extend(
                [
                    ("cross_source", ("low",)),
                    ("cross_sequence", ("low",)),
                    ("same_sequence", ("low",)),
                ]
            )

        for tier_name, bands in ordered_groups:
            candidates = self._structural_positive_candidates(
                seed_idx,
                dataset_base,
                tier_name=tier_name,
                bands=bands,
                exclude_low=exclude_low,
                used_indices=used,
            )
            if not candidates:
                continue
            if config.positive_quality_mode == "random":
                rng.shuffle(candidates)
            elif config.positive_quality_mode == "bucketed":
                high = [c for c in candidates if c.get("quality_band") == "high"]
                medium = [c for c in candidates if c.get("quality_band") == "medium"]
                low = [c for c in candidates if c.get("quality_band") == "low"]
                rng.shuffle(high)
                rng.shuffle(medium)
                rng.shuffle(low)
                candidates = high + medium + low
            while candidates and len(selected) < n_samples:
                picked = candidates.pop(0)
                picked_idx = int(picked["candidate_index"])
                if picked_idx in used:
                    continue
                selected.append(picked_idx)
                used.add(picked_idx)
            if len(selected) >= n_samples:
                return selected[:n_samples]

        return None
