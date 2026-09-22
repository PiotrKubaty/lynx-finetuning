"""The image path (no keypoint cache) on datasets whose frames differ in size.

CzechLynx/WildlifeReID frames are per-animal crops: every triplet and every candidate pool
mixes spatial sizes, so the collate cannot always build one tensor and RDD cannot run one
forward for the whole batch. The dataset resizes each frame to its canonical grid,
`collate_variable_images` keeps what is left as lists, and `features_from_batch` groups
those by shape — this is what makes `--trained_model rdd`/`lg+rdd` (which have no cache to
fall back on) runnable at all.
"""

import json

import pytest
import torch

from contrastive_finetuning import train_common
from contrastive_finetuning.loading import (
    IndexAssignedTripletDataset,
    PseudoAccuracyDataset,
    collate_variable_images,
)
from contrastive_finetuning.process import canonical_hw, resize_image_canonical
from contrastive_finetuning.train_common import (
    _flatten_candidates,
    _pseudo_batch_dims,
    features_from_batch,
    resize_long_side,
)


def _write_frames(root, sizes):
    """Write one PNG per (name, (H, W)) and return the relative paths."""
    from torchvision.utils import save_image

    paths = []
    for name, (height, width) in sizes.items():
        path = root / f"{name}.png"
        save_image(torch.rand(3, height, width), path)
        paths.append(path.name)
    return paths


def test_canonical_resize_matches_the_batched_rule():
    for height, width in [(1712, 1713), (1563, 1564), (400, 1200), (512, 512)]:
        image = torch.rand(3, height, width)
        resized = resize_image_canonical(image, 512)
        batched = resize_long_side(image.unsqueeze(0), 512)
        assert tuple(resized.shape[-2:]) == canonical_hw(height, width, 512)
        assert torch.allclose(resized, batched.squeeze(0))
    # the shapes that broke training: different raw sizes, same canonical grid
    assert canonical_hw(1712, 1713, 512) == canonical_hw(1563, 1564, 512) == (480, 512)


def test_collate_stacks_uniform_batches_and_keeps_mixed_ones_as_lists():
    uniform = [torch.rand(3, 64, 64) for _ in range(3)]
    assert collate_variable_images(uniform).shape == (3, 3, 64, 64)

    mixed = [torch.rand(3, 64, 64), torch.rand(3, 32, 64)]
    collated = collate_variable_images(mixed)
    assert isinstance(collated, list) and len(collated) == 2

    # dataset items: (query, positive, negative, meta) — meta goes through torch's collate
    items = [
        (torch.rand(3, 64, 64), torch.rand(3, 64, 64), torch.rand(3, 32, 64), {"neg_source": "index"}),
        (torch.rand(3, 32, 64), torch.rand(3, 64, 64), torch.rand(3, 64, 64), {"neg_source": "random"}),
    ]
    query, positive, negative, meta = collate_variable_images(items)
    assert isinstance(query, list) and positive.shape == (2, 3, 64, 64) and isinstance(negative, list)
    assert meta["neg_source"] == ["index", "random"]

    # candidate pools (lists per item): stacked when every frame agrees, nested otherwise
    pools = [[torch.rand(3, 64, 64) for _ in range(2)] for _ in range(2)]
    assert collate_variable_images(pools).shape == (2, 2, 3, 64, 64)
    pools[1][0] = torch.rand(3, 32, 64)
    nested = collate_variable_images(pools)
    assert isinstance(nested, list) and [len(p) for p in nested] == [2, 2]


def test_pseudo_accuracy_dataset_resizes_and_only_stacks_what_fits(tmp_path):
    root = tmp_path / "frames"
    root.mkdir()
    _write_frames(root, {"q": (1712, 1713), "a": (1563, 1564), "b": (400, 1200)})
    entries = [{"query_frame": "q.png", "positives": ["a.png"], "negatives": ["b.png"]}]
    from torchvision import transforms

    ds = PseudoAccuracyDataset(entries, root=root, transform=transforms.ToTensor(), resize=512)
    query, candidates, index = ds[0]
    assert index == 0
    assert tuple(query.shape[-2:]) == (480, 512)
    # a.png -> (480, 512), b.png -> (160, 512): not stackable, so a list comes back
    assert isinstance(candidates, list)
    assert [tuple(c.shape[-2:]) for c in candidates] == [(480, 512), (160, 512)]

    same = PseudoAccuracyDataset(
        [{"query_frame": "q.png", "positives": ["a.png"], "negatives": ["a.png"]}],
        root=root, transform=transforms.ToTensor(), resize=512,
    )
    assert isinstance(same[0][1], torch.Tensor) and same[0][1].shape[0] == 2


def test_triplet_dataset_keeps_canonical_shapes(tmp_path):
    # index paths carry the dataset layout <split>/<animal>/...: _lynx_id reads part[1]
    root = tmp_path / "frames"
    for animal in ("lynx_1", "lynx_2"):
        (root / "train" / animal).mkdir(parents=True)
    _write_frames(root / "train" / "lynx_1", {"q": (1712, 1713), "a": (1563, 1564)})
    _write_frames(root / "train" / "lynx_2", {"b": (400, 1200)})
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps([{
        "query_frame": "train/lynx_1/q.png",
        "positives": ["train/lynx_1/a.png"],
        "negatives": ["train/lynx_2/b.png"],
    }]))
    from torchvision import transforms

    ds = IndexAssignedTripletDataset(
        index_path, root=root, transform=transforms.ToTensor(), resize=512
    )
    query, positive, negative = ds[0]
    assert tuple(query.shape[-2:]) == (480, 512)
    assert tuple(positive.shape[-2:]) == (480, 512)
    assert tuple(negative.shape[-2:]) == (160, 512)
    assert ds._frame_shape("train/lynx_2/b.png") == (160, 512)


def test_features_from_batch_groups_mixed_shapes_and_keeps_the_order(monkeypatch):
    calls = []

    def fake_extract_train(rdd, images):
        calls.append(tuple(images.shape))
        # one keypoint per frame, carrying the frame's height so order is checkable
        return [
            {"keypoints": torch.tensor([[float(images.shape[-2]), 0.0]]),
             "descriptors": torch.zeros(1, 4)}
            for _ in range(images.shape[0])
        ]

    monkeypatch.setattr(train_common, "extract_train", fake_extract_train)
    batch = [
        torch.rand(3, 480, 512), torch.rand(3, 160, 512),
        torch.rand(3, 480, 512), torch.rand(3, 160, 512),
    ]
    feats, heights, widths = features_from_batch(batch, torch.nn.Identity(), 512, torch.device("cpu"))
    assert len(feats) == 4
    assert heights == [480, 160, 480, 160] and widths == [512, 512, 512, 512]
    assert [int(f["keypoints"][0, 0]) for f in feats] == [480, 160, 480, 160]
    assert sorted(calls) == [(2, 3, 160, 512), (2, 3, 480, 512)]  # one RDD call per shape

    # a uniform batch still takes the single stacked call
    calls.clear()
    stacked = torch.stack([torch.rand(3, 480, 512) for _ in range(3)])
    feats, height, width = features_from_batch(stacked, torch.nn.Identity(), 512, torch.device("cpu"))
    assert len(feats) == 3 and (int(height), int(width)) == (480, 512)
    assert calls == [(3, 3, 480, 512)]


def test_pseudo_helpers_accept_mixed_candidate_lists():
    pools = [[torch.rand(3, 480, 512), torch.rand(3, 160, 512)] for _ in range(3)]
    assert _pseudo_batch_dims(pools) == (3, 2)
    flat = _flatten_candidates(pools)
    assert isinstance(flat, list) and len(flat) == 6
    assert [tuple(t.shape[-2:]) for t in flat[:2]] == [(480, 512), (160, 512)]

    stacked = torch.rand(3, 2, 3, 480, 512)
    assert _pseudo_batch_dims(stacked) == (3, 2)
    assert _flatten_candidates(stacked).shape == (6, 3, 480, 512)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
