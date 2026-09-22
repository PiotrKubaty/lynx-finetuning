from typing import List, Tuple

import torch
import torch.nn.functional as F


def canonical_hw(height: int, width: int, size: int) -> Tuple[int, int]:
    """Target (H, W) for an image whose long side becomes ``size``, div-by-32.

    The single definition of the resize every RDD path uses: ``resize_long_side``
    (batched, on device) and ``resize_image_canonical`` (per image, in the DataLoader
    worker) must agree, or cached and freshly detected features would not match.
    """
    scale = size / max(int(height), int(width))
    # The epsilon only repairs float error on the long side (h * size / h can come back as
    # 511.999...), which would otherwise drop a whole 32-px step and, worse, make the rule
    # non-idempotent: resizing an image that is already canonical has to be a no-op, because
    # the DataLoader resizes it once and the model path may resize it again.
    return (
        max(32, int(int(height) * scale + 1e-6) // 32 * 32),
        max(32, int(int(width) * scale + 1e-6) // 32 * 32),
    )


def resize_image_canonical(image: torch.Tensor, size: int) -> torch.Tensor:
    """Resize one CHW image to ``canonical_hw``; a no-op when it already fits."""
    height, width = image.shape[-2:]
    target = canonical_hw(height, width, size)
    if (int(height), int(width)) == target:
        return image.float()
    return F.interpolate(
        image.unsqueeze(0).float(), target, mode="bilinear", align_corners=False
    ).squeeze(0)


def pad_to_length(x: torch.Tensor, length: int) -> Tuple[torch.Tensor]:
    if length <= x.shape[-2]:
        return x, torch.ones_like(x[..., :1], dtype=torch.bool)
    pad = torch.ones(
        *x.shape[:-2], length - x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype
    )
    y = torch.cat([x, pad], dim=-2)
    mask = torch.zeros(*y.shape[:-1], 1, dtype=torch.bool, device=x.device)
    mask[..., : x.shape[-2], :] = True
    return y, mask


def align_tensors_to_max_length(ts: List[torch.Tensor]):
    B = max(len(t) for t in ts)

    pts, ms = [], []
    for t in ts:
        pt, m = pad_to_length(t, B)
        pts.append(pt)
        ms.append(m)

    pts = torch.stack(pts)
    ms = torch.stack(ms)

    return pts, ms

