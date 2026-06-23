from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# ── seeds ─────────────────────────────────────────────────────────────────────
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# ── config ────────────────────────────────────────────────────────────────────
DATA_ROOT = Path(
    "/shared/sets/datasets/confidential/lynx/processed_frames/segmented/dfk-June-2026-merged/lynx/train/"
)
RDD_WEIGHTS = "rdd/weights/RDD-v2.pth"
LG_WEIGHTS  = "rdd/weights/RDD_lg-v2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESIZE = 512
TOP_K = 512
BATCH_SIZE = 8
MAX_VIS_MATCHES = 10

OUT_POS = Path(__file__).parent / "positive_pairs"
OUT_NEG = Path(__file__).parent / "negative_pairs"
OUT_POS.mkdir(exist_ok=True)
OUT_NEG.mkdir(exist_ok=True)

# ── dataset & loader ──────────────────────────────────────────────────────────
from contrastive_finetuning.loading import TripletImageFolder, get_loader
from contrastive_finetuning.models import build_rdd, build_masked_lg
from contrastive_finetuning.process import FrameFeat, get_batched_data, get_matches_lightglue

transform = transforms.ToTensor()

dataset = TripletImageFolder(DATA_ROOT, transform=transform)
loader = get_loader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    persistent_workers=False,
    seed=SEED,
)

# ── models ────────────────────────────────────────────────────────────────────
rdd = build_rdd(RDD_WEIGHTS, DEVICE, TOP_K)
lg  = build_masked_lg(DEVICE, weights=LG_WEIGHTS)

# ── helpers ───────────────────────────────────────────────────────────────────
def resize_long_side(images: torch.Tensor, size: int) -> torch.Tensor:
    """Resize so the long side == size (div-by-32 aligned)."""
    _, _, H, W = images.shape
    scale = size / max(H, W)
    new_H = int(H * scale) // 32 * 32
    new_W = int(W * scale) // 32 * 32
    return F.interpolate(images.float(), (new_H, new_W), mode="bilinear", align_corners=False)


def extract_feats(images: torch.Tensor) -> list[FrameFeat]:
    """One RDD forward pass for the full batch → list of FrameFeat."""
    _, _, H, W = images.shape
    image_size = np.array([H, W], dtype=np.int32)
    with torch.no_grad():
        outs = rdd.extract(images.to(DEVICE))
    return [
        FrameFeat(
            keypoints=out["keypoints"].cpu().numpy(),
            descriptors=out["descriptors"].cpu().numpy(),
            scores=out["scores"].cpu().numpy(),
            image_size=image_size,
        )
        for out in outs
    ]


def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    rgb = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_matches(
    pts0: np.ndarray, pts1: np.ndarray, conf: np.ndarray,
    img0: np.ndarray, img1: np.ndarray,
    max_matches: int = 10, line_thickness: int = 2, circle_radius: int = 4,
) -> np.ndarray:
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:] = img1

    if len(pts0) == 0:
        return canvas

    idx = np.argsort(-conf)[:max_matches]
    pts0, pts1 = pts0[idx], pts1[idx]
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
        (0, 128, 255), (128, 255, 0),
    ]
    for k, ((x0, y0), (x1, y1)) in enumerate(zip(pts0, pts1)):
        c = palette[k % len(palette)]
        cv2.circle(canvas, (int(x0), int(y0)), circle_radius, c, -1)
        cv2.circle(canvas, (int(x1) + w0, int(y1)), circle_radius, c, -1)
        cv2.line(canvas, (int(x0), int(y0)), (int(x1) + w0, int(y1)), c, line_thickness)
    return canvas


# ── main ──────────────────────────────────────────────────────────────────────
anchors, positives, negatives = next(iter(loader))  # (B, C, H, W) float [0,1]

# Resize to uniform resolution for batch processing
H_orig, W_orig = anchors.shape[-2], anchors.shape[-1]
anchors_r   = resize_long_side(anchors,   RESIZE).to(DEVICE)
positives_r = resize_long_side(positives, RESIZE).to(DEVICE)
negatives_r = resize_long_side(negatives, RESIZE).to(DEVICE)
H_r, W_r = anchors_r.shape[-2], anchors_r.shape[-1]

# Feature extraction — one RDD forward pass per image set
feats_a = extract_feats(anchors_r)
feats_p = extract_feats(positives_r)
feats_n = extract_feats(negatives_r)

# Prepare batched LightGlue inputs using process.py utilities
data_a = get_batched_data(feats_a, DEVICE)
data_p = get_batched_data(feats_p, DEVICE)
data_n = get_batched_data(feats_n, DEVICE)

# Matching — one LightGlue forward pass for all positive pairs, one for negative
with torch.no_grad():
    pred_pos = get_matches_lightglue(lg, data_a, data_p)
    pred_neg = get_matches_lightglue(lg, data_a, data_n)

# Scale factor: keypoints are in resized coords, display on original-res images
scale = np.array([W_orig / W_r, H_orig / H_r])

# Visualise — loop is over results only, not model forward passes
for i in range(BATCH_SIZE):
    anchor_bgr = tensor_to_bgr(anchors[i])
    pos_bgr    = tensor_to_bgr(positives[i])
    neg_bgr    = tensor_to_bgr(negatives[i])

    kpts_a = feats_a[i].keypoints * scale
    kpts_p = feats_p[i].keypoints * scale
    kpts_n = feats_n[i].keypoints * scale

    def unpack(pred, kpts0, kpts1):
        m = pred["matches"][i].cpu().numpy()   # (S, 2) match indices
        s = pred["scores"][i].cpu().numpy()    # (S,)   match scores
        valid = (m[:, 0] < len(kpts0)) & (m[:, 1] < len(kpts1))
        m, s = m[valid], s[valid]
        if len(m) == 0:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
        return kpts0[m[:, 0]], kpts1[m[:, 1]], s

    pts0_p, pts1_p, scores_p = unpack(pred_pos, kpts_a, kpts_p)
    pts0_n, pts1_n, scores_n = unpack(pred_neg, kpts_a, kpts_n)

    print(f"[{i:02d}]  positive matches: {len(scores_p):4d}  |  negative matches: {len(scores_n):4d}")

    cv2.imwrite(str(OUT_POS / f"pair_{i:02d}.jpg"),
                draw_matches(pts0_p, pts1_p, scores_p, anchor_bgr, pos_bgr, MAX_VIS_MATCHES))
    cv2.imwrite(str(OUT_NEG / f"pair_{i:02d}.jpg"),
                draw_matches(pts0_n, pts1_n, scores_n, anchor_bgr, neg_bgr, MAX_VIS_MATCHES))

print(f"\nSaved to:\n  {OUT_POS}\n  {OUT_NEG}")
