from __future__ import annotations

import random
from pathlib import Path


import cv2
import numpy as np
import torch
from torchvision import transforms

# ── seeds ────────────────────────────────────────────────────────────────────
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# ── config ───────────────────────────────────────────────────────────────────
DATA_ROOT = Path(
    "/shared/sets/datasets/confidential/lynx/processed_frames/segmented/dfk-June-2026-merged/lynx/train/"
)
RDD_WEIGHTS = "rdd/weights/RDD-v2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESIZE = 512
TOP_K = 512
BATCH_SIZE = 8
MAX_VIS_MATCHES = 10

OUT_POS = Path(__file__).parent / "positive_pairs_old"
OUT_NEG = Path(__file__).parent / "negative_pairs_old"
OUT_POS.mkdir(exist_ok=True)
OUT_NEG.mkdir(exist_ok=True)

# ── dataset & loader ─────────────────────────────────────────────────────────
from finetuning.loading import TripletImageFolder, get_loader
from finetuning.models import build_rdd
from rdd.RDD.RDD_helper import RDD_helper

# ToTensor needed so DataLoader can collate into a single batch tensor.
# Add transforms.Resize((H, W)) here if images have varying sizes.
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

# ── models ───────────────────────────────────────────────────────────────────
rdd_model = build_rdd(RDD_WEIGHTS, DEVICE, TOP_K)
rdd_helper = RDD_helper(rdd_model)

# ── helpers ───────────────────────────────────────────────────────────────────
def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    """CHW float [0,1] → HWC uint8 BGR."""
    rgb = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_matches(
    pts0: np.ndarray,
    pts1: np.ndarray,
    conf: np.ndarray,
    img0: np.ndarray,
    img1: np.ndarray,
    max_matches: int = 10,
    line_thickness: int = 2,
    circle_radius: int = 4,
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
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 255), (255, 128, 0), (0, 128, 255), (128, 255, 0),
    ]
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(pts0, pts1)):
        color = palette[i % len(palette)]
        cv2.circle(canvas, (int(x0), int(y0)), circle_radius, color, -1)
        cv2.circle(canvas, (int(x1) + w0, int(y1)), circle_radius, color, -1)
        cv2.line(canvas, (int(x0), int(y0)), (int(x1) + w0, int(y1)), color, line_thickness)
    return canvas


# ── main ─────────────────────────────────────────────────────────────────────
anchors, positives, negatives = next(iter(loader))  # (B, C, H, W)

for i in range(BATCH_SIZE):
    anchor_bgr = tensor_to_bgr(anchors[i])
    pos_bgr    = tensor_to_bgr(positives[i])
    neg_bgr    = tensor_to_bgr(negatives[i])

    mk0_p, mk1_p, conf_p = rdd_helper.match_lg(anchor_bgr, pos_bgr, resize=RESIZE, top_k=TOP_K)
    mk0_n, mk1_n, conf_n = rdd_helper.match_lg(anchor_bgr, neg_bgr, resize=RESIZE, top_k=TOP_K)

    canvas_pos = draw_matches(mk0_p, mk1_p, conf_p, anchor_bgr, pos_bgr, MAX_VIS_MATCHES)
    canvas_neg = draw_matches(mk0_n, mk1_n, conf_n, anchor_bgr, neg_bgr, MAX_VIS_MATCHES)

    print(f"[{i:02d}]  positive matches: {len(conf_p):4d}  |  negative matches: {len(conf_n):4d}")

    cv2.imwrite(str(OUT_POS / f"pair_{i:02d}.jpg"), canvas_pos)
    cv2.imwrite(str(OUT_NEG / f"pair_{i:02d}.jpg"), canvas_neg)

print(f"\nSaved to:\n  {OUT_POS}\n  {OUT_NEG}")
