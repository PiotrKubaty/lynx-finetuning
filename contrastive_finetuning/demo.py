from rdd.RDD.RDD import build
from rdd.RDD.RDD_helper import RDD_helper
from matplotlib import pyplot as plt
from time import time
import numpy as np
import cv2

RDD_model = build(weights='./rdd/weights/RDD-v2.pth')
RDD_model.eval()
RDD = RDD_helper(RDD_model)

img0_path = '/shared/sets/datasets/confidential/lynx/processed_frames/segmented/dfk-clean-Feb03/test/lynx_20/Staw_101a/0039/frame_0000.jpg'
img1_path = '/shared/sets/datasets/confidential/lynx/processed_frames/segmented/dfk-clean-Feb03/train/lynx_20/ZaFigurki_20b/0004/frame_0022.jpg'

im0 = cv2.imread(img0_path)
im1 = cv2.imread(img1_path)

import cv2
import numpy as np
import matplotlib.pyplot as plt

colors = [
    (255,0,0),(0,255,0),(0,0,255),
    (255,255,0),(255,0,255),(0,255,255),
    (128,0,255),(255,128,0),(0,128,255),(128,255,0)
]



def draw_matches(ref_points, dst_points, conf, img0, img1,
                 line_thickness=2,
                 circle_radius=4):

    ref_points = np.asarray(ref_points)
    dst_points = np.asarray(dst_points)
    conf = np.asarray(conf)

    cmap = plt.cm.viridis
    # colors = cmap(conf)[:, :3]
    # colors = (colors[:, ::-1] * 255).astype(np.uint8)

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    H = max(h0, h1)

    out = np.zeros((H, w0 + w1, 3), dtype=np.uint8)
    out[:h0, :w0] = img0
    out[:h1, w0:w0 + w1] = img1

    for i, ((x0, y0), (x1, y1)) in enumerate(zip(ref_points, dst_points)):
        color = tuple(int(c) for c in colors[i])
        t = line_thickness

        pt0 = (int(x0), int(y0))
        pt1 = (int(x1) + w0, int(y1))

        cv2.circle(out, pt0, circle_radius, color, -1)
        cv2.circle(out, pt1, circle_radius, color, -1)
        cv2.line(out, pt0, pt1, color, t)

    return out

import numpy as np

def select_top_matches_per_patch(
    mkpts_0,
    mkpts_1,
    conf,
    img0,
    patch_size=64,
    max_matches=10
):
    mkpts_0 = np.asarray(mkpts_0)
    mkpts_1 = np.asarray(mkpts_1)
    conf = np.asarray(conf)

    # sortujemy malejąco po confidence
    order = np.argsort(-conf)

    h0, w0 = img0.shape[:2]
    n_patches_x = int(np.ceil(w0 / patch_size))
    n_patches_y = int(np.ceil(h0 / patch_size))

    used_patches = set()
    selected_indices = []

    for idx in order:
        x0, y0 = mkpts_0[idx]

        px = int(x0 // patch_size)
        py = int(y0 // patch_size)

        # zabezpieczenie brzegowe
        if px < 0 or py < 0 or px >= n_patches_x or py >= n_patches_y:
            continue

        patch_id = (px, py)

        if patch_id not in used_patches:
            used_patches.add(patch_id)
            selected_indices.append(idx)

        if len(selected_indices) == max_matches:
            break

    return (
        mkpts_0[selected_indices],
        mkpts_1[selected_indices],
        conf[selected_indices],
    )
    
    
resize, top_k = 512, 512

start = time()
mkpts_0, mkpts_1, conf = RDD.match_lg(im0, im1, resize=resize, top_k=top_k)

idx = np.argsort(conf)[::-1]

conf = conf[idx]
mkpts_0 = mkpts_0[idx]
mkpts_1 = mkpts_1[idx]

warp_pts_0 = mkpts_0[:100]
warp_pts_1 = mkpts_1[:100]

patch_size = 64

mk0_sel, mk1_sel, conf_sel = select_top_matches_per_patch(
    mkpts_0,
    mkpts_1,
    conf,
    im0,
    patch_size=patch_size,
    max_matches=10
)

canvas_matches = draw_matches(
    mk0_sel,
    mk1_sel,
    conf_sel,
    im0,
    im1,
    line_thickness=10
)


plt.imshow(canvas_matches[..., ::-1])

plt.savefig("x.png")

