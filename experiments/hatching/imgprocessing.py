from pathlib import Path

import cv2
import numpy as np
import rasterio

from experiments.hatching.slope import get_slope

OUTPUT_PATH = Path("experiments/hatching/output")
ELEVATION_FILE = Path("experiments/hatching/data/GebcoToBlender/fullsize_reproject.tif")

CROP_SIZE = [15000, 15000]


def hstack(*args):
    num_rows = 0
    num_cols = 0

    for m in args:
        num_rows = max(num_rows, m.shape[0])
        num_cols += m.shape[1]

    output = np.zeros([num_rows, num_cols], dtype=np.uint8)

    col_advance = 0
    for m in args:
        output[0 : m.shape[0], col_advance : col_advance + m.shape[1]] = m
        col_advance += m.shape[1]

    return output


data = None
with rasterio.open(str(ELEVATION_FILE)) as dataset:
    data = dataset.read(1)

CROP_CENTER = [0.4, 0.4]
data = data[
    int(CROP_CENTER[1] * data.shape[1] - CROP_SIZE[1] // 2) : int(CROP_CENTER[1] * data.shape[1] + CROP_SIZE[1] // 2),
    int(CROP_CENTER[0] * data.shape[0] - CROP_SIZE[0] // 2) : int(CROP_CENTER[0] * data.shape[0] + CROP_SIZE[0] // 2),
]

for i in [60, 70, 80, 90, 100, 120, 140, 160, 180, 200]:
    data = cv2.blur(data, (i, i))
    _, _, _, _, angles, inclination = get_slope(data, 1)

    viz_inclination = normalize_to_uint8(inclination)
    angle_width = 30
    tanako_ang = cv2.inRange(np.degrees(angles), np.array([45 - angle_width / 2]), np.array([45 + angle_width / 2]))
    # tanako_inc = cv2.inRange(inclination, np.array([500]), np.array([np.max(inclination)]))
    # tanako_inc = cv2.inRange(inclination, np.array([20]), np.array([2000]))
    tanako_inc = inclination > 10

    tanako = (np.logical_and(tanako_ang, tanako_inc) * 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    tanako = cv2.morphologyEx(tanako, cv2.MORPH_OPEN, kernel)
    tanako = cv2.morphologyEx(tanako, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(
        str(Path(OUTPUT_PATH, f"tanako_base_blur_{i}.png")),
        hstack(normalize_to_uint8(inclination), normalize_to_uint8(tanako_ang), tanako),
    )
