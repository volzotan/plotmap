from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from experiments.hatching.slope import get_slope

INPUT_FILE = Path("shaded_relief3.png")
# INPUT_FILE =  Path("shaded_relief4.png")

OUTPUT_PATH = Path("experiments/hatching/output")

THRESHOLD = 100

ratio = 3
kernel_size = 3


def _convert_to_uint8(m: np.array) -> np.array:
    if np.min(m) < 0:
        m += abs(np.min(m))
    m *= 255.0 / np.max(m)
    return m.astype(np.uint8)


# EXPORT IMAGES FOR EXTERNAL PROCESSING

ELEVATION_FILE = Path("experiments/hatching/data/gebco_crop.tif")
SAMPLING_STEP = 1
# BLUR = [0, 3, 5, 7, 9, 11, 13, 15, 17]
BLUR = [21, 31, 41, 51]

for blur_kernel_size in BLUR:
    print(f"BLUR: {blur_kernel_size}")

    elevation = (cv2.imread(str(ELEVATION_FILE), cv2.IMREAD_UNCHANGED)).astype(np.float64)

    if blur_kernel_size > 0:
        elevation = cv2.blur(elevation, (blur_kernel_size, blur_kernel_size))

    X, Y, dX, dY, angles, inclination = get_slope(elevation, SAMPLING_STEP)

    angles_colormapped = plt.cm.hsv((angles - np.min(angles)) / np.ptp(angles)) * 255
    print(f"angles mapped {angles_colormapped.min()} {angles_colormapped.max()}")
    angles_colormapped = angles_colormapped.astype(np.uint8)[:, :, 0:3]
    cv2.imwrite(
        Path(OUTPUT_PATH, f"gebco_crop_angles_b{blur_kernel_size}.png"),
        cv2.cvtColor(angles_colormapped, cv2.COLOR_RGB2BGR),
    )

    inclination_colormapped = plt.cm.viridis((inclination - np.min(inclination)) / np.ptp(inclination)) * 255
    print(f"inclination mapped {inclination_colormapped.min()} {inclination_colormapped.max()}")
    inclination_colormapped = inclination_colormapped.astype(np.uint8)[:, :, 0:3]
    cv2.imwrite(
        Path(OUTPUT_PATH, f"gebco_crop_inclination_b{blur_kernel_size}.png"),
        cv2.cvtColor(inclination_colormapped, cv2.COLOR_RGB2BGR),
    )

    elevation_colormapped = plt.cm.viridis((elevation - np.min(elevation)) / np.ptp(elevation)) * 255
    print(f"elevation mapped {elevation_colormapped.min()} {elevation_colormapped.max()}")
    elevation_colormapped = elevation_colormapped.astype(np.uint8)[:, :, 0:3]
    cv2.imwrite(
        Path(OUTPUT_PATH, f"gebco_crop_elevation_b{blur_kernel_size}.png"),
        cv2.cvtColor(elevation_colormapped, cv2.COLOR_RGB2BGR),
    )


exit()

# src = cv2.imread(INPUT_FILE)
# src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# img_blur = cv2.blur(src_gray, (3, 3))
# detected_edges = cv2.Canny(img_blur, THRESHOLD, THRESHOLD*ratio, kernel_size)
# mask = detected_edges != 0
# dst = src * (mask[:, :, None].astype(src.dtype))
#
# overlay = src
# overlay[mask > 0, 2] = 255
#
# cv2.imwrite(Path(OUTPUT_PATH, INPUT_FILE.stem + "_edge.png"), overlay)
# cv2.imwrite(Path(OUTPUT_PATH, INPUT_FILE.stem + "_edge_maskonly.png"), detected_edges)


# ELEVATION_FILE = Path("experiments/hatching/data/GebcoToBlender/reproject.tif")
#
# elevation = (cv2.imread(str(ELEVATION_FILE), cv2.IMREAD_UNCHANGED)).astype(np.float64)
# # data = cv2.resize(data, [2000, 2000])
# # elevation[elevation > 0] = 0
#
# X, Y, dX, dY, angles, inclination = get_slope(elevation, 1)
#
# data = np.degrees(angles)
# # data = inclination
# # data = np.degrees(angles) * inclination
#
# anginc = np.degrees(angles) * inclination
# logger.debug(f"anginc min: {np.min(anginc)} | max: {np.max(anginc)}")
#
# data = data.astype(np.float64)
#
# logger.debug(f"data {ELEVATION_FILE} min: {np.min(data)} | max: {np.max(data)}")
#
# if np.min(data) < 0:
#     data += abs(np.min(data))
# data *= 255.0 / np.max(data)
#
# data = data.astype(np.uint8)
# data = cv2.blur(data, (5, 5))
# detected_edges = cv2.Canny(data, THRESHOLD, THRESHOLD * ratio, kernel_size)


# ELEVATION_FILE = Path("experiments/hatching/data/GebcoToBlender/reproject.tif")
ELEVATION_FILE = Path("experiments/hatching/data/gebco_crop.tif")
CROP_SIZE = [5000, 5000]
SAMPLING_STEP = 1

elevation = (cv2.imread(str(ELEVATION_FILE), cv2.IMREAD_UNCHANGED)).astype(np.float64)

elevation[elevation > -500] = -500

TARGET_RESOLUTION = [elevation.shape[1], elevation.shape[0]]

elevation = elevation[
    TARGET_RESOLUTION[1] // 2 - CROP_SIZE[1] // 2 : TARGET_RESOLUTION[1] // 2 + CROP_SIZE[1] // 2,
    TARGET_RESOLUTION[0] // 2 - CROP_SIZE[0] // 2 : TARGET_RESOLUTION[0] // 2 + CROP_SIZE[0] // 2,
]

X, Y, dX, dY, angles, inclination = get_slope(elevation, SAMPLING_STEP)

# angles_deriv2 = cv2.convertScaleAbs(cv2.Laplacian(angles, cv2.CV_64F))
angles_deriv2 = _convert_to_uint8(np.abs(cv2.Laplacian(angles, cv2.CV_64F)))

sobelx = cv2.Sobel(angles, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(angles, cv2.CV_64F, 0, 1, ksize=5)
angles_deriv1 = np.maximum(cv2.convertScaleAbs(sobelx), cv2.convertScaleAbs(sobely))

inclination_deriv2 = _convert_to_uint8(np.abs(cv2.Laplacian(inclination, cv2.CV_64F)))

ang_inc = angles_deriv2 * inclination
logger.debug(f"anginc min: {np.min(ang_inc)} | max: {np.max(ang_inc)}")
ang_inc = _convert_to_uint8(ang_inc)

_, angle_inc_thres = cv2.threshold(ang_inc, 10, 255, cv2.THRESH_BINARY)

fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4)
plt.tight_layout(pad=0.01)

ax1.imshow(cv2.blur(angles, (5, 5)), cmap="hsv")
ax1.set_title("angles")

# ksize = 7
# angles_deriv2 = _convert_to_uint8(np.abs(cv2.Laplacian(cv2.blur(angles, (ksize, ksize)), cv2.CV_64F)))
# ax2.imshow(angles_deriv2)
# ax2.set_title(f"angles 2nd deriv / blur {ksize}")
#
# ksize = 17
# angles_deriv2 = _convert_to_uint8(np.abs(cv2.Laplacian(cv2.blur(angles, (ksize, ksize)), cv2.CV_64F)))
# ax3.imshow(angles_deriv2)
# ax3.set_title(f"angles 2nd deriv / blur {ksize}")
#
# ksize = 23
# angles_deriv2 = _convert_to_uint8(np.abs(cv2.Laplacian(cv2.blur(angles, (ksize, ksize)), cv2.CV_64F)))
# ax4.imshow(angles_deriv2)
# ax4.set_title(f"angles 2nd deriv / blur {ksize}")


ksize = 3
rsize = 4000
angles_deriv2 = _convert_to_uint8(
    np.abs(cv2.Laplacian(cv2.blur(cv2.resize(angles, [rsize, rsize]), (ksize, ksize)), cv2.CV_64F))
)
ax2.imshow(angles_deriv2)
ax2.set_title(f"angles 2nd deriv / rsize {rsize}")

rsize = 3000
angles_deriv2 = _convert_to_uint8(
    np.abs(cv2.Laplacian(cv2.blur(cv2.resize(angles, [rsize, rsize]), (ksize, ksize)), cv2.CV_64F))
)
ax3.imshow(angles_deriv2)
ax3.set_title(f"angles 2nd deriv / rsize {rsize}")

rsize = 2000
angles_deriv2 = _convert_to_uint8(
    np.abs(cv2.Laplacian(cv2.blur(cv2.resize(angles, [rsize, rsize]), (ksize, ksize)), cv2.CV_64F))
)
ax4.imshow(angles_deriv2)
ax4.set_title(f"angles 2nd deriv / rsize {rsize}")

# ax5.imshow(~(angles_deriv2 * _convert_to_uint8(inclination)))
# ax5.set_title("angles * rev inclination")

ax6.imshow(angle_inc_thres)
ax6.set_title("angle_inc_thres")

# ax7.imshow(np.clip(elevation_deriv2, 0, np.max(elevation_deriv2) * 0.25))
# ax7.set_title("elevation_deriv2")

inclination_clipped = np.clip(inclination, 0, np.max(inclination) * 0.5)
ax7.imshow(inclination_clipped)
ax7.set_title("inclination (clipped)")

ax8.imshow(np.clip(inclination_deriv2, 0, np.max(inclination_deriv2) * 0.25))
ax8.set_title("inclination_deriv2")

# ax9.imshow(cv2.Canny(np.clip(angle_inc_thres, 0, np.max(angle_inc_thres) * 0.5).astype(np.uint8), 100, 200))
# ax9.set_title("edges: angle_inc_thres")
#
# ax10.imshow(cv2.Canny(np.clip(elevation_deriv2, 0, np.max(elevation_deriv2) * 0.5).astype(np.uint8), 100, 200))
# ax10.set_title("edges: elevation_deriv2")
#
# ax11.imshow(cv2.Canny(np.clip(inclination_deriv2, 0, np.max(inclination_deriv2) * 0.5).astype(np.uint8), 20, 150))
# ax11.set_title("edges: inclination_deriv2")


# gap = 10
#
# a = 40
#
# slice = _convert_to_uint8(angles.copy())
# slice[slice < a-gap//2] = 0
# slice[slice > a+gap//2] = 0
# ax9.imshow(slice)
# ax9.set_title(f"angle slice {a}")
#
# a = 45
#
# slice = _convert_to_uint8(angles.copy())
# slice[slice < a-gap//2] = 0
# slice[slice > a+gap//2] = 0
# ax10.imshow(slice)
# ax10.set_title(f"angle slice {a}")
#
# a = 50
#
# slice = _convert_to_uint8(angles.copy())
# slice[slice < a-gap//2] = 0
# slice[slice > a+gap//2] = 0
# ax11.imshow(slice)
# ax11.set_title(f"angle slice {a}")
#
# a = 55
#
# slice = _convert_to_uint8(angles.copy())
# slice[slice < a-gap//2] = 0
# slice[slice > a+gap//2] = 0
# ax12.imshow(slice)
# ax12.set_title(f"angle slice {a}")


ax9.imshow(elevation)
ax9.set_title("elevation")

elevation_deriv2 = _convert_to_uint8(np.abs(cv2.Laplacian(elevation, cv2.CV_64F)))
ax10.imshow(elevation)
ax10.set_title("elevation 2nd deriv")

elevation_deriv2 = _convert_to_uint8(np.abs(cv2.Laplacian(cv2.blur(elevation, (21, 21)), cv2.CV_64F)))
ax10.imshow(elevation_deriv2)
ax10.set_title("elevation 2nd deriv / blur 21")

elevation_deriv2 = _convert_to_uint8(np.abs(cv2.Laplacian(cv2.blur(elevation, (23, 23)), cv2.CV_64F)))
ax11.imshow(elevation_deriv2)
ax11.set_title("elevation 2nd deriv / blur 23")

elevation_deriv2 = _convert_to_uint8(np.abs(cv2.Laplacian(cv2.blur(elevation, (25, 25)), cv2.CV_64F)))
ax12.imshow(elevation_deriv2)
ax12.set_title("elevation 2nd deriv / blur 25")

fig.set_figheight(3 * 10)
fig.set_figwidth(4 * 10)
ax2.get_yaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
ax5.get_yaxis().set_visible(False)
ax6.get_yaxis().set_visible(False)
ax7.get_yaxis().set_visible(False)
ax8.get_yaxis().set_visible(False)
ax9.get_yaxis().set_visible(False)
ax10.get_yaxis().set_visible(False)
ax11.get_yaxis().set_visible(False)
ax12.get_yaxis().set_visible(False)

plt.savefig(Path(OUTPUT_PATH, "edge_overview.png"))

# cv2.imwrite(Path(OUTPUT_PATH, ELEVATION_FILE.stem + "_data.png"), data)
# cv2.imwrite(Path(OUTPUT_PATH, ELEVATION_FILE.stem + "_edge_maskonly.png"), detected_edges)

# cv2.imwrite(Path(OUTPUT_PATH, ELEVATION_FILE.stem + "_angles_deriv1.png"), angles_deriv1)
# cv2.imwrite(Path(OUTPUT_PATH, ELEVATION_FILE.stem + "_angles_deriv2.png"), angles_deriv2)
# cv2.imwrite(Path(OUTPUT_PATH, ELEVATION_FILE.stem + "_angles_inc_thres.png"), angle_inc_thres)
# cv2.imwrite(Path(OUTPUT_PATH, ELEVATION_FILE.stem + "_angles_inc.png"), ang_inc)
# cv2.imwrite(Path(OUTPUT_PATH, ELEVATION_FILE.stem + "_inclination_clipped.png"), _convert_to_uint8(np.copy(inclination_clipped)))
