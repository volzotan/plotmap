from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import numpy as np
from shapely import LineString, Polygon, MultiPolygon
from lineworld.core.hatching import HatchingOptions, HatchingDirection, create_hatching
from lineworld.core.maptools import DocumentInfo
from lineworld.core.svgwriter import SvgWriter
from lineworld.util.gebco_grid_to_polygon import _extract_polygons, get_elevation_bounds
from lineworld.util.geometrytools import unpack_multipolygon

def _read_data(input_path: Path) -> np.ndarray:
    data = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    # data = cv2.resize(img, [30, 30])

    # data = np.flipud(data)
    # data = (data * 120/20).astype(np.int8)
    # data = np.rot90(data)

    return data

def _unlog(x, n: float = 10) -> float:
    return ((n+1)*x) / ((n*x)+1)


def get_slope(data: np.ndarray, sampling_step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes angle (in rad) and magnitude of the given 2D array of values
    """
    test_slice = data[::sampling_step, ::sampling_step]
    r, c = np.shape(data)
    Y, X = np.mgrid[0:r:sampling_step, 0:c:sampling_step]
    dY, dX = np.gradient(test_slice)  # order! Y X

    angles = np.arctan2(dY, dX)
    magnitude = np.hypot(dY, dX)

    if sampling_step > 1:
        angles = cv2.resize(angles, data.shape)
        magnitude = cv2.resize(magnitude, data.shape)

    return (X, Y, dX, dY, angles, magnitude)



# INPUT_FILE = Path("data/hatching_dem.tif")
INPUT_FILE = Path("data/gebco_crop.tif")
# INPUT_FILE = Path("data/slope_test_2.tif")
# INPUT_FILE = Path("data/slope_test_4.tif")

OUTPUT_PATH = Path("output")

SAMPLING_STEP = 5


if __name__ == "__main__":

    data = _read_data(INPUT_FILE)

    print(f"data {INPUT_FILE} min: {np.min(data)} / max: {np.max(data)}")

    X, Y, dX, dY, angles, inclination = get_slope(data, SAMPLING_STEP)

    fig = plt.figure(figsize=[20, 20])
    ax = fig.subplots()
    ax.imshow(data)
    ax.quiver(X, Y, dX, dY, angles="xy", color='r')
    plt.savefig(Path(OUTPUT_PATH, "slope_arrow.png"))

    extent = max([abs(np.min(data)), abs(np.max(data))])

    slope_img = (_unlog((np.abs(dX) + np.abs(dY)) / extent, n=10) * 255).astype(np.uint8)
    # slope_img = (_unlog(np.abs(dX) / (extent/2), n=10) * 255).astype(np.uint8)
    slope_img = cv2.resize(slope_img, [1000, 1000], interpolation=cv2.INTER_NEAREST)
    # slope_img[data > 0] = 255
    cv2.imwrite(str(Path(OUTPUT_PATH, "slope.png")), slope_img)

    # angles = angles + 1
    # angles[lengths == 0] = 0

    angles = angles / np.linalg.norm(angles)
    inclination = inclination / np.linalg.norm(inclination)
    elevation = data / np.linalg.norm(data)

    comb = np.hstack([angles, inclination, elevation])

    fig = plt.figure(figsize=[20, 20])
    ax = fig.subplots()
    plt.imshow(comb, interpolation='none')
    plt.savefig(Path(OUTPUT_PATH, "slope2.png"))


    # Three Dimensions:
    # 1: angle
    # 2: steepness / inclination / slope
    # 3: depth / elevation






    # slope_img = np.zeros([1000, 1000, 3], dtype=np.uint8)
    # slope_img[:, :, 0] = cv2.resize(np.abs(dX) * 255/extent, [1000, 1000], interpolation=cv2.INTER_NEAREST)
    # slope_img[:, :, 1] = cv2.resize(np.abs(dY) * 255/extent, [1000, 1000], interpolation=cv2.INTER_NEAREST)
    # cv2.imwrite(str(OUTPUT_PNG), slope_img)


    # slope_img = np.zeros([1000, 1000], dtype=np.uint8)
    # slope_img[:, :] = 127
    # slope_img[:, :] += cv2.resize((dX+dY) * 255/extent/2, [1000, 1000], interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    # cv2.imwrite(str(OUTPUT_PNG), slope_img)






    # from matplotlib import cm
    # from matplotlib.colors import LightSource
    #
    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # ax.set_zlim(-20000, 20000)
    # # ax.set_zlim(-0, 50)
    #
    # Z = data[::SAMPLING_STEP, ::SAMPLING_STEP]
    #
    # slope_data = np.abs(dX) + np.abs(dY)
    #
    # thres = np.max(slope_data)*0.5
    # slope_data[slope_data > thres] = thres
    #
    # ls = LightSource(270, 45)
    #
    # rgb = ls.shade(slope_data, cmap=cm.cool, vert_exag=0.1, blend_mode='soft')
    # # rgb = ls.shade(Z, cmap=cm.bwr, vert_exag=0.1, blend_mode='soft')
    #
    # surf = ax.plot_surface(X, Y, Z, rcount=100, ccount=100, linewidth=1, facecolors=rgb, antialiased=False) #, cmap=cm.coolwarm)
    #
    # plt.show()







    # img = cv2.resize(img, [30, 30])
    #
    # fig = plt.figure(figsize=[10, 10])
    # ax = fig.subplots()
    #
    # ax.imshow(img)
    #
    # xr = np.arange(0, img.shape[1], 1)
    # yr = np.arange(0, img.shape[0], 1)
    # # xx, yy = np.meshgrid(xr, yr)
    # dy, dx = np.gradient(img, 1)
    # ax.quiver(dx, dy, angles="xy") #, headwidth = 5)
    #
    # plt.savefig("plot.png")