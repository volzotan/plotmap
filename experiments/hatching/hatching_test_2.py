from pathlib import Path

import cv2

import numpy as np
from matplotlib import pyplot as plt
import numpy as np

INPUT_FILE = Path("data/hatching_dem.tif")
# INPUT_FILE = Path("data/gebco_crop.tif")

def read_data(input_path: Path) -> np.ndarray:
    data = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    # data = cv2.resize(img, [30, 30])

    # data = np.flipud(data)
    # data = (data * 120/20).astype(np.int8)
    # data = np.rot90(data)

    return data


def get_slope(data: np.ndarray, sampling_step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sampling_step = 10
    test_slice = data[::sampling_step, ::sampling_step]
    r, c = np.shape(data)
    Y, X = np.mgrid[0:r:sampling_step, 0:c:sampling_step]
    dY, dX = np.gradient(test_slice)  # order! Y X

    return (X, Y, dX, dY)


if __name__ == "__main__":

    data = read_data(INPUT_FILE)

    print(f"data {INPUT_FILE} min: {np.min(data)} / max: {np.max(data)}")

    X, Y, dX, dY = get_slope(data, 10)


# fig = plt.figure(figsize=[20, 20])
# ax = fig.subplots()
# ax.imshow(img)
# ax.quiver(X, Y, dX, dY, angles="xy", color='r')
# plt.savefig("plot.png")

extent = max([abs(np.min(img)), abs(np.max(img))])
slope_img = ((np.abs(dX) + np.abs(dY)) * 255/extent).astype(np.uint8)
slope_img = cv2.resize(slope_img, [1000, 1000], interpolation=cv2.INTER_NEAREST)
cv2.imwrite("slope.png", slope_img)


from matplotlib import cm
from matplotlib.colors import LightSource

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
# ax.set_zlim(-20000, 20000)
ax.set_zlim(-0, 50)

Z = test_slice

slope = np.abs(dX) + np.abs(dY)

thres = np.max(slope)*0.5
slope[slope > thres] = thres

ls = LightSource(270, 45)

rgb = ls.shade(slope, cmap=cm.cool, vert_exag=0.1, blend_mode='soft')
# rgb = ls.shade(Z, cmap=cm.bwr, vert_exag=0.1, blend_mode='soft')

surf = ax.plot_surface(X, Y, Z, rcount=100, ccount=100, linewidth=1, facecolors=rgb, antialiased=False) #, cmap=cm.coolwarm)

plt.show()




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