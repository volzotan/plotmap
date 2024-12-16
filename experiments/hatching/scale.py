import math
from pathlib import Path
from typing import Callable, Any

import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

LUT_SIZE = 3000

def slope_linear(lut_size=LUT_SIZE) -> float:
    lut = np.linspace(0, 1, num=lut_size)
    return lut


def slope_power(exp=2, lut_size=LUT_SIZE) -> float:
    lut = np.linspace(0, 1, num=lut_size)
    return lut**exp

    # TODO: numpy-compatible vectorized output for even exponents

    # y = (x*2-1) ** exp

    # if exp % 2 == 0:
    #     if x < 0.5:
    #         return (1-y)/2
    #     else:
    #         return y/2 + 0.5
    # else:
    #     return y


def slope_sine(lut_size=LUT_SIZE) -> float:
    lut = np.linspace(0, 1, num=lut_size)
    return -(np.cos(math.pi * lut) - 1) / 2


def quadratic_bezier(p1=[.25, .25], p2=[.75, .75], lut_size=LUT_SIZE) -> np.ndarray:

    p0 = [0, 0]
    p3 = [1, 1]

    t = np.linspace(0, 1, num=lut_size*3)

    x = ((1 - t) ** 3) * p0[0] + 3 * ((1 - t) ** 2) * t * p1[0] + 3 * (1 - t) * (t ** 2) * p2[0] + (t ** 3) * p3[0]
    y = ((1 - t) ** 3) * p0[1] + 3 * ((1 - t) ** 2) * t * p1[1] + 3 * (1 - t) * (t ** 2) * p2[1] + (t ** 3) * p3[1]

    lut = np.full([lut_size], -1, dtype=float)

    for i in range(0, x.shape[0]):
        lut[int(x[i] * (lut_size-1))] = y[i]

    if np.min(lut) == -1:
        raise Exception(f"unfilled LUT: {np.where(lut == -1)}")

    return lut


def sigmoid(lut_size=LUT_SIZE) -> float:
    lut = np.linspace(0, 1, num=lut_size)
    return 1 / (1 + np.exp(-lut))


class Scale():

    def __init__(self, func: Callable, params: dict[str, Any], num_output_bins: int | None = None) -> None:
        self.func = func
        self.params = params
        self.num_output_bins = num_output_bins
        self.lut = self.func(**self.params)

    def apply(self, values: np.ndarray) -> None:
        output = self.lut[(values * (self.lut.shape[0] - 1)).astype(int)]

        if self.num_output_bins is None:
            return output
        else:
            return np.digitize(output, np.linspace(0, 1, num=self.num_output_bins))


if __name__ == "__main__":

    NUM_X_VALUES = 100

    scales = [
        [slope_linear, {}],
        # [slope_power, {"exp": 2}],
        [slope_power, {"exp": 3}],
        [slope_sine, {}],
        [quadratic_bezier, {"p1": [0, 0.75], "p2": [1, 0.25]}],
        [quadratic_bezier, {"p1": [0, 0.50], "p2": [1, 0.50]}],
        [quadratic_bezier, {}],
        [quadratic_bezier, {"p1": [0.25, 0], "p2": [.75, 1.0]}],
        [quadratic_bezier, {"p1": [0.50, 0], "p2": [.50, 1.0]}],
        [quadratic_bezier, {"p1": [0.75, 0], "p2": [.25, 1.0]}],
        [sigmoid, {}],
    ]

    fig, axes = plt.subplots(nrows=2, ncols=len(scales))
    fig.set_figheight(2 * 5)
    fig.set_figwidth(6 * 5)

    IMAGE_FILE = Path("experiments/hatching/data/gebco_crop.tif")
    img = (cv2.imread(str(IMAGE_FILE), cv2.IMREAD_UNCHANGED)).astype(np.float64)
    img = cv2.resize(img, [1000, 1000])
    img = (img - np.min(img)) / (np.max(img) - np.min(img))


    for i in range(len(scales)):
        func = scales[i][0]
        args = scales[i][1]

        xs = np.linspace(0, 1, endpoint=True, num=NUM_X_VALUES)

        scale_obj = Scale(func, args, num_output_bins=10)
        ys = scale_obj.apply(xs) / scale_obj.num_output_bins

        ax = axes[0, i]
        ax.plot(xs, ys)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        ax.set_aspect('equal')
        ax.set_title(func.__name__ + "\n" + "".join([f" {k}: {v}" for k, v in args.items()]))

        ax = axes[1, i]
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        lut_image = scale_obj.apply(img) / scale_obj.num_output_bins
        ax.imshow(lut_image, norm=norm)

    plt.savefig(Path("experiments/hatching/output", "scales.png"))
