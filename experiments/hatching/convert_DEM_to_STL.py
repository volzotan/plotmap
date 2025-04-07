import argparse
import textwrap

import cv2
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh

SAMPLING_RATE = 10  # per unit
Z_HEIGHT = 1


def coord_to_ind(x, y, dim, sampling):
    return y * dim[0] * sampling + x


ap = argparse.ArgumentParser(description="Generate mesh file from DEM/DET images")
ap.add_argument("-i", "--input", help="input file")
ap.add_argument("-x", type=int, default=100, help="width")
ap.add_argument("-y", type=int, default=100, help="depth")
ap.add_argument("-z", type=float, default=Z_HEIGHT, help="height")
ap.add_argument(
    "-s",
    "--sampling-rate",
    type=int,
    default=SAMPLING_RATE,
    help="number of points per unit (width/height) [int]",
)
ap.add_argument("--blur", type=int, default=0, help="blur input image kernel size")
ap.add_argument("--output-image", default=None, help="output image filename")
ap.add_argument("--output-xyz", default=None, help="output pointcloud filename")
ap.add_argument("--output-stl", default=None, help="output STL filename")
ap.add_argument("--output-ply", default=None, help="output PLY filename")
ap.add_argument("--cutoff", action="store_true", default=False, help="cut off positive values")
ap.add_argument(
    "--surface-only",
    action="store_true",
    default=False,
    help="for point cloud coordinates do not extrude the volume",
)

args = vars(ap.parse_args())

DIMENSIONS = [args["x"], args["y"]]
BLOCK_HEIGHT = args["z"] * 2

data = None
try:
    cv2.imread(args["input"], cv2.IMREAD_UNCHANGED)
except cv2.error:
    import rasterio

    with rasterio.open(args["input"]) as dataset:
        data = dataset.read(1)

# normalize
data = (data - np.min(data)) / np.ptp(data)

# CROP
CROP_CENTER = [0.4, 0.4]
CROP_SIZE = [15000, 15000]
data = data[
    int(CROP_CENTER[1] * data.shape[1] - CROP_SIZE[1] // 2) : int(CROP_CENTER[1] * data.shape[1] + CROP_SIZE[1] // 2),
    int(CROP_CENTER[0] * data.shape[0] - CROP_SIZE[0] // 2) : int(CROP_CENTER[0] * data.shape[0] + CROP_SIZE[0] // 2),
]

# resize so that 1 pixel equals 1 sampling point
res = cv2.resize(data, [DIMENSIONS[1] * SAMPLING_RATE, DIMENSIONS[0] * SAMPLING_RATE])
if args["blur"] > 0:
    res = cv2.blur(res, (args["blur"], args["blur"]))

res = np.flip(res, axis=0)

res_min = np.min(res)
res_max = np.max(res)
res = np.multiply(res, args["z"])

# cut off the hills, keep the valleys
if args["cutoff"]:
    res[res[:, :] > 0] = 0

if args["output_image"]:
    plt.imsave(
        args["output_image"],
        res,
        vmin=-1,  # -math.sqrt(2),
        vmax=1,  # +math.sqrt(2),
        origin="upper",
    )

if args["output_ply"]:
    s = args["sampling_rate"]
    num_vertices = (DIMENSIONS[0] * s) * (DIMENSIONS[1] * s)
    num_faces = (DIMENSIONS[0] * s - 1) * (DIMENSIONS[1] * s - 1) * 2

    with open(args["output_ply"], "w") as f:
        data = """
                ply
                format ascii 1.0
                element vertex {}
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                element face {}
                property list uchar int vertex_indices
                end_header
                """

        data = textwrap.dedent(data[1:])  # remove first newline (for dedent to work)
        data = data.format(num_vertices, num_faces)

        vertices = []
        faces = []

        f.write(data)

        for y in range(0, DIMENSIONS[1] * s):
            for x in range(0, DIMENSIONS[0] * s):
                pos = (res[y, x] / args["z"] - res_min) / (res_max - res_min)
                # c = colormap._viridis_data[int(pos * (len(colormap._viridis_data) - 1))]
                c = [pos, pos, pos]
                c = [int(x * 255) for x in c]

                f.write("{:.3f} {:.3f} {:.3f} {:d} {:d} {:d}\n".format(x / s, y / s, res[y, x], *c))

        for y in range(0, DIMENSIONS[1] * s - 1):
            for x in range(0, DIMENSIONS[0] * s - 1):
                f.write(
                    "3 {} {} {}\n".format(
                        coord_to_ind(x, y, DIMENSIONS, s),
                        coord_to_ind(x + 1, y, DIMENSIONS, s),
                        coord_to_ind(x, y + 1, DIMENSIONS, s),
                    )
                )

                f.write(
                    "3 {} {} {}\n".format(
                        coord_to_ind(x + 1, y, DIMENSIONS, s),
                        coord_to_ind(x + 1, y + 1, DIMENSIONS, s),
                        coord_to_ind(x, y + 1, DIMENSIONS, s),
                    )
                )

        f.write("\n")

if args["output_stl"]:
    s = args["sampling_rate"]

    num_faces = (
        (DIMENSIONS[0] * s - 1) * (DIMENSIONS[1] * s - 1) * 4 + (DIMENSIONS[0] * s) * 4 + (DIMENSIONS[1] * s) * 4
    )

    obj = mesh.Mesh(np.zeros(num_faces, dtype=mesh.Mesh.dtype))
    count = 0

    for x in range(0, DIMENSIONS[0] * s - 1):
        for y in range(0, DIMENSIONS[1] * s - 1):
            obj.vectors[count][0] = [x / s, y / s, res[y, x]]
            obj.vectors[count][1] = [(x + 1) / s, y / s, res[y, x + 1]]
            obj.vectors[count][2] = [x / s, (y + 1) / s, res[y + 1, x]]

            count += 1

            obj.vectors[count][0] = [(x + 1) / s, y / s, res[y, x + 1]]
            obj.vectors[count][1] = [(x + 1) / s, (y + 1) / s, res[y + 1, x + 1]]
            obj.vectors[count][2] = [x / s, (y + 1) / s, res[y + 1, x]]

            count += 1

    if not args["surface_only"]:
        # side T/B

        for y in [0, DIMENSIONS[1] * s - 1]:
            for x in range(0, DIMENSIONS[0] * s - 1):
                obj.vectors[count][0] = [x / s, y / s, -BLOCK_HEIGHT]
                obj.vectors[count][1] = [(x + 1) / s, y / s, -BLOCK_HEIGHT]
                obj.vectors[count][2] = [x / s, y / s, res[y, x]]

                count += 1

                obj.vectors[count][0] = [(x + 1) / s, y / s, -BLOCK_HEIGHT]
                obj.vectors[count][1] = [(x + 1) / s, y / s, res[y, x + 1]]
                obj.vectors[count][2] = [x / s, y / s, res[y, x]]

                count += 1

        # side L/R

        for x in [0, DIMENSIONS[0] * s - 1]:
            for y in range(0, DIMENSIONS[0] * s - 1):
                obj.vectors[count][0] = [x / s, y / s, -BLOCK_HEIGHT]
                obj.vectors[count][1] = [x / s, (y + 1) / s, -BLOCK_HEIGHT]
                obj.vectors[count][2] = [x / s, y / s, res[y, x]]

                count += 1

                obj.vectors[count][0] = [x / s, (y + 1) / s, -BLOCK_HEIGHT]
                obj.vectors[count][1] = [x / s, (y + 1) / s, res[y + 1, x]]
                obj.vectors[count][2] = [x / s, y / s, res[y, x]]

                count += 1

        # bottom

        for x in range(0, DIMENSIONS[0] * s - 1):
            for y in range(0, DIMENSIONS[1] * s - 1):
                obj.vectors[count][0] = [x / s, y / s, -BLOCK_HEIGHT]
                obj.vectors[count][1] = [(x + 1) / s, y / s, -BLOCK_HEIGHT]
                obj.vectors[count][2] = [x / s, (y + 1) / s, -BLOCK_HEIGHT]

                count += 1

                obj.vectors[count][0] = [(x + 1) / s, y / s, -BLOCK_HEIGHT]
                obj.vectors[count][1] = [(x + 1) / s, (y + 1) / s, -BLOCK_HEIGHT]
                obj.vectors[count][2] = [x / s, (y + 1) / s, -BLOCK_HEIGHT]

                count += 1

    obj.save(args["output_stl"])

if args["output_xyz"]:
    with open(args["output_xyz"], "w") as f:
        for i in range(0, DIMENSIONS[1] * args["sampling_rate"]):
            for j in range(0, DIMENSIONS[0] * args["sampling_rate"]):
                f.write("{} {} {}\n".format(j / args["sampling_rate"], i / args["sampling_rate"], res[i, j]))

        if not args["surface_only"]:
            ## additional points
            # side L/R

            for i in range(0, DIMENSIONS[1] * args["sampling_rate"]):
                for j in [0, DIMENSIONS[0] * args["sampling_rate"] - 1]:
                    zs = np.linspace(
                        -BLOCK_HEIGHT,
                        res[i, j] * args["z"],
                        int(BLOCK_HEIGHT * args["sampling_rate"]),
                        endpoint=False,
                    )
                    for z in zs:
                        f.write("{} {} {}\n".format(j / args["sampling_rate"], i / args["sampling_rate"], z))

            # side T/B

            for i in [0, DIMENSIONS[1] * args["sampling_rate"] - 1]:
                for j in range(0, DIMENSIONS[0] * args["sampling_rate"]):
                    zs = np.linspace(
                        -BLOCK_HEIGHT,
                        res[i, j] * args["z"],
                        int(BLOCK_HEIGHT * args["sampling_rate"]),
                        endpoint=False,
                    )
                    for z in zs:
                        f.write("{} {} {}\n".format(j / args["sampling_rate"], i / args["sampling_rate"], z))

            # bottom plane

            for j in linx[1:]:
                for i in liny[1:]:
                    f.write("{} {} {}\n".format(j / args["ppu"], i / args["ppu"], -BLOCK_HEIGHT))
