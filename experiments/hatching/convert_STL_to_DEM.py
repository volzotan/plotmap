import numpy as np
import trimesh
from PIL import Image

STL_PATH = "hatching_dem.stl"
OUTPUT_PATH = "data/hatching_dem.tif"
# OUTPUT_PATH = "data/hatching_dem.png"

INP_DIMENSIONS = [100, 100]
OUT_DIMENSIONS = [1000, 1000]

mesh = trimesh.load_mesh(STL_PATH)

scaler = [
    OUT_DIMENSIONS[0] / INP_DIMENSIONS[0],
    OUT_DIMENSIONS[1] / INP_DIMENSIONS[1],
    1 # 255 / mesh.bounds[1, 2]
]

xs = np.linspace(0, INP_DIMENSIONS[0], num=OUT_DIMENSIONS[0], endpoint=False)
ys = np.linspace(0, INP_DIMENSIONS[1], num=OUT_DIMENSIONS[1], endpoint=False)

xv, yv = np.meshgrid(xs, ys)
zv = np.zeros_like(xv)
zv.fill(mesh.bounds[1, 2] * 1.10)

ray_origins = np.dstack((xv, yv, zv)).reshape([xs.shape[0]*ys.shape[0], 3])
ray_directions = np.tile(np.array([0, 0, -100]), (ray_origins.shape[0], 1))

locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions)

output = np.zeros([*OUT_DIMENSIONS], dtype=np.float32)
# output = np.zeros([*OUT_DIMENSIONS], dtype=np.uint8)

for loc in locations:
    # switch row col / Y axis flip
    output[(OUT_DIMENSIONS[1]-1)-int(loc[1] * scaler[1]), int(loc[0] * scaler[0])] = loc[2] * scaler[2]

if OUTPUT_PATH.lower().endswith(".png"):
    output *= 255.0 / mesh.bounds[1, 2]
    output = output.astype(np.uint8)
    im = Image.fromarray(output)
    im.save(OUTPUT_PATH, "PNG")

if OUTPUT_PATH.lower().endswith(".tif"):
    im = Image.fromarray(output, mode='F') # float32
    im.save(OUTPUT_PATH, "TIFF")


# ray_visualize = trimesh.load_path(
#     np.hstack((ray_origins, ray_origins + ray_directions * 5.0)).reshape(-1, 2, 3)
# )
#
# # unmerge so viewer doesn't smooth
# mesh.unmerge_vertices()
# # make mesh white- ish
# mesh.visual.face_colors = [255, 255, 255, 255]
# mesh.visual.face_colors[index_tri] = [255, 0, 0, 255]
#
# scene = trimesh.Scene([mesh, ray_visualize])
#
# scene.show()