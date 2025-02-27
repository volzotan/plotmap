import pyproj
import shapely
from shapely.geometry import Polygon, LineString
from shapely.affinity import affine_transform

import numpy as np

from lineworld.core.svgwriter import SvgWriter


EQUATOR = 40075016.68557849

rect = Polygon(
    [
        [-(180 - 20), +(90 - 20)],
        [-(180 - 20), -(90 - 20)],
        [+(180 - 20), -(90 - 20)],
        [+(180 - 20), +(90 - 20)],
        [-(180 - 20), +(90 - 20)],
    ]
)

width: int = 1000
height: int = width


def get_transformation_matrix() -> list[float]:
    a = 1 / EQUATOR * width
    b = 0
    d = 0
    e = 1 / EQUATOR * width * -1  # vertical flip
    xoff = width / 2.0
    yoff = height / 2
    return [a, b, d, e, xoff, yoff]


def project_func2(coordinates: np.array) -> np.array:
    coords = np.copy(coordinates)

    coords[:, 0] = np.degrees(np.radians(coordinates[:, 1]) * np.cos(np.radians(coordinates[:, 0])))
    coords[:, 1] = coordinates[:, 0]

    return coords


LATITUDE_LINES = 11
LONGITUDE_LINES = 11 + 6
LAT_LON_MIN_SEGMENT_LENGTH = 0.1

crs_src = pyproj.CRS("EPSG:4326")
# crs_dst = pyproj.CRS('EPSG:3857')
crs_dst = pyproj.CRS.from_proj4("+proj=vandg +over")

# project_func = pyproj.Transformer.from_crs(crs_src, crs_dst, always_xy=True).transform
project_func = pyproj.Transformer.from_crs(crs_src, crs_dst).transform

mat = get_transformation_matrix()

lines = []

minmax_lat = [-90 - 0, 90 + 0]
minmax_lon = [-180 - 90, 180 + 90]

lats = np.linspace(*minmax_lat, num=LATITUDE_LINES).tolist()  # [1:-1]
# lons = np.linspace(-180, 180, num=LONGITUDE_LINES).tolist() #[1:-1]
lons = np.linspace(*minmax_lon, num=LONGITUDE_LINES).tolist()  # [1:-1]

for lat in lats:
    lines.append(LineString([[lat, minmax_lon[0]], [lat, minmax_lon[1]]]))

for lon in lons:
    lines.append(LineString([[minmax_lat[0], lon], [minmax_lat[1], lon]]))

lines += [shapely.box(-50, -50, 50, 50)]

lines = shapely.segmentize(lines, LAT_LON_MIN_SEGMENT_LENGTH)  # .tolist()

lines = [shapely.ops.transform(project_func, l) for l in lines]  # TODO: should not work...
# lines = [shapely.transform(l, project_func2) for l in lines]
lines = [affine_transform(l, mat) for l in lines]

geometries = lines

# for g in geometries:
#     print(g)

svg = SvgWriter("test.svg", [width, height])

options_grid = {
    "fill": "none",
    "stroke": "black",
    "stroke-width": "1.0",
}
svg.add("grid", geometries, options=options_grid)

svg.write()
