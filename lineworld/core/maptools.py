from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pyproj
import shapely
from shapely.geometry import Polygon

class Projection(Enum):
    WGS84 = "EPSG", 4326
    WEB_MERCATOR = "EPSG", 3857
    ECKERT_IV = "ESRI", 54012
    VAN_DER_GRINTEN_I = "ESRI", 54029


@dataclass
class Pen():
    color: list[int]
    stroke_size: float


class DocumentInfo():

    EQUATOR = 40075016.68557849

    def __init__(self, config: dict[str, Any]):
        self.config = config

        self.projection = Projection[config.get("projection", "VAN_DER_GRINTEN_I")]
        self.wrapover = config.get("wrapover", True)

        self.width = config.get("width", 1000)
        self.height = config.get("height", 1000)

        self.offset_x = config.get("offset_x", 0)
        self.offset_y = config.get("offset_y", 0)

    def get_transformation_matrix(self) -> list[float]:
        """world space to map space"""

        a = 1 / self.EQUATOR * self.width
        b = 0
        d = 0
        e = 1 / self.EQUATOR * self.width * -1  # vertical flip
        xoff = self.width / 2. + self.offset_x
        yoff = self.height / 2. + self.offset_y
        return [a, b, d, e, xoff, yoff]

    def get_transformation_matrix_raster_to_map(self, raster_width: int, raster_height: int) -> list[float]:
        """raster space to map space"""

        a = (1 / raster_width) * self.width
        b = 0
        d = 0
        e = (1 / raster_height) * self.width
        xoff = self.offset_x

        yoff = self.offset_y
        yoff = -(self.width-self.height)/2 + self.offset_y

        return [a, b, d, e, xoff, yoff]

    def get_transformation_matrix_map_to_raster(self, raster_width: int, raster_height: int) -> list[float]:
        a, b, d, e, xoff, yoff = self.get_transformation_matrix_raster_to_map(raster_width, raster_height)
        mat = np.matrix([[a, b, xoff], [d, e, yoff], [0, 0, 1]])
        mat_inv = np.linalg.inv(mat)
        return [float(e) for e in [mat_inv[0, 0], mat_inv[0, 1], mat_inv[1, 0], mat_inv[1, 1], mat_inv[0, 2], mat_inv[1, 2]]]

    def get_transformation_matrix_font(self, xoff: float, yoff: float) -> list[float]:
        a = 1
        b = 0
        d = 0
        e = -1
        return [a, b, d, e, xoff, yoff]

    def get_viewport(self) -> Polygon:
        if self.config.get("debug", False):
            return shapely.box(-self.width, -self.height, self.width*2, self.height*2)
        else:
            return shapely.box(0, 0, self.width, self.height)

    def get_projection_func(self, src_projection: Projection) -> Any:
        crs_src = pyproj.CRS(f"{src_projection.value[0]}:{src_projection.value[1]}")
        crs_dst = pyproj.CRS(f"{self.projection.value[0]}:{self.projection.value[1]}")

        if self.wrapover:
            crs_dst = pyproj.CRS.from_proj4("+proj=vandg +over")

        return pyproj.Transformer.from_crs(crs_src, crs_dst, always_xy=True).transform
