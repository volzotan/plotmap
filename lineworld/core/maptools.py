from dataclasses import dataclass
from enum import Enum

import pyproj
import shapely
from shapely.geometry import Polygon

EQUATOR = 40075016.68557849
EQUATOR_HALF = EQUATOR / 2.0


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
    projection: Projection = Projection.VAN_DER_GRINTEN_I
    # projection: Projection = Projection.WEB_MERCATOR
    # projection: Projection = Projection.ECKERT_IV

    wrapover: bool = True

    # units in mm
    width: float = 2000
    height: float = 1100

    offset_x: float = 0.0
    offset_y: float = 180

    tolerance: float = 0.1

    def get_transformation_matrix(self) -> list[float]:
        a = 1 / EQUATOR * self.width
        b = 0
        d = 0
        e = 1 / EQUATOR * self.width * -1  # vertical flip
        xoff = self.width / 2. + self.offset_x
        yoff = self.height / 2. + self.offset_y
        return [a, b, d, e, xoff, yoff]

    def get_viewport(self) -> Polygon:
        # return shapely.box(0, 0, self.width, self.height)
        return shapely.box(-self.width, -self.height, self.width*2, self.height*2)

    def get_projection_func(self, src_projection: Projection) -> pyproj.Transformer:
        crs_src = pyproj.CRS(f"{src_projection.value[0]}:{src_projection.value[1]}")
        crs_dst = pyproj.CRS(f"{self.projection.value[0]}:{self.projection.value[1]}")

        if self.wrapover:
            crs_dst = pyproj.CRS.from_proj4("+proj=vandg +over")

        return pyproj.Transformer.from_crs(crs_src, crs_dst, always_xy=True).transform