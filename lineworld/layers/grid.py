from dataclasses import dataclass

import geoalchemy2
import shapely
from core.maptools import DocumentInfo, Projection
from geoalchemy2.shape import from_shape, to_shape
from layers.layer import Layer
from shapely import Point, envelope
from shapely.affinity import affine_transform, translate
from sqlalchemy import MetaData
from sqlalchemy import Table, Column, Integer
from sqlalchemy import engine
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy import text

from lineworld.util.geometrytools import *

from HersheyFonts import HersheyFonts

import numpy as np

@dataclass
class GridMapLines():
    id: int | None
    lines: MultiLineString

    def todict(self) -> dict[str, int | float | str | None]:
        return {
            "lines": str(from_shape(self.lines))
        }


class Grid(Layer):

    DATA_SRID = Projection.WGS84

    LAT_LON_MIN_SEGMENT_LENGTH = 1e-1

    def __init__(self, layer_label: str, db: engine.Engine) -> None:
        super().__init__(layer_label, db)

    def extract(self) -> None:
        pass

    def transform(self) -> None:
        pass

    def load(self, geometries: list[GridMapLines]) -> None:

        if geometries is None:
            return

        if len(geometries) == 0:
            logger.warning("no geometries to load. abort")
            return
        else:
            logger.info(f"loading geometries: {len(geometries)}")

        with self.db.begin() as conn:
            result = conn.execute(text(f"TRUNCATE TABLE {self.map_lines_table.fullname} CASCADE"))
            result = conn.execute(insert(self.map_lines_table), [g.todict() for g in geometries])

    def _get_gridminmax(self, document_info: DocumentInfo) -> tuple[list[float], list[float]]:

        minmax_lat = [-90, 90]
        minmax_lon = [-180, 180]

        if document_info.wrapover:
            minmax_lon = [-180 - 90, 180 + 90]

        return minmax_lat, minmax_lon

    def _get_gridpositions(self, document_info: DocumentInfo, distance_lat_lines: float, distance_lon_lines: float) -> tuple[list[float], list[float]]:

        minmax_lat, minmax_lon = self._get_gridminmax(document_info)

        lons = [x * distance_lat_lines for x in range(1, minmax_lon[1]//distance_lat_lines)]
        lons = [x * -1 for x in lons] + [0] + lons

        lats = [x * distance_lon_lines for x in range(1, minmax_lat[1]//distance_lon_lines)]
        lats = [x * -1 for x in lats] + [0] + lats

        return lats, lons

    def _get_gridlines(self, document_info: DocumentInfo, distance_lat_lines: float, distance_lon_lines: float) -> list[GridMapLines]:

        lines = []

        project_func = document_info.get_projection_func(self.DATA_SRID)
        mat = document_info.get_transformation_matrix()

        minmax_lat, minmax_lon = self._get_gridminmax(document_info)
        lats, lons = self._get_gridpositions(document_info, distance_lat_lines, distance_lon_lines)

        for lat in lats:
            lines.append(LineString([
                [minmax_lon[0], lat],
                [minmax_lon[1], lat]
            ]))

        for lon in lons:
            lines.append(LineString([
                [lon, minmax_lat[0]],
                [lon, minmax_lat[1]]
            ]))

        lines = shapely.segmentize(lines, self.LAT_LON_MIN_SEGMENT_LENGTH)

        lines = [shapely.ops.transform(project_func, l) for l in lines]
        lines = [affine_transform(l, mat) for l in lines]

        return [GridMapLines(None, line) for line in lines]


    def project(self, document_info: DocumentInfo) -> list[GridMapLines]:
        pass

    def draw(self, document_info: DocumentInfo) -> None:
        pass

    def style(self, polygons: np.ndarray, document_info: DocumentInfo) -> None:
        pass

    def out(self, exclusion_zones: MultiPolygon, document_info: DocumentInfo) -> tuple[list[shapely.Geometry], MultiPolygon]:
        raise NotImplementedError("Must override method")


class GridBathymetry(Grid):

    LAYER_NAME = "GridBathymetry"

    LATITUDE_LINE_DIST = 40
    LONGITUDE_LINE_DIST = 40

    EXCLUDE_BUFFER = 0.5

    def __init__(self, layer_label: str, db: engine.Engine) -> None:
        super().__init__(layer_label, db)

        metadata = MetaData()

        self.map_lines_table = Table(
            "gridbathymetry_map_lines", metadata,
            Column("id", Integer, primary_key=True),
            Column("lines", geoalchemy2.Geometry("MULTILINESTRING"), nullable=False)
        )

        metadata.create_all(self.db)

    def project(self, document_info: DocumentInfo) -> list[GridMapLines]:
        return self._get_gridlines(document_info, self.LATITUDE_LINE_DIST, self.LONGITUDE_LINE_DIST)

    def out(self, exclusion_zones: MultiPolygon, document_info: DocumentInfo) -> tuple[
        list[shapely.Geometry], MultiPolygon]:
        """
        Returns (drawing geometries, exclusion polygons)
        """

        drawing_geometries = []
        with self.db.begin() as conn:
            result = conn.execute(select(self.map_lines_table))
            drawing_geometries = [to_shape(row.lines) for row in result]

        # extend extrusion zones
        cutting_tool = shapely.unary_union(np.array(drawing_geometries))
        cutting_tool = cutting_tool.buffer(self.EXCLUDE_BUFFER)
        cutting_tool = shapely.simplify(cutting_tool, document_info.tolerance)
        exclusion_zones = shapely.union(exclusion_zones, cutting_tool)

        return ([], exclusion_zones)

class GridLabels(Grid):

    LAYER_NAME = "GridLabels"

    LATITUDE_LINE_DIST = 20
    LONGITUDE_LINE_DIST = 20

    EXCLUDE_BUFFER = 2.0

    FONT_SIZE = 8

    OFFSET_TOP = 15

    def __init__(self, layer_label: str, db: engine.Engine) -> None:
        super().__init__(layer_label, db)

        metadata = MetaData()

        self.map_lines_table = Table(
            "gridlabels_map_lines", metadata,
            Column("id", Integer, primary_key=True),
            Column("lines", geoalchemy2.Geometry("MULTILINESTRING"), nullable=False)
        )

        metadata.create_all(self.db)

        self.hfont = HersheyFonts()
        self.hfont.load_default_font("futural")
        self.hfont.normalize_rendering(self.FONT_SIZE)

        # hfont_large = HersheyFonts()
        # hfont_large.load_default_font("futuram")
        # hfont_large.normalize_rendering(FONT_SIZE_LARGE)

    def _get_text(self, font, text):

        lines_raw = font.lines_for_text(text)
        # lines_restructured = []
        # for (x1, y1), (x2, y2) in lines_raw:
        #     lines_restructured.append([[x1, y1], [x2, y2]])
        # lines = MultiLineString(lines_restructured)

        return MultiLineString([[[x1, y1], [x2, y2]] for (x1, y1), (x2, y2) in lines_raw])

    def project(self, document_info: DocumentInfo) -> list[GridMapLines]:

        gridlines = self._get_gridlines(document_info, self.LATITUDE_LINE_DIST, self.LONGITUDE_LINE_DIST)
        lats, lons = self._get_gridpositions(document_info, self.LATITUDE_LINE_DIST, self.LONGITUDE_LINE_DIST)

        project_func = document_info.get_projection_func(self.DATA_SRID)
        mat = document_info.get_transformation_matrix()

        labels = []
        for lon in lons:
            lines = self._get_text(self.hfont, f"{lon}")

            lon_line = LineString([[lon, -90], [lon, +90]])
            lon_line = shapely.segmentize(lon_line, self.LAT_LON_MIN_SEGMENT_LENGTH)

            lon_line = shapely.ops.transform(project_func, lon_line)
            lon_line = affine_transform(lon_line, mat)

            intersect_line = LineString([[0, self.OFFSET_TOP], [document_info.width, self.OFFSET_TOP]])
            intersect_point = lon_line.intersection(intersect_line)

            if intersect_point is None or intersect_point.is_empty:
                logger.warning("failed computing position for label")
                continue

            center_offset = -envelope(lines).centroid.x

            mat_font = document_info.get_transformation_matrix_font(xoff=intersect_point.x+center_offset, yoff=intersect_point.y)
            lines = affine_transform(lines, mat_font)

            labels.append(GridMapLines(None, lines))

        # return gridlines + labels
        return labels

    def out(self, exclusion_zones: MultiPolygon, document_info: DocumentInfo) -> tuple[
        list[shapely.Geometry], MultiPolygon]:
        """
        Returns (drawing geometries, exclusion polygons)
        """

        drawing_geometries = []
        with self.db.begin() as conn:
            result = conn.execute(select(self.map_lines_table))
            drawing_geometries = [to_shape(row.lines) for row in result]

        # remove extrusion zones
        drawing_geometries_cut = []
        for g in drawing_geometries:
            drawing_geometries_cut.append(shapely.difference(g, exclusion_zones))

        # extend extrusion zones
        cutting_tool = shapely.unary_union(np.array(drawing_geometries))
        cutting_tool = cutting_tool.buffer(self.EXCLUDE_BUFFER)
        cutting_tool = shapely.simplify(cutting_tool, document_info.tolerance)
        exclusion_zones = shapely.union(exclusion_zones, cutting_tool)

        return (drawing_geometries_cut, exclusion_zones)