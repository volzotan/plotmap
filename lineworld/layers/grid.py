from dataclasses import dataclass

import geoalchemy2
import shapely
from core.maptools import DocumentInfo, Projection
from geoalchemy2.shape import from_shape, to_shape
from layers.layer import Layer
from shapely.affinity import affine_transform
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


    def _create_gridlines(self, document_info: DocumentInfo,
                          num_lat_lines: int, num_lon_lines: int,
                          minmax_lat: list[float], minmax_lon: list[float]) -> list[GridMapLines]:

        if num_lat_lines % 2 == 0 or num_lon_lines % 2 == 0:
            logger.warning("LATITUDE_LINES and LONGITUDE_LINES need to be an odd number to create null meridian grid lines")

        lines = []

        project_func = document_info.get_projection_func(self.DATA_SRID)
        mat = document_info.get_transformation_matrix()

        lats = np.linspace(*minmax_lat, num=num_lat_lines).tolist()#[1:-1]
        lons = np.linspace(*minmax_lon, num=num_lon_lines).tolist()#[1:-1]

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

    LATITUDE_LINES = 11
    LONGITUDE_LINES = 17

    MINMAX_LAT = [-90 - 0, 90 + 0]
    MINMAX_LON = [-180 - 90, 180 + 90]

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
        return self._create_gridlines(
            document_info,
            self.LATITUDE_LINES, self.LONGITUDE_LINES,
            self.MINMAX_LAT, self.MINMAX_LON
        )

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
        # for g in drawing_geometries:
        #     drawing_geometries_cut.append(shapely.difference(g, exclusion_zones))

        # extend extrusion zones
        cutting_tool = shapely.unary_union(np.array(drawing_geometries))
        cutting_tool = cutting_tool.buffer(self.EXCLUDE_BUFFER)
        exclusion_zones = shapely.union(exclusion_zones, cutting_tool)

        return (drawing_geometries_cut, exclusion_zones)

class GridLabels(Grid):

    LAYER_NAME = "GridLabels"

    LATITUDE_LINES = 11
    LONGITUDE_LINES = 17

    MINMAX_LAT = [-90 - 0, 90 + 0]
    MINMAX_LON = [-180 - 90, 180 + 90]

    EXCLUDE_BUFFER = 0.5

    FONT_SIZE = 13

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
        lines = self._create_gridlines(
            document_info,
            self.LATITUDE_LINES, self.LONGITUDE_LINES,
            self.MINMAX_LAT, self.MINMAX_LON
        )

        text_lines = self._get_text(self.hfont, "FOO")

        return [GridMapLines(None, text_lines)]

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
        exclusion_zones = shapely.union(exclusion_zones, cutting_tool)

        return (drawing_geometries_cut, exclusion_zones)